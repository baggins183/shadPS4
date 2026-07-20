// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/div_ceil.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ordered_count.h"

namespace Shader::Backend::SPIRV {

using namespace OrderedCount;

namespace {

void SyncWorkgroupBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory = spv::Scope::Workgroup;
    spv::MemorySemanticsMask memory_semantics =
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory;
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

} // Anonymous namespace

// Grab a workgroup index by bumping a counter "next_emulated_workgroup_index"
// used by all dispatched workgroups.
// This is to guarantee forward progress in a shader that uses DS_ORDERED_COUNT,
// because workgroups aren't dispatched in particular order according to the actual
// gl_WorkgroupId, but each workgroup must wait for all lower-numbered ones to be counted.
// This may not be enough in theory if workgroups can be unscheduled, blocking on
// progress of others, after they've started executing.
void EmitContext::InitEmulatedWorkgroupIndex() {
    const auto& scratch_buffer{buffers[ordered_count_scratch_index]};
    const auto [scratch_buffer_id, pointer_type] = scratch_buffer.Alias(PointerType::U32);

    const Id emulated_block_id_assignment_ptr{
        OpAccessChain(pointer_type, scratch_buffer_id,
                      ConstU32(ScratchBufferOffsets::NextEmulatedWorkgroupIndex))};

    const Id device_scope{ConstU32(static_cast<u32>(spv::Scope::Device))};
    workgroup_index_id = OpAtomicIAdd(U32[1], emulated_block_id_assignment_ptr, device_scope,
                                      u32_zero_value, u32_one_value);
}

void EmitContext::InitEmulatedWorkgroupId() {
    const Id num_workgroups{OpLoad(U32[3], num_workgroups_id)};
    const Id num_workgroups_x{OpCompositeExtract(U32[1], num_workgroups, 0)};
    const Id num_workgroups_y{OpCompositeExtract(U32[1], num_workgroups, 1)};

    const Id id_x{OpUMod(U32[1], workgroup_index_id, num_workgroups_x)};
    const Id id_y{
        OpUMod(U32[1], OpUDiv(U32[1], workgroup_index_id, num_workgroups_x), num_workgroups_y)};
    const Id id_z{
        OpUDiv(U32[1], workgroup_index_id, OpIMul(U32[1], num_workgroups_x, num_workgroups_y))};

    emulated_workgroup_id_id = OpCompositeConstruct(U32[3], id_x, id_y, id_z);
}

// GLSL reference:
//
// # subgroups in the thread block must be <= the number of threads in
// a subgroup (in particular subgroup 0) for this to work,
// otherwise this needs more code. subgroup size depends on gpu/driver,
// and could be situational.
//
// uint orderedCount(uint block_index, uint packer_id, bool is_active) {
//     const uvec4 cond_mask = subgroupBallot(is_active);
//     const uint subgroup_count = subgroupBallotBitCount(cond_mask);
//     // TODO faster to derive count from exclusive scan ^ ?
//     const uint my_count_offset = subgroupBallotExclusiveBitCount(cond_mask);
//
//     subgroup_counts[gl_SubgroupID] = subgroup_count;
//     barrier();
//
//     // TODO do we need to enable maximal reconvergence?
//     if (gl_SubgroupID == 0) {
//         // gl_NumSubgroups must be less than gl_SubgroupSize
//         // or else this needs more work
//
//         const uint my_count = gl_LocalInvocationIndex < gl_NumSubgroups ?
//                 subgroup_counts[gl_LocalInvocationIndex] :
//                 0;
//         const uint my_scan = subgroupExclusiveAdd(my_count);
//
//         if (gl_LocalInvocationIndex < gl_NumSubgroups) {
//             subgroup_counts[gl_LocalInvocationIndex] = my_scan;
//         }
//     }
//
//     barrier();
//
//     if (gl_SubgroupID == gl_NumSubgroups - 1 && gl_SubgroupInvocationID == 0) {
//         for (;;) {
//             uint val = atomicLoad(next_block_owner_id, gl_ScopeDevice, gl_StorageSemanticsBuffer,
//             gl_SemanticsAcquire); if (val == block_index) {
//                 uint block_count = subgroup_count + subgroup_counts[gl_SubgroupID];
//                 scratch = atomicAdd(global_count, block_count);
//                 atomicStore(next_block_owner_id, block_index + 1, gl_ScopeDevice,
//                 gl_StorageSemanticsBuffer, gl_SemanticsRelease); break;
//             }
//         }
//     }
//
//     barrier();
//
//     return scratch + subgroup_counts[gl_SubgroupID] + my_count_offset;
// }
Id EmitContext::DefineOrderedCountFunction() {
    // uint orderedCount(uint block_index, uint packer_id, bool is_active)
    const auto func_type{TypeFunction(U32[1], U32[1], U32[1], U1[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto block_index{OpFunctionParameter(U32[1])};
    Name(block_index, "block_index");
    const auto packer_id{OpFunctionParameter(U32[1])};
    Name(packer_id, "packer_id");
    const auto is_active{OpFunctionParameter(U1[1])};
    Name(is_active, "is_active");

    Name(func, "ordered_count_function");
    AddLabel();

    ASSERT_MSG(profile.subgroup_size > max_num_subgroups,
               "more subgroups ({}) than subgroup size ({}), unimplemented (DS_ORDERED_COUNT "
               "implementation)",
               max_num_subgroups, profile.subgroup_size);
    ASSERT(info.num_ordered_count_packers == 1);

    const Id shared_u32_ptr{TypePointer(spv::StorageClass::Workgroup, U32[1])};

    const Id shared_subgroup_counts_ptr{OpAccessChain(
        TypePointer(spv::StorageClass::Workgroup, ordered_count_subgroup_counts_array_type),
        ordered_count_scratch_mem_variable, u32_zero_value)};
    const Id shared_scratch_ptr{
        OpAccessChain(shared_u32_ptr, ordered_count_scratch_mem_variable, u32_zero_value)};

    const Id subgroup_id_val{OpLoad(U32[0], subgroup_id)};
    const Id subgroup_local_invocation_id_val{OpLoad(U32[0], subgroup_local_invocation_id)};
    const Id num_subgroups_val{OpLoad(U32[0], num_subgroups)};
    const Id local_invocation_index_val{OpLoad(U32[0], local_invocation_index)};

    const auto& scratch_buffer{buffers[ordered_count_scratch_index]};
    const auto [scratch_buffer_id, pointer_type] = scratch_buffer.Alias(PointerType::U32);

    const Id subgroup_scope{ConstU32(static_cast<u32>(spv::Scope::Subgroup))};
    const Id device_scope{ConstU32(static_cast<u32>(spv::Scope::Device))};
    const Id acquire_semantics{ConstU32(static_cast<u32>(spv::MemorySemanticsMask::Acquire))};
    const Id release_semantics{ConstU32(static_cast<u32>(spv::MemorySemanticsMask::Release))};

    const Id cond_mask{OpSubgroupBallotKHR(U32[4], is_active)};
    const Id subgroup_count{OpGroupNonUniformBallotBitCount(
        U32[1], subgroup_scope, spv::GroupOperation::Reduce, cond_mask)};
    const Id my_count_offset{OpGroupNonUniformBallotBitCount(
        U32[1], subgroup_scope, spv::GroupOperation::ExclusiveScan, cond_mask)};

    const Id subgroup_counts_at_subgroup_id{
        OpAccessChain(shared_u32_ptr, shared_subgroup_counts_ptr, subgroup_id_val)};
    OpStore(subgroup_counts_at_subgroup_id, subgroup_count);
    SyncWorkgroupBarrier(*this);

    const auto scan_label{OpLabel()};
    const auto scan_merge_label{OpLabel()};

    const Id is_first_subgroup{OpIEqual(U1[1], subgroup_id_val, u32_zero_value)};

    OpSelectionMerge(scan_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(is_first_subgroup, scan_label, scan_merge_label);

    AddLabel(scan_label);

    const Id in_scan{OpULessThan(U1[1], local_invocation_index_val, num_subgroups_val)};

    const auto read_smem_label{OpLabel()};
    const auto read_smem_merge_label{OpLabel()};

    OpSelectionMerge(read_smem_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(in_scan, read_smem_label, read_smem_merge_label);

    AddLabel(read_smem_label);
    const Id my_count_0{OpLoad(U32[1], OpAccessChain(shared_u32_ptr, shared_subgroup_counts_ptr,
                                                     local_invocation_index_val))};
    OpBranch(read_smem_merge_label);

    AddLabel(read_smem_merge_label);
    const Id my_count{OpPhi(U32[1], my_count_0, read_smem_label, u32_zero_value, scan_label)};

    const Id my_scan{OpGroupNonUniformIAdd(U32[1], subgroup_scope,
                                           spv::GroupOperation::ExclusiveScan, my_count)};

    const auto write_smem_label{OpLabel()};
    const auto write_smem_merge_label{OpLabel()};

    OpSelectionMerge(write_smem_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(in_scan, write_smem_label, write_smem_merge_label);

    AddLabel(write_smem_label);
    OpStore(OpAccessChain(shared_u32_ptr, shared_subgroup_counts_ptr, local_invocation_index_val),
            my_scan);
    OpBranch(write_smem_merge_label);

    AddLabel(write_smem_merge_label);

    OpBranch(scan_merge_label);

    AddLabel(scan_merge_label);

    SyncWorkgroupBarrier(*this);

    const auto last_subgroup_label{OpLabel()};
    const auto loop_header_label{OpLabel()};
    const auto poll_success_label{OpLabel()};
    const auto poll_failed_label{OpLabel()};
    const auto loop_continue_label{OpLabel()};
    const auto end_merge_label{OpLabel()};

    const Id last_subgroup_id{OpISub(U32[1], num_subgroups_val, u32_one_value)};
    const Id is_last_subgroup{OpIEqual(U1[1], subgroup_id_val, last_subgroup_id)};
    const Id is_first_lane{OpIEqual(U1[1], subgroup_local_invocation_id_val, u32_zero_value)};
    const Id both{OpLogicalAnd(U32[1], is_last_subgroup, is_first_lane)};

    OpSelectionMerge(end_merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(both, last_subgroup_label, end_merge_label);

    OpLoopMerge(poll_success_label, loop_continue_label, spv::LoopControlMask::MaskNone);
    OpBranch(loop_header_label);

    const Id counter_ptr{OpAccessChain(pointer_type, scratch_buffer_id,
                                       ConstU32(ScratchBufferOffsets::LastCountedWorkgroup))};

    const auto val{OpAtomicLoad(U32[1], counter_ptr, device_scope, u32_zero_value)};

    const Id equals_target{OpIEqual(U1[1], val, block_index)};

    OpSelectionMerge(poll_failed_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(equals_target, poll_success_label, poll_failed_label);

    AddLabel(poll_failed_label);
    OpBranch(loop_continue_label);

    // TODO check if necessary or can make poll_failed_label the continue target
    AddLabel(loop_continue_label);
    OpBranch(loop_header_label);

    AddLabel(poll_success_label);
    const Id last_exclusive_sum{OpLoad(U32[1], subgroup_counts_at_subgroup_id)};
    const Id block_count{OpIAdd(U32[1], last_exclusive_sum, subgroup_count)};
    const Id global_count_ptr{OpAccessChain(pointer_type, scratch_buffer_id,
                                            ConstU32(ScratchBufferOffsets::GlobalCount))};
    const Id prev_global_count{
        OpAtomicIAdd(U32[1], global_count_ptr, device_scope, u32_zero_value, block_count)};
    OpAtomicStore(counter_ptr, device_scope, release_semantics,
                  OpIAdd(U32[1], block_index, u32_one_value));
    OpStore(shared_scratch_ptr, prev_global_count);
    OpBranch(end_merge_label);

    AddLabel(end_merge_label);

    SyncWorkgroupBarrier(*this);

    const Id my_subgroup_scan{OpLoad(U32[1], subgroup_counts_at_subgroup_id)};
    const Id result{
        OpIAdd(U32[1], prev_global_count, OpIAdd(U32[1], my_subgroup_scan, my_count_offset))};
    OpReturnValue(result);
    OpFunctionEnd();

    return func;
}

Id EmitOrderedCount(EmitContext& ctx, u32 packer_id, Id is_active) {
    return ctx.OpFunctionCall(ctx.U32[1], ctx.ordered_count_function, ctx.workgroup_index_id,
                              ctx.ConstU32(packer_id), is_active);
}

} // namespace Shader::Backend::SPIRV

// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
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

void EmitContext::InitEmulatedWorkgroupIndex() {
    workgroup_index_id = OpFunctionCall(U32[1], init_emulated_workgroup_index_function);
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

void EmitContext::DefineOrderedCountFunctions() {
    // uint orderedCount(uint block_index, uint packer_id, bool is_active)
    const Id u32_fn_type{TypePointer(spv::StorageClass::Function, U32[1])};
    const Id bool_fn_type{TypePointer(spv::StorageClass::Function, U1[1])};

    {
        const Id ordered_count_func_type{
            TypeFunction(U32[1], u32_fn_type, u32_fn_type, u32_fn_type, bool_fn_type)};
        ordered_count_add_function =
            OpFunction(U32[1], spv::FunctionControlMask::MaskNone, ordered_count_func_type);
        const auto block_index{OpFunctionParameter(u32_fn_type)};
        Name(block_index, "block_index");
        const auto packer_id{OpFunctionParameter(u32_fn_type)};
        Name(packer_id, "packer_id");
        const auto value{OpFunctionParameter(u32_fn_type)};
        Name(value, "value");
        const auto is_active{OpFunctionParameter(bool_fn_type)};
        Name(is_active, "is_active");

        Name(ordered_count_add_function, "ordered_count_one_thread");

        // DecorateLinkage(ordered_count_function, spv::LinkageType::Import, "ordered_count");
        DecorateLinkage(ordered_count_add_function, spv::LinkageType::Import,
                        "ordered_count_one_thread");
        OpFunctionEnd();
    }

    {
        const Id ordered_count_func_type{
            TypeFunction(U32[1], u32_fn_type, u32_fn_type, u32_fn_type, bool_fn_type)};
        ordered_count_swap_function =
            OpFunction(U32[1], spv::FunctionControlMask::MaskNone, ordered_count_func_type);
        const auto block_index{OpFunctionParameter(u32_fn_type)};
        Name(block_index, "block_index");
        const auto packer_id{OpFunctionParameter(u32_fn_type)};
        Name(packer_id, "packer_id");
        const auto value{OpFunctionParameter(u32_fn_type)};
        Name(value, "value");
        const auto is_active{OpFunctionParameter(bool_fn_type)};
        Name(is_active, "is_active");

        Name(ordered_count_swap_function, "ordered_count_swap_one_thread");

        DecorateLinkage(ordered_count_swap_function, spv::LinkageType::Import,
                        "ordered_count_swap_one_thread");
        OpFunctionEnd();
    }

    {
        const Id init_workgroup_id_func_type = TypeFunction(U32[1]);
        init_emulated_workgroup_index_function =
            OpFunction(U32[1], spv::FunctionControlMask::MaskNone, init_workgroup_id_func_type);
        Name(init_emulated_workgroup_index_function, "init_emulated_workgroup_index");
        OpFunctionEnd();

        DecorateLinkage(init_emulated_workgroup_index_function, spv::LinkageType::Import,
                        "init_emulated_workgroup_index");
    }
}

Id EmitOrderedCount(EmitContext& ctx, IR::Inst* inst, u32 packer_id, Id value, Id is_active) {
    auto flags = inst->Flags<OrderedCount::Flags>();

    ctx.OpStore(ctx.ordered_count_function_arg_vars[0], ctx.workgroup_index_id);
    ctx.OpStore(ctx.ordered_count_function_arg_vars[1], ctx.ConstU32(packer_id));
    ctx.OpStore(ctx.ordered_count_function_arg_vars[2], value);
    ctx.OpStore(ctx.ordered_count_function_arg_vars[3], is_active);

    Id function;
    switch (flags.instruction_type.Value()) {
    case OrderedCount::Op::Add:
        function = ctx.ordered_count_add_function;
        break;
    case OrderedCount::Op::Swap:
        function = ctx.ordered_count_swap_function;
        break;
    default:
        UNREACHABLE();
    }
    return ctx.OpFunctionCall(ctx.U32[1], function, ctx.ordered_count_function_arg_vars);
}

} // namespace Shader::Backend::SPIRV

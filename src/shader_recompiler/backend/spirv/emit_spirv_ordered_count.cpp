// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/microinstruction.h"
#include "shader_recompiler/ordered_count.h"

namespace Shader::Backend::SPIRV {

using namespace OrderedCount;

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
    const Id init_emulated_workgroup_index_func_type = TypeFunction(U32[1]);
    init_emulated_workgroup_index_function = OpFunction(U32[1], spv::FunctionControlMask::MaskNone,
                                                        init_emulated_workgroup_index_func_type);
    OpFunctionEnd();
    DecorateLinkage(init_emulated_workgroup_index_function, spv::LinkageType::Import,
                    "init_emulated_workgroup_index");

    const Id ordered_count_func_type{TypeFunction(U32[1], U32[1], U32[1], U32[1], U1[1])};

    ordered_count_add_per_wave_function =
        OpFunction(U32[1], spv::FunctionControlMask::MaskNone, ordered_count_func_type);
    OpFunctionEnd();
    DecorateLinkage(ordered_count_add_per_wave_function, spv::LinkageType::Import,
                    "ordered_count_add_per_wave");

    ordered_count_add_per_workgroup_function =
        OpFunction(U32[1], spv::FunctionControlMask::MaskNone, ordered_count_func_type);
    OpFunctionEnd();
    DecorateLinkage(ordered_count_add_per_workgroup_function, spv::LinkageType::Import,
                    "ordered_count_add_workgroupsync");

    ordered_count_swap_function =
        OpFunction(U32[1], spv::FunctionControlMask::MaskNone, ordered_count_func_type);
    OpFunctionEnd();
    DecorateLinkage(ordered_count_swap_function, spv::LinkageType::Import,
                    "ordered_count_swap_per_wave");
}

Id EmitOrderedCount(EmitContext& ctx, IR::Inst* inst, u32 packer_id, Id value, Id is_active) {
    const auto flags = inst->Flags<OrderedCount::Flags>();
    const bool can_reconverge_workgroup = flags.can_reconverge_workgroup.Value();

    ASSERT(flags.wave_release.Value());

    Id function;
    switch (flags.instruction_type.Value()) {
    case OrderedCount::Op::Add:
        // function = can_reconverge_workgroup ? ctx.ordered_count_add_per_workgroup_function :
        // ctx.ordered_count_add_per_wave_function;
        function = ctx.ordered_count_add_per_workgroup_function;
        // function = ctx.ordered_count_add_per_wave_function; // GOW needs atm
        break;
    case OrderedCount::Op::Swap:
        function = ctx.ordered_count_swap_function;
        break;
    default:
        UNREACHABLE();
    }

    return ctx.OpFunctionCall(ctx.U32[1], function, ctx.workgroup_index_id, ctx.ConstU32(packer_id),
                              value, is_active);
}

} // namespace Shader::Backend::SPIRV

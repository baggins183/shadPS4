// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {
using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

using AtomicFunc = Id (Sirit::Module::*)(Id, Id, Id, Id, Id);
using AtomicIncDecFunc = Id (Sirit::Module::*)(Id, Id, Id, Id);
using AtomicCmpSwpFunc = Id (Sirit::Module::*)(Id, Id, Id, Id, Id, Id, Id);

static std::pair<Id, Id> SharedAtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Workgroup))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

static std::pair<Id, Id> BufferAtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Device))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

static Id GetBufferAtomicPointer(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                                 PointerType pointer_type) {
    const auto& buffer = ctx.buffers[handle];
    const PointerSize pointer_size = EmitContext::GetPointerSize(pointer_type);
    if (const Id offset = buffer.Offset(pointer_size); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, spv_pointer_type] = buffer.Alias(pointer_type);
    return ctx.OpAccessChain(spv_pointer_type, id, ctx.u32_zero_value, address);
}

static std::pair<Id, Id> GetSharedAlias(EmitContext& ctx, PointerType pointer_type) {
    switch (pointer_type) {
    case PointerType::U16:
        return {ctx.shared_u16, ctx.shared_memory_u16};
    case PointerType::U32:
        return {ctx.shared_u32, ctx.shared_memory_u32};
    case PointerType::F32:
        return {ctx.shared_f32, ctx.shared_memory_f32};
    case PointerType::U64:
        return {ctx.shared_u64, ctx.shared_memory_u64};
    default:
        UNREACHABLE();
    }
}

static Id GetSharedAtomicPointer(EmitContext& ctx, Id offset, PointerType pointer_type) {
    const PointerSize pointer_size = EmitContext::GetPointerSize(pointer_type);
    const Id shift_id{ctx.ConstU32(static_cast<u32>(pointer_size))};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    auto [base, result_type] = GetSharedAlias(ctx, pointer_type);
    return ctx.EmitSharedMemoryAccess(result_type, base, index);
}

static Id GetImageAtomicPointer(EmitContext& ctx, u32 handle, Id coords) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    return ctx.OpImageTexelPointer(ctx.image_u32, texture.id, coords, ctx.ConstU32(0U));
}

template <bool is_float = false>
static Id SharedAtomicB32(EmitContext& ctx, Id offset, Id value, AtomicFunc atomic_func) {
    const Id type = is_float ? ctx.F32[1] : ctx.U32[1];
    PointerType pointer_type = is_float ? PointerType::F32 : PointerType::U32;
    const Id ptr = GetSharedAtomicPointer(ctx, offset, pointer_type);
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics, value);
}

Id SharedAtomicU32IncDec(EmitContext& ctx, Id offset,
                         Id (Sirit::Module::*atomic_func)(Id, Id, Id, Id)) {
    const Id pointer{GetSharedAtomicPointer(ctx, offset, PointerType::U32)};
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics);
}

Id SharedAtomicU64(EmitContext& ctx, Id offset, Id value, AtomicFunc atomic_func) {
    const Id pointer{GetSharedAtomicPointer(ctx, offset, PointerType::U64)};
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics, value);
}

Id SharedAtomicU64IncDec(EmitContext& ctx, Id offset, AtomicIncDecFunc atomic_func) {
    const Id pointer{GetSharedAtomicPointer(ctx, offset, PointerType::U64)};
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, pointer, scope, semantics);
}

Id SharedAtomicU32CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id pointer{GetSharedAtomicPointer(ctx, offset, PointerType::U32)};
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

Id SharedAtomicU64CmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    const Id pointer{GetSharedAtomicPointer(ctx, offset, PointerType::U64)};
    const auto [scope, semantics]{SharedAtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U64, pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

template <bool is_float = false>
Id BufferAtomicB32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   AtomicFunc atomic_func) {
    const Id type = is_float ? ctx.F32[1] : ctx.U32[1];
    PointerType pointer_type = is_float ? PointerType::F32 : PointerType::U32;
    const Id ptr = GetBufferAtomicPointer(ctx, inst, handle, address, pointer_type);
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(type, ptr, scope, semantics, value);
}

Id BufferAtomicU32IncDec(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                         AtomicIncDecFunc atomic_func) {
    const Id ptr = GetBufferAtomicPointer(ctx, inst, handle, address, PointerType::U32);
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics);
}

Id BufferAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                          Id cmp_value, AtomicCmpSwpFunc atomic_func) {
    const Id ptr = GetBufferAtomicPointer(ctx, inst, handle, address, PointerType::U32);
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], ptr, scope, semantics, semantics, value, cmp_value);
}

Id BufferAtomicU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                   AtomicFunc atomic_func) {
    const Id ptr = GetBufferAtomicPointer(ctx, inst, handle, address, PointerType::U64);
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U64, ptr, scope, semantics, value);
}

Id ImageAtomicU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  AtomicFunc atomic_func) {
    const Id pointer{GetImageAtomicPointer(ctx, handle, coords)};
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, value);
}

Id ImageAtomicF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                  AtomicFunc atomic_func) {
    const Id pointer{GetImageAtomicPointer(ctx, handle, coords)};
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.F32[1], pointer, scope, semantics, value);
}

Id ImageAtomicU32CmpSwap(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                         Id cmp_value, AtomicCmpSwpFunc atomic_func) {
    const Id pointer{GetImageAtomicPointer(ctx, handle, coords)};
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    return (ctx.*atomic_func)(ctx.U32[1], pointer, scope, semantics, semantics, value, cmp_value);
}
} // Anonymous namespace

Id EmitSharedAtomicIAdd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicIAdd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitSharedAtomicUMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicUMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitSharedAtomicSMax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicSMax64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitSharedAtomicUMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicUMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitSharedAtomicSMin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicSMin64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitSharedAtomicAnd32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicAnd64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitSharedAtomicOr32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicOr64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicOr);
}

Id EmitSharedAtomicXor32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicXor64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicXor);
}

Id EmitSharedAtomicISub32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicB32(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicISub64(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicU64(ctx, offset, value, &Sirit::Module::OpAtomicISub);
}

Id EmitSharedAtomicCmpSwap32(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU32CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicCmpSwap64(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicU64CmpSwap(ctx, offset, value, cmp_value);
}

Id EmitSharedAtomicInc32(EmitContext& ctx, Id offset) {
    return SharedAtomicU32IncDec(ctx, offset, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicInc64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicDec32(EmitContext& ctx, Id offset) {
    return SharedAtomicU32IncDec(ctx, offset, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitSharedAtomicDec64(EmitContext& ctx, Id offset) {
    return SharedAtomicU64IncDec(ctx, offset, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicIAdd64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitBufferAtomicISub32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicISub);
}

Id EmitBufferAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicSMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitBufferAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicUMin64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitBufferAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicSMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitBufferAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicUMax64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicU64(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicUMax);
}

Id EmitBufferAtomicInc32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, inst, handle, address, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitBufferAtomicDec32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, inst, handle, address, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitBufferAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicOr);
}

Id EmitBufferAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicXor);
}

Id EmitBufferAtomicSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicB32(ctx, inst, handle, address, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitBufferAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                             Id cmp_value) {
    return BufferAtomicU32CmpSwap(ctx, inst, handle, address, value, cmp_value,
                                  &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitBufferAtomicFCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value,
                              Id cmp_value) {
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto u32_cmp = ctx.OpBitcast(ctx.U32[1], cmp_value);
    const auto result = BufferAtomicU32CmpSwap(ctx, inst, handle, address, u32_value, u32_cmp,
                                               &Sirit::Module::OpAtomicCompareExchange);
    return ctx.OpBitcast(ctx.F32[1], result);
}

Id EmitImageAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicIAdd);
}

Id EmitImageAtomicSMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMin);
}

Id EmitImageAtomicUMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMin);
}

Id EmitImageAtomicSMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicSMax);
}

Id EmitImageAtomicUMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicUMax);
}

template <bool is_min>
static Id FMinMaxFallback(EmitContext& ctx, Id ptr, Id value) {
    const auto [scope, semantics]{BufferAtomicArgs(ctx)};
    const Id true_label{ctx.OpLabel()};
    const Id false_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};

    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto sign_bit_set = ctx.OpINotEqual(
        ctx.U1[1],
        ctx.OpBitFieldUExtract(ctx.U32[1], u32_value, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);

    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(sign_bit_set, true_label, false_label);

    ctx.AddLabel(true_label);
    const Id sign_bit_set_atomic =
        is_min ? ctx.OpAtomicUMax(ctx.U32[1], ptr, scope, semantics, u32_value)
               : ctx.OpAtomicUMin(ctx.U32[1], ptr, scope, semantics, u32_value);
    ctx.OpBranch(merge_label);

    ctx.AddLabel(false_label);
    const Id sign_bit_unset_atomic =
        is_min ? ctx.OpAtomicSMin(ctx.U32[1], ptr, scope, semantics, u32_value)
               : ctx.OpAtomicSMax(ctx.U32[1], ptr, scope, semantics, u32_value);
    ctx.OpBranch(merge_label);

    ctx.AddLabel(merge_label);
    Id result =
        ctx.OpPhi(ctx.U32[1], sign_bit_set_atomic, true_label, sign_bit_unset_atomic, false_label);
    return ctx.OpBitcast(ctx.F32[1], result);
}

template <bool is_min>
static Id ImageAtomicFMinMax(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        auto atomic_fn = is_min ? &Sirit::Module::OpAtomicFMin : &Sirit::Module::OpAtomicFMax;
        return ImageAtomicF32(ctx, inst, handle, coords, value, atomic_fn);
    } else {
        const Id ptr{GetImageAtomicPointer(ctx, handle, coords)};
        return FMinMaxFallback<is_min>(ctx, ptr, value);
    }
}

Id EmitImageAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicFMinMax<false>(ctx, inst, handle, coords, value);
}

Id EmitImageAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicFMinMax<true>(ctx, inst, handle, coords, value);
}

Id EmitImageAtomicInc32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitImageAtomicDec32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitImageAtomicAnd32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicAnd);
}

Id EmitImageAtomicOr32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicOr);
}

Id EmitImageAtomicXor32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicXor);
}

Id EmitImageAtomicExchange32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    return ImageAtomicU32(ctx, inst, handle, coords, value, &Sirit::Module::OpAtomicExchange);
}

Id EmitImageAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value,
                            Id cmp_value) {
    return ImageAtomicU32CmpSwap(ctx, inst, handle, coords, value, cmp_value,
                                 &Sirit::Module::OpAtomicCompareExchange);
}

Id EmitDataAppend(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitDataConsume(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

template <bool is_min>
static Id BufferAtomicFMinMax(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicB32<true>(ctx, inst, handle, address, value,
                                     is_min ? &Sirit::Module::OpAtomicFMin
                                            : &Sirit::Module::OpAtomicFMax);
    } else {
        const Id ptr{GetBufferAtomicPointer(ctx, inst, handle, address, PointerType::U32)};
        return FMinMaxFallback<is_min>(ctx, ptr, value);
    }
}

Id EmitBufferAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicFMinMax<true>(ctx, inst, handle, address, value);
}

Id EmitBufferAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    return BufferAtomicFMinMax<false>(ctx, inst, handle, address, value);
}

template <bool is_min>
static Id SharedAtomicFMinMax(EmitContext& ctx, Id offset, Id value) {
    if (ctx.profile.supports_shared_f32_atomic_minmax) {
        auto atomic_fn = is_min ? &Sirit::Module::OpAtomicFMin : &Sirit::Module::OpAtomicFMax;
        return SharedAtomicB32<true>(ctx, offset, value, atomic_fn);
    } else {
        const Id ptr{GetSharedAtomicPointer(ctx, offset, PointerType::U32)};
        return FMinMaxFallback<is_min>(ctx, ptr, value);
    }
}

Id EmitSharedAtomicFmin32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicFMinMax<true>(ctx, offset, value);
}

Id EmitSharedAtomicFmax32(EmitContext& ctx, Id offset, Id value) {
    return SharedAtomicFMinMax<false>(ctx, offset, value);
}

} // namespace Shader::Backend::SPIRV

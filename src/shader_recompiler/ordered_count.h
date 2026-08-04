// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/bit_field.h"
#include "common/types.h"
namespace OrderedCount {

enum class ShaderType : u32 {
    Cs = 0,
    Ps = 1,
    Vs = 2,
    Gs = 3,
};

enum class Op : u32 {
    Add = 0,
    Swap = 1,
    Wrap = 3,
};

union Flags {
    BitField<2, 6, u32> packer_id;
    BitField<8, 1, u32> wave_release;
    BitField<9, 1, u32> wave_done;
    BitField<10, 2, ShaderType> shader_type;
    BitField<12, 2, Op> instruction_type;
    u32 raw;
};

// Assume only one counter for now
// in bytes
static constexpr u32 UtilityBufferSize = 12;

// in dwords
namespace UtilityBufferOffsets {
static const u32 NextEmulatedWorkgroupIndex = 0;
static const u32 LastCountedWorkgroup = 1;
static const u32 GlobalCount = 2;
} // namespace UtilityBufferOffsets

namespace SharedMemStructIndices {
static const u32 SubgroupCounts = 0;
static const u32 ScratchVal = 1;
} // namespace SharedMemStructIndices

} // namespace OrderedCount
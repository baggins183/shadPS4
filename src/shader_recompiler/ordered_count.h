// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
namespace OrderedCount {

// Assume only one counter for now
// in bytes
static constexpr u32 ScratchBufferSize = 12;

// in dwords
namespace ScratchBufferOffsets {
static const u32 NextEmulatedWorkgroupIndex = 0;
static const u32 LastCountedWorkgroup = 1;
static const u32 GlobalCount = 2;
} // namespace ScratchBufferOffsets

} // namespace OrderedCount
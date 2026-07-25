// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/types.h"

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst);
std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts);
std::vector<u32> TranslateToSpirvForOrderedCount(u32 workgroup_size_x, u32 num_workgroups_x,
                                                 u32& utility_buffer_size);

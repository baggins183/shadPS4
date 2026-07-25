// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "translator.hpp"

#include <iostream>

#include "common/io_file.h"
#include "instructions.hpp"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

using namespace Shader;

namespace Shader::Optimization {
void ResourceTrackingPassStub(IR::Program& program, const Profile& profile);
}

std::vector<u32> TranslateToSpirv(u64 raw_gcn_inst) {
    return TranslateToSpirv(std::span<const u64>{&raw_gcn_inst, 1});
}

std::vector<u32> TranslateToSpirv(std::span<const u64> raw_gcn_insts) {
    std::array<u32, 2> store{
        0xe0700000,
        0x80000000 // buffer_store_dword v0, v0, s[0:3], 0
    };
    Gcn::GcnCodeSlice second(store.data(), store.data() + store.size());

    Gcn::GcnDecodeContext decoder;
    std::vector<Gcn::GcnInst> instructions;
    instructions.reserve(raw_gcn_insts.size());
    for (const u64 raw_gcn_inst : raw_gcn_insts) {
        std::array<u32, 2> provided_inst{static_cast<u32>(raw_gcn_inst & 0xFFFFFFFFU),
                                         static_cast<u32>(raw_gcn_inst >> 32)};
        Gcn::GcnCodeSlice slice(provided_inst.data(), provided_inst.data() + provided_inst.size());
        instructions.push_back(decoder.decodeInstruction(slice));
    }
    Gcn::GcnInst store_inst = decoder.decodeInstruction(second);

    Shader::Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    AmdGpu::Buffer buf = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &buf, sizeof(buf));

    IR::Program program{info};
    Pools pools{};

    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32; // TODO
    profile.max_shared_memory_size = 128;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.num_user_data = 4;
    runtime_info.cs_info.workgroup_size = {1, 1, 1};

    Gcn::Translator translator(program.info, runtime_info, profile);
    translator.EmitPrologue(block);

    for (int i = 0; i < 4; ++i) {
        // copy user data from SGPR to VGPR as (most?) instructions cannot access
        // two SGPRs
        Shader::Gcn::GcnInst mov{};
        mov.src[0].field = Shader::Gcn::OperandField::ScalarGPR;
        mov.src[0].code = i;
        mov.dst[0].field = Shader::Gcn::OperandField::VectorGPR;
        mov.dst[0].code = i;
        translator.S_MOV(mov);
    }
    for (const Gcn::GcnInst& inst : instructions) {
        translator.TranslateInstruction(inst);
    }
    translator.TranslateInstruction(store_inst);

    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::Optimization::ResourceTrackingPassStub(program, profile);
    Shader::Optimization::ConstantPropagationPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};

    const auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

    return spirv;
}

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

// TODO refactor all this
std::vector<u32> TranslateToSpirvForOrderedCount(u32 workgroup_size_x, u32 num_workgroups_x,
                                                 u32& utility_buffer_size) {
    const u32 packer_id = 0;
    u32 num_threads = workgroup_size_x * num_workgroups_x;

    // Initial Sgprs:
    // S[0:3] : buffer
    // S4     : WorkgroupId.x

    // Initial Vgprs:
    // V0     : LocalInvocationId.x

    // buffer contents (dword indices):
    // [0:num_threads)               : 0/1 if thread[index] is active in ordered count
    // [num_threads:2 * num_threads) : ordered count result

    struct Code {
        std::vector<u32> raw_insts;

        void AppendInst(u32 raw) {
            raw_insts.push_back(raw);
        }
        void AppendInst(u64 raw) {
            raw_insts.push_back(static_cast<u32>(raw & 0xFFFFFFFFU));
            raw_insts.push_back(static_cast<u32>(raw >> 32));
        }
    };

    Code bytecode;

    bytecode.AppendInst(
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S5, SOperand8::LiteralConstant).Get());
    bytecode.AppendInst(workgroup_size_x);
    bytecode.AppendInst(
        SOP1(OpcodeSOP1::S_MOV_B32, SOperand7::S6, SOperand8::LiteralConstant).Get());
    bytecode.AppendInst(num_threads << 2);
    bytecode.AppendInst(
        SOP2(OpcodeSOP2::S_MUL_I32, SOperand7::S4, SOperand8::S4, SOperand8::S5).Get());
    // V0 <- WorkgroupId.x * workgroup_size_x + LocalInvocationId.x
    bytecode.AppendInst(
        VOP2(OpcodeVOP2::V_ADD_I32, VOperand8::V0, SOperand9::S4, VOperand8::V0).Get());

    bytecode.AppendInst(
        VOP2(OpcodeVOP2::V_LSHLREV_B32, VOperand8::V0, SOperand9::Const2, VOperand8::V0).Get());

    // load is_active bit to V1
    bytecode.AppendInst(MUBUF(OpcodeMUBUF::BUFFER_LOAD_DWORD, VOperand8::V0, VOperand8::V1,
                              SOperand5::S0_S1_S2_S3, SOperand8::Const0, 0)
                            .SetOffEn()
                            .Get());

    bytecode.AppendInst(VOPC(OpcodeVOPC::V_CMP_EQ_U32, SOperand9::Const1, VOperand8::V1).Get());

    // Save exec to S7
    bytecode.AppendInst(
        SOP1(OpcodeSOP1::S_AND_SAVEEXEC_B64, SOperand7::S7, SOperand8::VccLo).Get());

    // most operands ignored?
    bytecode.AppendInst(DS(OpcodeDS::DS_ORDERED_COUNT, VOperand8::V1, VOperand8::V21,
                           VOperand8::V21, VOperand8::V21, packer_id << 2, 0, true)
                            .Get());

    bytecode.AppendInst(SOP1(OpcodeSOP1::S_MOV_B64, SOperand7::ExecLo, SOperand8::S7).Get());

    // store ordered count result to buffer
    bytecode.AppendInst(MUBUF(OpcodeMUBUF::BUFFER_STORE_DWORD, VOperand8::V0, VOperand8::V1,
                              SOperand5::S0_S1_S2_S3, SOperand8::S6, 0)
                            .SetOffEn()
                            .Get());

    bytecode.AppendInst(SOPP(OpcodeSOPP::S_ENDPGM, 0).Get());

    Gcn::GcnCodeSlice slice(bytecode.raw_insts.data(),
                            bytecode.raw_insts.data() + bytecode.raw_insts.size());
    Gcn::GcnDecodeContext decoder;

    Shader::Pools pools;

    Shader::Info info{};
    info.stage = Stage::Compute;
    info.l_stage = LogicalStage::Compute;
    info.flattened_ud_buf.resize(4);
    AmdGpu::Buffer buf = AmdGpu::Buffer::Null();
    std::memcpy(info.flattened_ud_buf.data(), &buf, sizeof(buf));

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(bytecode.raw_insts.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = 32; // TODO
    profile.max_shared_memory_size = 128;

    RuntimeInfo runtime_info{};
    runtime_info.Initialize(Stage::Compute);
    runtime_info.num_user_data = 4;
    runtime_info.cs_info.workgroup_size = {workgroup_size_x, 1, 1};
    runtime_info.cs_info.tgid_enable = {true, false, false};

    // Structurize control flow graph and create program.
    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    std::vector<Gcn::GcnInst> insts;

    IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);

    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Gcn::Translator translator(program.info, runtime_info, profile);

    for (auto inst : insts) {
        translator.TranslateInstruction(inst);
    }

    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::IR::DumpIrProgram(program, program.info, "test.post_ssa.irprogram");
    Shader::Optimization::ResourceTrackingPassStub(program, profile);
    Shader::IR::DumpIrProgram(program, program.info, "test.post_resource_tracking_stub.irprogram");
    Shader::Optimization::ConstantPropagationPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::IR::DumpIrProgram(program, program.info, "test.post_dce.irprogram");
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Backend::Bindings bindings{};

    const auto spirv = Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);

    utility_buffer_size = info.OrderedCountScratchBufferSize();

    return spirv;
}
/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * 实现 RV32I + RV32F 指令解释器，按 SIMT 模型组织 warp 执行。
 * warp 内 32 个 lane 锁步执行同一条指令，通过 active_mask 控制活跃性。
 *
 * VRAM 访问通过 s->vram_ptr 直接指针操作（内部路径），
 * 与 CPU 侧的 BAR2 MMIO 路径（gpgpu_vram_read/write）访问同一块内存。
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/*
 * ============================================================================
 * RV32I / RV32F 指令解码辅助宏
 * ============================================================================
 * RISC-V 指令编码格式：
 *   [31:25] funct7    [24:20] rs2    [19:15] rs1    [14:12] funct3
 *   [11:7]  rd        [6:0]   opcode
 */
#define RV_OPCODE(inst)   ((inst) & 0x7F)
#define RV_RD(inst)       (((inst) >> 7) & 0x1F)
#define RV_FUNCT3(inst)   (((inst) >> 12) & 0x7)
#define RV_RS1(inst)      (((inst) >> 15) & 0x1F)
#define RV_RS2(inst)      (((inst) >> 20) & 0x1F)
#define RV_FUNCT7(inst)   (((inst) >> 25) & 0x7F)

/* 符号扩展的 12 位立即数（I-type），利用算术右移 */
#define RV_IMM_I(inst)    ((int32_t)(inst) >> 20)
/* S-type 立即数：[31:25] + [11:7] */
#define RV_IMM_S(inst)    ((((int32_t)(inst) >> 25) << 5) | (((inst) >> 7) & 0x1F))
/* LUI 的 20 位立即数 */
#define RV_IMM_U(inst)    ((inst) & 0xFFFFF000)
/* 移位量（I-type 的 [24:20]） */
#define RV_SHAMT(inst)    (((inst) >> 20) & 0x1F)

/* RISC-V opcode 常量 */
#define OPCODE_LUI    0x37
#define OPCODE_AUIPC  0x17
#define OPCODE_OP_IMM 0x13
#define OPCODE_OP     0x33
#define OPCODE_STORE  0x23
#define OPCODE_LOAD   0x03
#define OPCODE_BRANCH 0x63
#define OPCODE_SYSTEM 0x73
#define OPCODE_OP_FP  0x53

/* 写通用寄存器，x0 始终为 0 */
static inline void gpr_write(GPGPULane *lane, uint32_t rd, uint32_t val)
{
    if (rd != 0) {
        lane->gpr[rd] = val;
    }
}

/* 写浮点寄存器 */
static inline void fpr_write(GPGPULane *lane, uint32_t rd, uint32_t val)
{
    lane->fpr[rd] = val;
}

static float32 minifloat_subnormal_to_float32(uint32_t sign, uint32_t mant,
                                              uint32_t mant_bits,
                                              int32_t bias)
{
    uint32_t hidden = 1u << mant_bits;
    int32_t exp = 1 - bias;

    while ((mant & hidden) == 0) {
        mant <<= 1;
        exp--;
    }

    return (sign << 31) | ((uint32_t)(exp + 127) << 23)
           | ((mant & (hidden - 1)) << (23 - mant_bits));
}

static uint32_t fp32_to_minifloat_subnormal(uint32_t mant, int32_t unbiased,
                                            uint32_t mant_bits, int32_t bias,
                                            uint32_t max_mant)
{
    uint32_t sig = (1u << 23) | mant;
    int32_t shift = unbiased + bias + (int32_t)mant_bits - 24;
    uint32_t result;

    if (shift >= 0) {
        uint64_t shifted = (uint64_t)sig << shift;
        result = shifted > max_mant ? max_mant : (uint32_t)shifted;
    } else {
        int32_t rshift = -shift;
        result = rshift >= 32 ? 0 : sig >> rshift;
    }

    return result > max_mant ? max_mant : result;
}

static float32 e4m3_to_float32(uint32_t e4m3)
{
    uint32_t sign = (e4m3 >> 7) & 1;
    uint32_t exp = (e4m3 >> 3) & 0xF;
    uint32_t mant = e4m3 & 0x7;

    if (e4m3 == 0x7F || e4m3 == 0xFF) {
        return 0x7FC00000;
    }
    if (exp == 0) {
        if (mant == 0) {
            return sign << 31;
        }
        return minifloat_subnormal_to_float32(sign, mant, 3, 7);
    }

    return (sign << 31) | ((exp + 120) << 23) | (mant << 20);
}

static uint32_t float32_to_e4m3(float32 fp32)
{
    uint32_t sign = (fp32 >> 31) & 1;
    uint32_t exp = (fp32 >> 23) & 0xFF;
    uint32_t mant = fp32 & 0x7FFFFF;
    uint32_t e4_exp;
    uint32_t e4_mant;

    if (exp == 0xFF) {
        return mant != 0 ? 0x7F : ((sign << 7) | 0x7E);
    }
    if (exp == 0) {
        return sign << 7;
    }

    int32_t target_exp = (int32_t)exp - 127 + 7;
    if (target_exp <= 0) {
        e4_mant = fp32_to_minifloat_subnormal(mant, (int32_t)exp - 127,
                                              3, 7, 0x7);
        return e4_mant == 0 ? (sign << 7) : ((sign << 7) | e4_mant);
    }
    if (target_exp > 15) {
        return (sign << 7) | 0x7E;
    }

    e4_exp = (uint32_t)target_exp;
    e4_mant = mant >> 20;
    if (e4_exp == 15 && e4_mant == 7) {
        return (sign << 7) | 0x7E;
    }

    return (sign << 7) | (e4_exp << 3) | e4_mant;
}

static float32 e5m2_to_float32(uint32_t e5m2)
{
    uint32_t sign = (e5m2 >> 7) & 1;
    uint32_t exp = (e5m2 >> 2) & 0x1F;
    uint32_t mant = e5m2 & 0x3;

    if (exp == 0x1F) {
        return (sign << 31) | (mant == 0 ? 0x7F800000 : 0x7FC00000);
    }
    if (exp == 0) {
        if (mant == 0) {
            return sign << 31;
        }
        return minifloat_subnormal_to_float32(sign, mant, 2, 15);
    }

    return (sign << 31) | ((exp + 112) << 23) | (mant << 21);
}

static uint32_t float32_to_e5m2(float32 fp32)
{
    uint32_t sign = (fp32 >> 31) & 1;
    uint32_t exp = (fp32 >> 23) & 0xFF;
    uint32_t mant = fp32 & 0x7FFFFF;
    uint32_t e5_mant;

    if (exp == 0xFF) {
        return (sign << 7) | (mant == 0 ? 0x7C : 0x7D);
    }
    if (exp == 0) {
        return sign << 7;
    }

    int32_t target_exp = (int32_t)exp - 127 + 15;
    if (target_exp <= 0) {
        e5_mant = fp32_to_minifloat_subnormal(mant, (int32_t)exp - 127,
                                              2, 15, 0x3);
        return e5_mant == 0 ? (sign << 7) : ((sign << 7) | e5_mant);
    }
    if (target_exp >= 31) {
        return (sign << 7) | 0x7C;
    }

    return (sign << 7) | ((uint32_t)target_exp << 2) | (mant >> 21);
}

static float32 e2m1_to_float32(uint32_t e2m1)
{
    static const float32 table[16] = {
        0x00000000, 0x3F000000, 0x3F800000, 0x3FC00000,
        0x40000000, 0x40400000, 0x40800000, 0x40C00000,
        0x80000000, 0xBF000000, 0xBF800000, 0xBFC00000,
        0xC0000000, 0xC0400000, 0xC0800000, 0xC0C00000,
    };

    return table[e2m1 & 0xF];
}

static uint32_t float32_to_e2m1(float32 fp32)
{
    uint32_t sign = (fp32 >> 31) & 1;
    uint32_t exp = (fp32 >> 23) & 0xFF;
    uint32_t mant = fp32 & 0x7FFFFF;
    uint32_t abs = fp32 & 0x7FFFFFFF;
    uint32_t code;

    if (exp == 0xFF && mant != 0) {
        return (sign << 3) | 0x7;
    }

    if (abs > 0x40C00000) {
        code = 0x7;
    } else if (abs > 0x40A00000) {
        code = 0x7;
    } else if (abs > 0x40600000) {
        code = 0x6;
    } else if (abs > 0x40200000) {
        code = 0x5;
    } else if (abs > 0x3FE00000) {
        code = 0x4;
    } else if (abs > 0x3FA00000) {
        code = 0x3;
    } else if (abs > 0x3F400000) {
        code = 0x2;
    } else if (abs > 0x3E800000) {
        code = 0x1;
    } else {
        code = 0x0;
    }

    return (sign << 3) | code;
}

/*
 * ============================================================================
 * exec_one_inst - 对一个 warp 的所有活跃 lane 执行一条指令
 * ============================================================================
 *
 * SIMT 锁步执行：所有活跃 lane 从同一 PC 取同一条指令，但各 lane
 * 操作自己的寄存器文件，产生不同结果。sw 指令各 lane 写不同 VRAM 地址。
 *
 * @s:     GPGPU 设备状态（用于访问 vram_ptr）
 * @warp:  当前 warp
 * @inst:  从 VRAM 取出的 32 位 RISC-V 指令
 */
static void exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{
    uint32_t opcode = RV_OPCODE(inst);
    uint32_t rd     = RV_RD(inst);
    uint32_t funct3 = RV_FUNCT3(inst);
    uint32_t rs1    = RV_RS1(inst);
    uint32_t rs2    = RV_RS2(inst);
    uint32_t funct7 = RV_FUNCT7(inst);

    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        if (!(warp->active_mask & (1u << i))) {
            continue;
        }

        GPGPULane *lane = &warp->lanes[i];

        switch (opcode) {
        case OPCODE_LUI:
            /* LUI: rd = imm20 << 12 */
            gpr_write(lane, rd, RV_IMM_U(inst));
            break;

        case OPCODE_OP_IMM:
            /* I-type 整数运算：ADDI, ANDI, SLLI 等 */
            switch (funct3) {
            case 0: /* ADDI: rd = rs1 + imm12 */
                gpr_write(lane, rd, lane->gpr[rs1] + (uint32_t)RV_IMM_I(inst));
                break;
            case 1: /* SLLI: rd = rs1 << shamt */
                gpr_write(lane, rd, lane->gpr[rs1] << RV_SHAMT(inst));
                break;
            case 7: /* ANDI: rd = rs1 & sign_ext(imm12) */
                gpr_write(lane, rd, lane->gpr[rs1] & (uint32_t)RV_IMM_I(inst));
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: unhandled OP_IMM funct3=%d\n", funct3);
                break;
            }
            break;

        case OPCODE_OP:
            /* R-type 整数运算 */
            switch (funct3) {
            case 0:
                if (funct7 == 0x00) {
                    /* ADD: rd = rs1 + rs2 */
                    gpr_write(lane, rd, lane->gpr[rs1] + lane->gpr[rs2]);
                }
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: unhandled OP funct3=%d\n", funct3);
                break;
            }
            break;

        case OPCODE_STORE:
            /* S-type 存储 */
            switch (funct3) {
            case 2: {
                /* SW: mem[rs1 + imm12] = rs2（写 VRAM） */
                uint32_t addr = lane->gpr[rs1] + (uint32_t)RV_IMM_S(inst);
                if (addr + 4 <= s->vram_size) {
                    stl_le_p(s->vram_ptr + addr, lane->gpr[rs2]);
                }
                break;
            }
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: unhandled STORE funct3=%d\n", funct3);
                break;
            }
            break;

        case OPCODE_SYSTEM:
            if (funct3 == 0 && inst == 0x00100073) {
                /* EBREAK: 该 lane 退出执行，清除 active 位 */
                lane->active = false;
                warp->active_mask &= ~(1u << i);
            } else if (funct3 == 2) {
                /* CSRRS: rd = CSR[csr]; CSR[csr] |= rs1 */
                uint32_t csr_addr = (inst >> 20) & 0xFFF;
                switch (csr_addr) {
                case CSR_MHARTID:
                    gpr_write(lane, rd, lane->mhartid);
                    break;
                case CSR_FCSR:
                    gpr_write(lane, rd, lane->fcsr);
                    break;
                default:
                    qemu_log_mask(LOG_GUEST_ERROR,
                                  "gpgpu: unhandled CSR 0x%x\n", csr_addr);
                    break;
                }
            }
            break;

        case OPCODE_OP_FP:
            /* RV32F 浮点运算 */
            switch (funct7) {
            case 0x00: {
                /* FADD.S: f[rd] = f[rs1] + f[rs2] */
                float32 result = float32_add(lane->fpr[rs1], lane->fpr[rs2],
                                               &lane->fp_status);
                fpr_write(lane, rd, result);
                break;
            }
            case 0x08: {
                /* FMUL.S: f[rd] = f[rs1] * f[rs2] */
                float32 result = float32_mul(lane->fpr[rs1], lane->fpr[rs2],
                                               &lane->fp_status);
                fpr_write(lane, rd, result);
                break;
            }
            case 0x68:
                if (rs2 == 0) {
                    /* FCVT.S.W: f[rd] = (float)x[rs1]（int32 → float32） */
                    float32 result = int32_to_float32((int32_t)lane->gpr[rs1],
                                                         &lane->fp_status);
                    fpr_write(lane, rd, result);
                }
                break;
            case 0x60:
                if (rs2 == 0) {
                    /* FCVT.W.S: x[rd] = (int)f[rs1]（float32 → int32）
                     * funct3[2:0] 是 rm（舍入模式），1=RTZ 时用 round_toward_zero */
                    uint32_t rm = funct3 & 0x7;
                    int old_rm = get_float_rounding_mode(&lane->fp_status);
                    if (rm == 1) {
                        /* RTZ: round toward zero */
                        set_float_rounding_mode(float_round_to_zero,
                                                &lane->fp_status);
                    }
                    int32_t result = float32_to_int32(lane->fpr[rs1],
                                                      &lane->fp_status);
                    gpr_write(lane, rd, (uint32_t)result);
                    /* 恢复舍入模式 */
                    set_float_rounding_mode(old_rm, &lane->fp_status);
                }
                break;
            case 0x22:
                if (rs2 == 0) {
                    fpr_write(lane, rd, (lane->fpr[rs1] & 0xFFFF) << 16);
                } else if (rs2 == 1) {
                    fpr_write(lane, rd, (lane->fpr[rs1] >> 16) & 0xFFFF);
                }
                break;
            case 0x24:
                if (rs2 == 0) {
                    fpr_write(lane, rd, e4m3_to_float32(lane->fpr[rs1] & 0xFF));
                } else if (rs2 == 1) {
                    fpr_write(lane, rd, float32_to_e4m3(lane->fpr[rs1]));
                } else if (rs2 == 2) {
                    fpr_write(lane, rd, e5m2_to_float32(lane->fpr[rs1] & 0xFF));
                } else if (rs2 == 3) {
                    fpr_write(lane, rd, float32_to_e5m2(lane->fpr[rs1]));
                }
                break;
            case 0x26:
                if (rs2 == 0) {
                    fpr_write(lane, rd, e2m1_to_float32(lane->fpr[rs1] & 0xF));
                } else if (rs2 == 1) {
                    fpr_write(lane, rd, float32_to_e2m1(lane->fpr[rs1]));
                }
                break;
            case 0x78:
                if (rs2 == 0) {
                    fpr_write(lane, rd, lane->gpr[rs1]);
                }
                break;
            default:
                qemu_log_mask(LOG_GUEST_ERROR,
                              "gpgpu: unhandled OP_FP funct7=0x%02x\n", funct7);
                break;
            }
            break;

        default:
            qemu_log_mask(LOG_GUEST_ERROR,
                          "gpgpu: unhandled opcode 0x%02x (inst=0x%08x)\n",
                          opcode, inst);
            break;
        }
    }
}

/*
 * ============================================================================
 * gpgpu_core_init_warp - 初始化一个 warp 的执行上下文
 * ============================================================================
 *
 * 为 warp 内每个 lane 设置 mhartid、PC、active 状态。
 * mhartid 编码为 block_id(19) | warp_id(8) | lane_id(5)。
 *
 * @warp:            warp 状态
 * @pc:              初始 PC（kernel 代码在 VRAM 中的偏移）
 * @thread_id_base:  该 warp 的起始线程 ID（warp_id * 32）
 * @block_id:        block 在 grid 中的三维索引 [x, y, z]
 * @num_threads:     该 warp 内活跃 lane 数量（最多 32）
 * @warp_id:         warp 在 block 内的编号
 * @block_id_linear: 线性化的 block ID
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                           uint32_t thread_id_base, const uint32_t block_id[3],
                           uint32_t num_threads,
                           uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    warp->thread_id_base = thread_id_base;

    /* 计算活跃 lane 数量：该 warp 负责的线程数，最多 32 */
    uint32_t active_lanes = (num_threads < GPGPU_WARP_SIZE)
                            ? num_threads : GPGPU_WARP_SIZE;

    /* 设置 active_mask：低 active_lanes 位置 1 */
    warp->active_mask = (active_lanes == GPGPU_WARP_SIZE)
                        ? 0xFFFFFFFFu
                        : (1u << active_lanes) - 1;

    /* 初始化每个活跃 lane */
    for (uint32_t i = 0; i < active_lanes; i++) {
        warp->lanes[i].active = true;
        warp->lanes[i].pc = pc;
        warp->lanes[i].mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        warp->lanes[i].gpr[0] = 0; /* x0 硬连线为 0 */

        /* 初始化 softfloat 状态：默认 round-to-nearest-even */
        set_float_rounding_mode(float_round_nearest_even,
                                &warp->lanes[i].fp_status);
    }
}

/*
 * ============================================================================
 * gpgpu_core_exec_warp - 执行一个 warp 直到所有 lane 退出或超时
 * ============================================================================
 *
 * SIMT 锁步循环：每轮从 VRAM 取一条指令，对所有活跃 lane 执行，
 * 然后 PC+4。lane 执行 ebreak 后从 active_mask 中移除。
 *
 * @s:          GPGPU 设备状态
 * @warp:       warp 状态
 * @max_cycles: 最大执行周期数（防止死循环）
 *
 * 返回: 0 成功，-1 错误
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;

    while (warp->active_mask != 0 && cycles < max_cycles) {
        /* 所有活跃 lane 的 PC 相同（锁步），从 lanes[0] 取指即可 */
        uint32_t pc = warp->lanes[0].pc;

        /* 从 VRAM 取指令（内部路径，直接指针访问） */
        uint32_t inst = ldl_le_p((uint8_t *)s->vram_ptr + pc);

        /* 对所有活跃 lane 执行该指令 */
        exec_one_inst(s, warp, inst);

        /* 所有活跃 lane 的 PC 前进 4 字节 */
        for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
            if (warp->active_mask & (1u << i)) {
                warp->lanes[i].pc += 4;
            }
        }

        cycles++;
    }

    return 0;
}

/*
 * ============================================================================
 * gpgpu_core_exec_kernel - 执行完整的 kernel
 * ============================================================================
 *
 * 根据 s->kernel 中的 grid/block 维度，遍历所有 block 和 warp，
 * 依次初始化并执行每个 warp。
 *
 * Grid/Block 层次：
 *   Grid(grid_dim[0..2]) 包含多个 Block
 *   每个 Block(block_dim[0..2]) 包含多个 warp
 *   每个 warp 包含最多 32 个 lane
 *
 * @s: GPGPU 设备状态
 *
 * 返回: 0 成功，-1 错误
 */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t grid_x = s->kernel.grid_dim[0];
    uint32_t grid_y = s->kernel.grid_dim[1];
    uint32_t grid_z = s->kernel.grid_dim[2];
    uint32_t block_x = s->kernel.block_dim[0];
    uint32_t block_y = s->kernel.block_dim[1];
    uint32_t block_z = s->kernel.block_dim[2];

    /* 每个 block 的总线程数 */
    uint32_t threads_per_block = block_x * block_y * block_z;
    if (threads_per_block == 0) {
        return -1;
    }

    /* 每个 block 的 warp 数量 */
    uint32_t warps_per_block = (threads_per_block + GPGPU_WARP_SIZE - 1)
                               / GPGPU_WARP_SIZE;

    /* kernel 代码在 VRAM 中的起始地址 */
    uint32_t kernel_addr = (uint32_t)s->kernel.kernel_addr;

    /* 遍历所有 block */
    for (uint32_t gz = 0; gz < grid_z; gz++) {
        for (uint32_t gy = 0; gy < grid_y; gy++) {
            for (uint32_t gx = 0; gx < grid_x; gx++) {
                /* 线性 block ID */
                uint32_t block_id_linear = gz * grid_y * grid_x
                                          + gy * grid_x + gx;
                uint32_t block_id[3] = { gx, gy, gz };

                /* 遍历该 block 的所有 warp */
                for (uint32_t w = 0; w < warps_per_block; w++) {
                    GPGPUWarp warp;

                    /* 该 warp 的活跃 lane 数量：
                     * 最后一个 warp 可能不满 32 个线程 */
                    uint32_t remaining = threads_per_block - w * GPGPU_WARP_SIZE;
                    uint32_t active_threads = (remaining < GPGPU_WARP_SIZE)
                                              ? remaining : GPGPU_WARP_SIZE;

                    gpgpu_core_init_warp(&warp, kernel_addr,
                                         w * GPGPU_WARP_SIZE,
                                         block_id,
                                         active_threads,
                                         w, block_id_linear);

                    gpgpu_core_exec_warp(s, &warp, 100000);
                }
            }
        }
    }

    return 0;
}

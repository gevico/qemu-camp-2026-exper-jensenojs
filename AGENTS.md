# QEMU Camp GPGPU 方向学习指南

## 背景要求

本实验假设你已有系统级软件知识（数据库内核），但不要求有 GPU 或 QEMU 基础。以下是你需要补充的核心概念，按优先级排序。

---

## 一、GPGPU 硬件模型（最高优先级）

### 1.1 为什么要 GPGPU？

CPU 适合分支控制，GPU 适合数据并行。GPGPU 把 GPU 的并行计算能力开放给通用计算。

### 1.2 SIMT 执行模型（最重要）

**SIMT = Single Instruction, Multiple Thread**

GPU 不像 CPU 那样每个核心独立执行。它把一组线程打包成 warp（通常 32 线程），同一个 warp 里的线程永远执行同一条指令，但操作不同的数据。

```
Grid (kernel 调用一次)
├── Block 0 ──── Warp 0: thread 0-31 执行同一条指令，操作不同数据
│           └── Warp 1: thread 32-63 ...
├── Block 1
└── Block 2
```

- **thread**: 最小执行单元，有自己的 PC、寄存器
- **warp**: 32 个线程的集合，同时执行同一条指令
- **block**: 一组线程共享共享内存，可做 barrier 同步
- **grid**: 整个 kernel 的线程组织

**关键直觉**：同一个 warp 里如果线程走不同分支（if-else），需要串行执行分支，再合并——这叫 **branch divergence**，是 GPU 性能杀手。

### 1.3 线程 ID 编码

本实验用 `mhartid` CSR 编码线程 ID：

```
mhartid bit layout: [block_id(19位) | warp_id(8位) | thread_id(5位)]
```

- `thread_id = mhartid & 0x1F` → 0-31
- `warp_id = (mhartid >> 5) & 0xFF` → 0-255
- `block_id = mhartid >> 13` → 0-...

### 1.4 内存层次

| 内存类型     | 位置    | 延迟  | 共享范围      |
| ----------- | ------- | ------ | ------------- |
| 寄存器       | GPU核内  | 1 cycle | thread 私有 |
| 共享内存     | GPU核内  | ~10 cycle | block 内共享 |
| VRAM (显存)  | GPU显存  | ~100 cycle | 全局可见   |
| 主机内存     | CPU侧   | ~更慢     | 需要 DMA      |

**本实验的 VRAM = 64MB，通过 PCIe BAR2 映射**

### 1.5 PCIe BAR 空间

PCIe 设备通过 BAR（Base Address Register）向 CPU 暴露内存区域：

```
BAR0 (1MB): 控制寄存器 MMIO
  - 0x0000-0x00FF: 设备信息
  - 0x0100-0x01FF: 全局控制/状态
  - 0x0200-0x02FF: 中断控制
  - 0x0300-0x03FF: kernel 分发配置
  - 0x0400-0x04FF: DMA 引擎
  - 0x1000-0x1FFF: SIMT 上下文（线程读取自己的 ID）
  - 0x2000-0x2FFF: 同步（barrier）

BAR2 (64MB): VRAM（显存）
  - 存放 kernel 代码和输入/输出数据
  - CPU 通过 PCIe 读写
```

### 1.6 低精度浮点

本实验涉及 GPU 常用低精度格式：

| 格式      | 位宽 | 特点              | 用途        |
| --------- | ---- | ----------------- | ----------- |
| FP32      | 32b  | IEEE 单精度        | 基准格式     |
| FP16      | 16b  | 半精度            | 加速        |
| BF16      | 16b  | Google Brain float，指数8b比FP16多 | 深度学习 |
| FP8-E4M3  | 8b   | 4位指数+3位尾数    | Hopper 新支持 |
| FP8-E5M2  | 8b   | 5位指数+2位尾数    | 范围优先    |
| FP4-E2M1  | 4b   | 2位指数+1位尾数    | 极低精度    |

**往返转换**：float → low-precision → float，期望值不变（或有控制的精度损失）

---

## 二、QEMU 设备建模基础（理解代码必需）

### 2.1 QOM（QEMU Object Model）

QEMU 所有设备都是 QOM 对象，类型系统类似 GObject：

```c
// 定义设备类型（放在 type_init 或构造函数里）
#define TYPE_GPGPU "gpgpu"
OBJECT_DECLARE_SIMPLE_TYPE(GPGPUState, GPGPU)

// 设备结构体（第一个成员必须是父类）
struct GPGPUState {
    PCIDevice parent_obj;  // 必须是第一个

    // 设备特有字段
    MemoryRegion ctrl_mmio;
    MemoryRegion vram;
    uint8_t *vram_ptr;
    ...
};
```

### 2.2 MemoryRegion（地址空间抽象）

设备通过 MemoryRegion 向 CPU 暴露寄存器：

```c
// 初始化 MMIO region
memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ops, s,
                      "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);

// 在 PCI BAR 映射时
pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->ctrl_mmio);
```

读写 MMIO 通过 `MemoryRegionOps` 定义：

```c
// 读寄存器
static uint64_t gpgpu_read(void *opaque, hwaddr addr, unsigned size) {
    GPGPUState *s = opaque;
    switch (addr) {
    case GPGPU_REG_DEV_ID:
        return GPGPU_DEV_ID_VALUE;
    ...
    }
}

// 写寄存器
static void gpgpu_write(void *opaque, hwaddr addr, uint64_t val, unsigned size) {
    GPGPUState *s = opaque;
    switch (addr) {
    case GPGPU_REG_GLOBAL_CTRL:
        if (val & GPGPU_CTRL_ENABLE) { ... }
    ...
    }
}

static const MemoryRegionOps gpgpu_ops = {
    .read = gpgpu_read,
    .write = gpgpu_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {.min_access_size = 4, .max_access_size = 4},
};
```

### 2.3 PCI 设备注册

```c
// 在 type_init 中注册
static void gpgpu_class_init(ObjectClass *klass, void *data) {
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->vendor_id = GPGPU_VENDOR_ID;    // 0x1234
    pc->device_id = GPGPU_DEVICE_ID;    // 0x1337
    pc->revision = GPGPU_REVISION;
    pc->class_id = PCI_CLASS_DISPLAY_3D; // 0x0302

    dc->vmsd = &vmstate_gpgpu;  // 迁移支持
}

static const TypeInfo gpgpu_info = {
    .name = TYPE_GPGPU,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init = gpgpu_class_init,
};

type_init(gpgpu_register_types)
```

---

## 三、本实验 GPGPU 设备架构

### 3.1 组件关系

```
qemu-system-riscv64 -device gpgpu
                        │
                        ├── BAR0 (控制寄存器 MMIO)
                        │     0x0000: DEV_ID = "GPPU"
                        │     0x0100: GLOBAL_CTRL (enable/reset)
                        │     0x0200: IRQ_ENABLE/STATUS
                        │     0x0300: KERNEL_ADDR, GRID_DIM, DISPATCH
                        │     0x0400: DMA_SRC/DST/SIZE/CTRL
                        │     0x1000: thread_id_x/y/z, block_id_x/y/z, warp_id, lane_id
                        │     0x2000: BARRIER, THREAD_MASK
                        │
                        ├── BAR2 (VRAM 64MB)
                        │     kernel 代码 + 输入/输出数据
                        │
                        └── MSI-X 中断
                              4 向量: KERNEL_DONE, DMA_DONE, ERROR
```

### 3.2 dispatch 流程（核心）

```
CPU 驱动:
  1. 写 BAR2 + 0x0000: kernel 代码
  2. 写 BAR2 + 0x1000: 输出数组
  3. 写 BAR0 + 0x0300: kernel 地址
  4. 写 BAR0 + 0x0310-0x0324: grid/block 维度
  5. 写 BAR0 + 0x0330 (DISPATCH): 触发执行  ← 关键！

GPGPU 设备（模拟侧）:
  dispatch handler 收到写:
  → 解析 grid_dim, block_dim, kernel_addr
  → 为每个 warp 创建模拟执行上下文
  → 在 VRAM 中执行 kernel 代码（RISC-V 指令通过 TCG）
  → 设置完成状态，触发 MSI-X 中断
```

### 3.3 SIMT 上下文模拟

每个 warp 执行时，设备需要模拟线程的"视角"：

```c
// 线程读取自己的 thread_id 时，返回模拟的值
case GPGPU_REG_THREAD_ID_X:
    return s->simt.thread_id[0];  // 不是固定值，是动态计算的
```

当 `mhartid` CSR 读取时：
- TCG 生成代码读取 `GPGPU_REG_THREAD_ID_X` → 设备返回当前 warp 的 thread_id
- 这让同一个 kernel 二进制在不同 warp 执行时得到不同结果

---

## 四、测试框架（QTest）

### 4.1 QTest 工作原理

QTest 是 QEMU 的设备级测试框架，不启动完整虚拟机：

```c
// QGPGPU 结构（测试侧）
typedef struct QGPGPU {
    QOSGraphObject obj;
    QPCIDevice dev;     // PCI 设备句柄
    QPCIBar bar0;       // BAR0 映射
    QPCIBar bar2;       // BAR2 映射
} QGPGPU;

// 测试步骤（通用模式）
qpci_device_enable(pdev);
bar0 = qpci_iomap(pdev, 0, NULL);  // 映射 BAR0

qpci_io_writel(pdev, bar0, REG, value);  // 写寄存器
val = qpci_io_readl(pdev, bar0, REG);   // 读寄存器
g_assert_cmpuint(val, ==, expected);     // 断言

qpci_iounmap(pdev, bar0);
```

### 4.2 测试分类

| 类型       | 代表测试          | 验证内容                    |
| --------- | --------------- | --------------------------- |
| 寄存器读写 | device-id, vram-size | MMIO 读写通路                |
| DMA       | dma-regs           | 数据搬运                     |
| SIMT 上下文 | thread-id, warp-lane | 线程 ID 编码               |
| Kernel 执行 | kernel-exec, fp-kernel-exec | 端到端 dispatch + 执行 |

---

## 五、学习路径（按天）

### Day 1: 环境 + 跑通基线

```
make -f Makefile.camp configure
make -f Makefile.camp build
make -f Makefile.camp test-gpgpu
```
记下分数，理解测试输出的含义。

### Day 2: 理解 PCIe + MMIO 基础

读 `hw/gpgpu/gpgpu.c`：
- `gpgpu_class_init` → 设备注册
- `gpgpu_realize` → BAR 映射
- `gpgpu_ops` → MMIO read/write handlers
- 跑 `gpgpu_test_device_id` 手动加 log 理解 MMIO 读写

### Day 3: 理解 Dispatch 流程

读 `hw/gpgpu/gpgpu_core.c`：
- warp 执行循环
- `mhartid` CSR 如何被 TCG 模拟
- `gpgpu_test_kernel_exec` 端到端流程

### Day 4-5: 实现/调试

根据分数，针对性实现缺失的 handler。

### Day 6-7: DMA + 中断

`gpgpu_test_dma_regs` 需要 DMA 引擎实现。
`gpgpu_test_kernel_exec` 完成后需要 MSI-X 中断。

### Day 8-10: FP 支持

RV32F 浮点指令在 TCG 中已有支持，重点是理解 kernel 如何使用 `fcvt.s.w` 等指令。

---

## 六、快速参考

### 寄存器速查

```
0x0000  DEV_ID        只读，0x47505055
0x0004  DEV_VERSION  只读，0x00010000
0x0100  GLOBAL_CTRL   bit0=enable, bit1=reset
0x0104  GLOBAL_STATUS bit0=ready, bit1=busy
0x0200  IRQ_ENABLE   bit0=KERNEL_DONE, bit1=DMA_DONE, bit2=ERROR
0x0300  KERNEL_ADDR  kernel 代码在 VRAM 中的地址
0x0310  GRID_DIM_X   grid X 维度
0x031C  BLOCK_DIM_X  block X 维度（=线程数）
0x0330  DISPATCH     写任意值触发 kernel 执行
0x0400  DMA_SRC_LO   DMA 源地址
0x0414  DMA_CTRL     bit0=START, bit1=DIR
0x1000  THREAD_ID_X  线程在 block 中的 X 索引
0x1010  BLOCK_ID_X   block 在 grid 中的 X 索引
0x1020  WARP_ID      warp 在 block 中的索引
0x1024  LANE_ID      线程在 warp 中的索引（0-31）
0x2004  THREAD_MASK  活跃线程掩码
```

### Kernel 代码模板（simple_kernel）

```assembly
# kernel: C[thread_id] = thread_id
# mhartid = block_id<<13 | warp_id<<5 | thread_id
# 输出地址 = 0x1000

csrrs x6, mhartid, x0    # t1 = mhartid
andi x6, x6, 0x1F        # t1 = thread_id (lane)
slli x7, x6, 2            # t2 = thread_id * 4 (byte offset)
lui x28, 1                 # t3 = 0x1000
add x28, x28, x7           # t3 = &C[thread_id]
sw x6, 0(x28)              # C[thread_id] = thread_id
ebreak                     # 停止
```

### 有问题时的调试顺序

1. 测试 1-7（寄存器）全过吗？→ 检查 MMIO handler
2. 测试 8-12（SIMT 上下文）过吗？→ 检查 simt 上下文赋值
3. 测试 13（kernel exec）过吗？→ 检查 dispatch handler + TCG 执行
4. 测试 14-17（FP）过吗？→ 检查 FPU 支持是否正确连接

---

## 七、推荐阅读顺序

1. 本文档（30分钟）
2. `hw/gpgpu/gpgpu.h`（全部注释，1小时）
3. `hw/gpgpu/gpgpu.c`（MMIO handler，2小时）
4. `tests/qtest/gpgpu-test.c`（理解测试意图，1小时）
5. `hw/gpgpu/gpgpu_core.c`（SIMT 引擎，2小时）

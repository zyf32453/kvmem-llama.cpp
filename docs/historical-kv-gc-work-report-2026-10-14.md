# Historical KV 后台 GC 迁移阶段性工作报告

**日期：**2026 年 10 月 14 日

**目标项目：**`kvmem-llama.cpp`
**参考实现：**本机 `kvmem-qw3` 的 legacy historical-block GC

## 一、工作目标与结论

本次工作核查并尝试迁移 `kvmem-qw3` 中对单个 historical KV block 的后台生命周期管理。核查结果是：迁移前的 `kvmem-llama.cpp` 已有 CPU pinned 缓存的 LRU 淘汰和会话级缓存管理，但没有对任意 historical block 进行后台候选判定、逻辑退休及权威 K/V 回收的完整链路。两种缓存淘汰不能代替历史块 GC。

目前已完成**宿主侧 GC 核心及逻辑块目录的第一阶段迁移**，并通过相关单元测试；尚未接通服务端入口与 `RawKvStore` 的物理释放，因此不能报告为“CPU 历史 KV 已自动回收”。默认推理行为没有变化。这一边界是本报告的主要结论。

## 二、修改前后的机制对比

迁移前，`KvMemStore` 以连续 block ID 记录历史块。选块预算足够时，top-k、prefill pressure 等路径直接返回全部 ID；`RawKvStore` 只支持清空或按 token 位置截断尾部，不能按任意历史 block ID 删除其多层 K/V。已经移出 GPU 工作集的块仍可留在 CPU 上作为后续检索的来源。

迁移后，新增可选的 `KvMemGc`：后台线程按最近使用、衰减频率、检索分数与保护状态扫描块，经过候选观察期后发布带代数的逻辑删除视图。旧读者持有旧视图或工作集 lease 时，释放回调延后；块重新被使用可撤销待执行的退休。该策略并非纯 LRU，也不会选择仍标记为 GPU 驻留、IO 进行中或受保护的块。

`KvMemStore` 现可在追加历史前挂接 GC，并同步块的封闭、层级、CPU 缓存和 IO 状态。top-k、prefill pressure、semantic group、媒体约束路径会跳过已退休的“洞”；显式请求已删除块时失败，检查点辅助接口可识别必需前缀中缺失的块。例如测试中四块 `0、1、2、3` 的预算足以全选，块 `2` 逻辑退休后，新目录的选择结果为 `0、1、3`。旧目录仍能保护退休前已开始的读操作。

## 三、本次代码交付

1. 移植 `kvmem/include/kvmem/kvmem_gc.hpp` 与 `kvmem/src/host/kvmem_gc.cpp`，纳入 `kvmem` 构建。保留来源项目 Apache-2.0 许可文本于 `kvmem/LICENSE.kvmem-qw3`。
2. 扩展 `kvmem/include/kvmem/kvmem_store.hpp` 与 `kvmem/src/host/kvmem_store.cpp`：可选 GC 挂接、目录视图、工作集和检查点 lease、逻辑存活性检查、选块过滤，以及块容量预检。旧窗口 lease 随选择计划保留，避免过早释放。
3. 新增 `kvmem/tests/kvmem_gc_test.cpp`，并在 `kvmem/CMakeLists.txt` 注册测试。
4. 编写 `docs/historical-kv-gc-migration.md`，列明剩余生产接入步骤和验收条件。

本次没有修改 `RawKvStore`、GPU stage-in/stage-out 机制或服务端命令行配置，也没有迁移 `kvmem-qw3` 的另一套固定物理槽位 `KvMemRepositoryGc`。当前 block 元数据仍随历史追加而增长。

## 四、测试方法与结果

在 Windows 上以 `KVMEM_BUILD_LLAMA=OFF` 的便携构建编译，通过 CTest 执行宿主逻辑测试。新增测试分别覆盖：策略优先级；容量不足时追加操作不留下半写入状态；all-fit 三种选块路径不复活已删除块；检查点前缀与显式选择的失败保护；旧视图下延迟释放；晚到 lease 撤销退休；shadow 模式不释放；截断后同一 block ID 重用不被旧删除标记误伤。释放操作在单测中由计数回调模拟，未分配或释放真实 CPU K/V。

最终代码的便携构建成功；8 个可运行的 CTest 测试全部通过，其中新增 GC 测试在最终修改后另连续重复运行 10 次，均通过。完整套件中的 `kvmem-session-cache-lifecycle-test` 在当前 Windows 沙箱遇到符号链接权限及 Python 子进程输出解码问题，未纳入上述 8 项通过统计；该测试不覆盖本次 GC 路径。

## 五、尚未完成及下一步

当前服务端不会创建 `KvMemGc` worker，也未把其释放回调接到 `RawKvStore`。所以现阶段**没有实际回收 CPU 上的历史 K/V 内存**。下一阶段应按以下顺序完成：

1. 给 `RawKvStore` 实现带分配代数校验的任意单块释放，覆盖每层 packed K/V、mean-K 和可选 NVMe 分配；与异步 IO 互斥，忙时延期重试。
2. 在服务端为活跃工作集、异步读写、检索、查询回放和可复用检查点建立完整 view/lease 生命周期；缺失必要历史块时拒绝不安全的检查点恢复。
3. 限定支持配置，先接入 shadow 模式与统计，再用真实 CPU 字节下降、多层数据一致性、并发和长上下文质量测试验收，最后考虑开启物理删除。

`max_live_blocks` 应视为可回收历史块的**软目标**，不能代替 CPU RAM 硬上限。受保护块可能使实际存活数高于目标；CPU 空间不足还需要明确的压力与拒绝策略。

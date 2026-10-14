# Historical KV 后台 GC 迁移方案

## 结论与边界

`kvmem-qw3` 的 legacy GC（`include/qw3/kvmem_gc.hpp`、`src/kvmem_gc.cpp`、`docs/kvmem_gc_v1.md`）会在后台选择**单个 historical block**，先在新目录视图中将其逻辑删除，等旧读者与 lease 退出后再释放该块的原始 K/V 权威存储。它是永久缩减可检索历史，并非把 GPU 块挪到 CPU 的缓存淘汰。它的策略也不是纯 LRU：最近使用、指数衰减频率、检索分数、重复/覆盖证据、anchor 和保护状态共同参与决策。

本项目原有的 `PinnedKvTier` LRU 管理 CPU 缓存，session cache 管理整段会话；两者都不会删除任意 historical block。本次迁移已加入 Qwen3 的独立 CPU worker 和 `KvMemStore` 的可选逻辑块接口，但**未接入服务器的生产删除路径**。没有配置项会启动这套 worker；默认运行行为保持原样。`RawKvStore` 的实际 K/V 释放、服务端的安全 lease、检查点失效和压力策略仍需完成后才能启用真实删除。

## 已实现的迁移基础

1. 以 Apache-2.0 来源文件为基础移植 `KvMemGc`（许可文本见 `kvmem/LICENSE.kvmem-qw3`）：后台增量扫描、按分数排队、观察期、shadow、块代数、不可变目录视图、工作集 lease、迟延释放回调及指标。`max_live_blocks == 0` 表示不开启；开启时由调用方创建 worker 并提供带分配代数校验的非阻塞释放回调。
2. `KvMemStore` 可在追加历史前挂接 GC；追加/截断/换层/IO 状态同步到 worker。top-k、prefill pressure、semantic group 与媒体约束在新目录中跳过已删除的块；显式要求的已删除块会失败，检查点可检查其必要前缀是否完整。
3. 单测覆盖策略优先级、all-fit 分支不复活“洞”、旧目录仍可读、延迟释放，以及晚到 lease 撤销退休。普通构建不创建 worker，不会改变既有选块结果。

## 生产接入顺序

### 1. 固定支持范围与门槛

先在单会话、单 lane、无 MTP、无多模态、无 query replay 的配置下接入。拒绝不支持的组合，默认关闭，并提供 `shadow` 模式。Qwen3 的旧 GC 本身也有这些适用边界；不能直接照搬 CLI 开关后放开全部模式。以“live block 软上限”而非“CPU RAM 硬上限”理解 `max_live_blocks`：受保护块可以使 live 数超过目标，GC 不应强行删除。

### 2. 接入安全生命周期

在请求/推理步骤开始时取得目录 view，在结束时释放；GPU 工作集、前一窗口的 stage-out、异步拷贝、检索索引读取、查询回放和可复用检查点都必须持有相应 block 的 lease。恢复检查点前检查必需的历史前缀；若已有块删除，只能重算/拒绝恢复，不能从空洞拼接旧 KV。对 GPU 仍驻留、IO 进行中、CPU cache 有副本但状态未协调、未封块、sink/recent、其他强制块设置保护位。

### 3. 做真正的物理回收

为 `RawKvStore` 增加按 block 回收的事务接口：首先锁定并确认该 block 的分配代数、没有排队/进行中的读写，随后释放每一层的 packed K、packed V、mean-K 及可选原始 F16 数据，并同步释放 NVMe 分配；若忙则返回 false 供 worker 下轮重试。截断、清空、同一 block ID 重用时递增代数，防止旧回调误删新块。当前实现只有 suffix `truncate_to`，没有任意块删除，因此不能把现有 `clear` 或 `truncate_to` 当作 GC 回调。Windows 没有 POSIX NVMe tier；在 CPU-only 权威模式下释放后数据不可恢复，这是有意的历史删除。CPU 空间不足时需要显式的写入压力/拒绝策略；后台 GC 仅能回收符合条件的旧块，不能保证硬上限。

### 4. 连接证据与服务端开关

为 worker 提交同一代数的检索得分事件、选中标记和实际使用事件；低分证据缺失时按普通块处理，不把未知当低分。只有可证明 durable 的 K/V 才能成为候选。提供统计与 trace（扫描、候选、撤销、逻辑退休、物理回收、读者延迟、失败），先 shadow 对比检索质量，再小规模开启删除。

### 5. 验收

除现有宿主单测外，还需测试：CPU `RawKvStore` 真正下降的字节数、多层 K/V 一致性、异步写入竞态、旧 view/lease 下不释放、ID 重用和 truncate/clear 的 ABA、检查点恢复、查询回放、短尾块、media group、持久化 session 与多 lane 隔离。随后在长上下文压测中比较 live block 数、CPU 内存、检索准确性和回收延迟。生产配置只有在这些用例通过后才开放。

## 与 Qwen3 新 bounded repository 的区别

`kvmem-qw3` 另有 `KvMemRepositoryGc` 和固定物理槽位方案（`docs/kvmem_bounded_repository.md`）。它解决历史 metadata/slot 随总追加量持续增长的问题，涉及 `(slot, generation)` 句柄与整个索引的改造。本次先迁移 legacy 的后台单块 GC；即便完成上述生产接入，`KvMemStore::blocks_` 的逻辑目录仍随历史追加增长，不能宣称已经获得 bounded repository 的固定内存上限。

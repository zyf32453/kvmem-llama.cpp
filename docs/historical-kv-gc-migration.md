# Historical KV 后台 GC 迁移方案

## 结论与边界

`kvmem-qw3` 的 legacy GC（`include/qw3/kvmem_gc.hpp`、`src/kvmem_gc.cpp`、`docs/kvmem_gc_v1.md`）会在后台选择**单个 historical block**，先在新目录视图中将其逻辑删除，等旧读者与 lease 退出后再释放该块的原始 K/V 权威存储。它是永久缩减可检索历史，并非把 GPU 块挪到 CPU 的缓存淘汰。它的策略也不是纯 LRU：最近使用、指数衰减频率、检索分数、重复/覆盖证据、anchor 和保护状态共同参与决策。

本项目原有的 `PinnedKvTier` LRU 管理 CPU 缓存，session cache 管理整段会话；两者都不会删除任意 historical block。本次迁移已加入 Qwen3 的独立 CPU worker 和 `KvMemStore` 的可选逻辑块接口，但**未接入服务器的生产删除路径**。没有配置项会启动这套 worker；默认运行行为保持原样。下文记录了当前新增的仓库/策略解耦与受限的 RAM 物理释放接口；服务端的安全 lease、检查点失效和压力策略仍需完成后才能启用真实删除。

## 2026-10-16 阶段进度

在早期 `KvMemGc` 移植之外，已增加按 `(slot, generation)` 管理块生命周期的 `HistoricalKvRepository`、独立的 `HistoricalKvGc` 管理器和 `HistoricalKvPolicy` 接口。LRU 是一个可替换策略，管理器负责扫描、退休、等待已有 lease、重试及触发物理释放。`HistoricalKvRawStore` 提供按句柄存储的独立验证实现；`HistoricalKvActiveSet` 为每次操作重新取得执行许可，跳过已退休的可选块，必选块缺失时撤销整次申请，并只在实际执行结束后记录最近使用时间。

真实 `RawKvStore` 现有一个**受限桥接接口**：只有完整、对齐、所有必需层 K/V 均已连续写入、无 NVMe 的 RAM 块可以由仓库 writer 封存。仓库将该句柄退休、所有旧 lease 退出并进入 `Releasing` 后，后台回调可逐层移走 K、V、mean 的向量，在存储锁外销毁，并返回实际释放的 payload 容量。接口验证仓库身份和代数，禁止封存后的旧位置写入。测试已覆盖旧读者阻止释放、两层数据释放、同槽新代数、跨仓库同值句柄、增量写入空洞以及管理器到真实 RAM 存储的回收闭环。

适配器现将模型真正拥有注意力 KV 的物理层号传给 `RawKvStore`。受限封存只要求这些层 K/V 写满，同时要求非注意力层保持空白；检索模式还要求这些层的 Mean-K 连续写满。未显式给层号的旧调用仍要求全部配置层写满。这样混合架构中没有 KV 的层不会让完整块永远无法封存。新增用例验证稀疏层的封存与物理释放、缺失 Mean-K、非注意力层意外写入和无效层号列表。生产封存仍未启用：当前位置式写入和 snapshot/truncate 路径尚不能在已封存块上安全运行。

这里计量的物理释放仅是 `RawKvStore` 的 K/V/mean 权威 payload。`PinnedKvTier` 可能仍缓存该块的 GPU 格式副本，且其 CPU arena 预先分配；生产删除还需在所有 transfer lease 结束后同步失效对应缓存槽。槽位可复用、payload 容量下降和进程 RSS 下降是三个不同指标，不应混为一次“CPU 内存释放”。

独立的按句柄 `HistoricalKvRawStore` 也将读写许可和元数据观察绑定到仓库身份，物理释放要求仓库进入 `Releasing`。这防止调用方绕过退休与旧 lease 清空阶段直接释放，也防止两个仓库使用相同外部 incarnation/slot/generation 时互相读取、写入或回收数据。

该按句柄存储现支持 `write_slice`：在明确的原始位置与代数下逐层连续追加 K/V/mean，拒绝跳写、重复写及长度不符的片段；整层完成前不会作为可读的完整层返回。`publish_complete` 仅在每层写满预留 span 时发布。策略可在片段写入途中退休块，已有 writer 可以完成其片段或直接放弃，后台都会在 lease 退出后回收该代 payload；一旦发布 Live，writer 不能再改写。这是 Host 存储接口的验收，生产 adapter 尚未调用它。

对真实采集路径的审查发现，Mean-K、packed K 和 packed V 会由不同的调用或异步任务分别到达。按句柄存储因此增加 `write_mean_k_slice`、`write_packed_k_slice` 和 `write_packed_v_slice`：每个分量独立记录连续前缀，顺序可以不同；所有必需分量和所有配置层写满以前，整层读取与 `publish_complete` 均拒绝。原 `write_slice` 仍在同一 slot 锁下验证并一次提交全部分量，不会因后一分量错误而留下半次写入。测试覆盖错序完成、缺分量/缺层、洞与重复、非对齐短 span、退休后旧 writer 完成、旧 reader 阻止回收，以及跨仓库同值句柄。

`KvMemRuntime` 现有一个默认空的 `HistoricalKvShadowDirectory`，随 runtime 一起归属和移动到对应会话。它只允许把完整的现有 `KvMemBlock` 与同一仓库的精确 writer/handle/span 绑定，供将来异步任务在提交时捕获原代数；绑定失败不会消耗 writer。通过 runtime 追加/截断时，若会改变已绑定跨度就先拒绝，避免密集 block ID 被悄悄复用。目录绑定长期持有 writer lease，因此退休后若不解除绑定并等待异步 writer 退出，后台释放会一直延后。解绑仅把目录状态改为 `Retired` 并放掉目录 writer；仓库确认同代物理释放后，目录才可改为 `Reclaimed`。两个状态都保留关闭墓碑，`find()` 不返回可用绑定；只有运行时实际截断并移走旧密集块后，才删除墓碑并允许编号重用。该状态机避免“payload 已释放而旧块尚在目录”时把它误认为未托管块。目录仍**不参与当前选块、执行许可、退休或物理释放**，也尚未由适配器使用。`runtime.store()` 暴露的直接修改入口目前可绕过此预检，生产接线前必须收口。仓库/worker、异步 writer 与 Raw 存储还需统一销毁顺序，不能让后台释放回调捕获先于其销毁的裸 `RawKvStore` 指针。

**此桥接尚未接入真实模型推理。** 现有 `KvMemStore` 和 `RawKvStore` 的生产读取仍依赖 `pos / block_tokens` 的密集位置；检索选择、GPU stage、异步传输、checkpoint、session 和 MTP 路径未统一持有逐块许可。封存时 `truncate`、`clear` 和 snapshot 也会拒绝操作，NVMe、短尾块及共享位置块不受支持。由此不能打开生产删除开关，也不能把单测中的释放字节数当作实际模型运行的内存收益。下一步需要把生产目录与所有读写路径迁到稳定句柄，补足这几条路径的 lease 和恢复语义，再做远程模型验收。

生产接线的首个受限配置应是单会话、单 lane、文本、RAM 权威存储、MTP 关闭且所需注意力层与 packed V 均明确可用。会话对象需携带 `block_id → (handle, original span)`，异步 harvest job 必须在提交时捕获 exact handle 和 writer lease；完成时不能按 `pos` 查询可能已经复用的块。当前 stage-out 可能从 offset 0 重写增长中的 partial block，接线时要只提交未写的后缀。`valid_tokens` 在预约时固定，因此尾块增长/退休切换、非对齐新 span、请求级容量预约也还需要设计和验证。`score_retrieval` 在选块前读取 Mean-K，须先取得 Read 许可；stage-in/prefetch、同 query、全驻留、keep-selected 及压力换块都要取得新的执行或搬运许可，旧许可保留到 GPU/异步 D2H 真正完成。

在正式绑定异步 writer 前，已修正两个现有错误路径：D2H 提交失败时保留位置队列供同步回退，并在回退读图张量前完成 scheduler 同步；CUDA stage-in 的直接 H2D fallback 等待 stream 拷贝完成才返回，避免调用者的局部 packed 缓冲区先析构。这些修复不构成 GC 生产接入。

实际接线还要拒绝没有注意力层的配置，并保证 Mean-K 每段只提交一次：连续前缀证明覆盖范围，不能识别重复累加。`Writing` 块尚未封存，不能被生产删除策略选中。退休和确认回收后的目录墓碑都必须由读取端识别为关闭状态，禁止按“找不到绑定”退回旧位置读取；现有 `KvMemStore::is_live` 仅连接旧 `KvMemGc`，尚不会自动过滤新仓库退休块。`seq_rm`、清空、截断、回放和快照要在修改 GPU/Raw 前完成统一退休与异步排空；退休的 GPU 块再次 stage-out 时也不能重写已释放的 Raw 位置。

### 本阶段验收记录

提交 `97c1672` 在本地 MinGW 环境通过 11 项相关 CTest；远程隔离宿主构建通过全部 20 项 CTest，影子目录和 Raw RAM 回收的 ASan/UBSan 两项测试通过，CUDA CLI 与 server 均完成构建。使用提供的 GGUF、单张 CUDA GPU、`--kvmem-budget 96`、`--kvmem-cpu-gb 1` 和 356-token 提示词生成 8 个 token 成功；`KVMEM_TRACE` 明确记录 9 次 `stage_out_cpu`。这一轮在加入稀疏注意力层和检索 Mean-K 封存检查后重跑，仍只验证**原有 GPU 工作集溢出到 CPU 的路径**及新代码未破坏该路径；影子目录仍为空，后台 GC 未启动，也没有实测历史块的物理删除或 CPU 释放量。

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

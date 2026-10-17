# Historical KV 后台 GC 迁移方案

## 结论与边界

`kvmem-qw3` 的 legacy GC（`include/qw3/kvmem_gc.hpp`、`src/kvmem_gc.cpp`、`docs/kvmem_gc_v1.md`）会在后台选择**单个 historical block**，先在新目录视图中将其逻辑删除，等旧读者与 lease 退出后再释放该块的原始 K/V 权威存储。它是永久缩减可检索历史，并非把 GPU 块挪到 CPU 的缓存淘汰。它的策略也不是纯 LRU：最近使用、指数衰减频率、检索分数、重复/覆盖证据、anchor 和保护状态共同参与决策。

本项目原有的 `PinnedKvTier` LRU 管理 CPU 缓存，session cache 管理整段会话；两者都不会删除任意 historical block。早期移植的 Qwen3 `KvMemGc` 未接入服务器生产删除路径。新的 repository/manager 已有**受限 CLI 显式入口**，可在单请求 recency/RAM 配置下运行 off、shadow 或 delete；默认行为仍是 off。服务器、多请求及其它配置的安全接线尚未完成，不能把这个入口称为整套生产 GC。

2026-10-17 新增 CLI 的显式实验入口 `--kvmem-historical-capture N`：普通模型请求可以自动预约 RAM、创建按句柄历史块并完成 packed K/V 采集。该参数单独使用时删除关闭；配合 `--kvmem-historical-gc-mode shadow|delete` 才启动新管理器。当前仅支持单请求、单 GPU、单序列文本、recency、flash attention、RAM，不等于整请求资源准入。默认不加该参数时仍走原路径。

后续实施以导师的 `historical-kv-gc-implementation-lru-policy-template.md`（2026-10-15）为目标：策略可以退休 Writing、未封口、active 和在途块；机制只延后物理释放，不能以年龄、驻留或引用数否决退休。旧 `KvMemGc` 的保护/rescue 行为仅留作原始移植与回归参考，完整交付使用新的 repository/manager 协议。完整交付还要求长期元数据有界、真实请求预约和实际模型/agent 对照证据，不能用受限完整块测试替代这些要求。

## 2026-10-17 阶段进度

在早期 `KvMemGc` 移植之外，已增加按 `(slot, generation)` 管理块生命周期的 `HistoricalKvRepository`、独立的 `HistoricalKvGc` 管理器和 `HistoricalKvPolicy` 接口。LRU 是一个可替换策略，管理器负责扫描、退休、等待已有 lease、重试及触发物理释放。`HistoricalKvRawStore` 提供按句柄存储的独立验证实现；`HistoricalKvActiveSet` 为每次操作重新取得执行许可，跳过已退休的可选块，必选块缺失时撤销整次申请，并在实际模型计算成功提交时记录最近使用时间。该调用现已接到受限 recency CLI 请求；server 和其它入口尚未覆盖。

真实 `RawKvStore` 现有一个**受限桥接接口**：只有完整、对齐、所有必需层 K/V 均已连续写入、无 NVMe 的 RAM 块可以由仓库 writer 封存。仓库将该句柄退休、所有旧 lease 退出并进入 `Releasing` 后，后台回调可逐层移走 K、V、mean 的向量，在存储锁外销毁，并返回实际释放的 payload 容量。接口验证仓库身份和代数，禁止封存后的旧位置写入。测试已覆盖旧读者阻止释放、两层数据释放、同槽新代数、跨仓库同值句柄、增量写入空洞以及管理器到真实 RAM 存储的回收闭环。

适配器现将模型真正拥有注意力 KV 的物理层号传给 `RawKvStore`。受限封存只要求这些层 K/V 写满，同时要求非注意力层保持空白；检索模式还要求这些层的 Mean-K 连续写满。未显式给层号的旧调用仍要求全部配置层写满。这样混合架构中没有 KV 的层不会让完整块永远无法封存。新增用例验证稀疏层的封存与物理释放、缺失 Mean-K、非注意力层意外写入和无效层号列表。生产封存仍未启用：当前位置式写入和 snapshot/truncate 路径尚不能在已封存块上安全运行。

这里计量的物理释放仅是 `RawKvStore` 的 K/V/mean 权威 payload。`PinnedKvTier` 可能仍缓存该块的 GPU 格式副本，且其 CPU arena 预先分配；生产删除还需在所有 transfer lease 结束后同步失效对应缓存槽。槽位可复用、payload 容量下降和进程 RSS 下降是三个不同指标，不应混为一次“CPU 内存释放”。

独立的按句柄 `HistoricalKvRawStore` 也将读写许可和元数据观察绑定到仓库身份，物理释放要求仓库进入 `Releasing`。这防止调用方绕过退休与旧 lease 清空阶段直接释放，也防止两个仓库使用相同外部 incarnation/slot/generation 时互相读取、写入或回收数据。

该按句柄存储现支持 `write_slice`：在明确的原始位置与代数下逐层连续追加 K/V/mean，拒绝跳写、重复写及长度不符的片段；整层完成前不会作为可读的完整层返回。`publish_complete` 仅在必需注意力层写满预留 span 时发布。策略可在片段写入途中退休块，已有 writer 可以完成其片段或直接放弃，后台都会在 lease 退出后回收该代 payload；一旦发布 Live，writer 不能再改写。该接口的 Host 验收已完成，适配器现已开始接入其分量写入接口，详见下文的异步 packed K/V 接线。

对真实采集路径的审查发现，Mean-K、packed K 和 packed V 会由不同的调用或异步任务分别到达。按句柄存储因此增加 `write_mean_k_slice`、`write_packed_k_slice` 和 `write_packed_v_slice`：每个分量独立记录连续前缀，顺序可以不同；所有必需分量和所有配置层写满以前，整层读取与 `publish_complete` 均拒绝。原 `write_slice` 仍在同一 slot 锁下验证并一次提交全部分量，不会因后一分量错误而留下半次写入。测试覆盖错序完成、缺分量/缺层、洞与重复、非对齐短 span、退休后旧 writer 完成、旧 reader 阻止回收，以及跨仓库同值句柄。

`KvMemRuntime` 现有一个默认空的 `HistoricalKvShadowDirectory`，随 runtime 一起归属和移动到对应会话。早期接口只绑定完整块；现在 `bind_historical_block` 也接受由 `HistoricalKvRawStore` 承载的短块和非对齐 span，legacy Raw 桥接仍限于完整对齐块。绑定验证同一仓库、精确 writer/handle/span 和实际 payload，失败不会消耗 writer；异步任务及读写许可保留原存储 owner。绑定后的范围固定，追加转到新块，截断不能缩短尚未安全回收的旧块。目录现仅保留 producer 的弱引用，创建 writer 通过显式 `Producer` 输出交给请求方。请求方把它交给实际任务后归还自己的引用，物理回收只等待这些任务及已获准读/搬运/执行许可结束，不再等待前台解绑。目录独立持有存储 owner，writer 完成后 Live 数据仍能获得读取许可。`unbind_retired` 仅是可选的退休状态记录；即使未调用它，仓库确认同代物理释放后也可直接确认 `Reclaimed`。退休或回收完成的代数始终拒绝新读取，`find()` 的存在及其中 writer 是否为空都不能作为读取许可；只有运行时实际截断并移走旧密集块后，才删除墓碑并允许编号重用。该状态机避免“payload 已释放而旧块尚在目录”时把它误认为未托管块。

宿主 `KvMemStore` 的现有选块入口现可查询这个目录：仓库刚提交退休、尚未由前台解绑时，下一轮 top-k、prefill pressure、显式选块和必需前缀检查已把该代看作不可选；解绑后及确认回收后的墓碑也继续阻止重新选中。此检查只是**选块预过滤**，不会取得覆盖 GPU 执行的 ExecLease，也不阻止尚未接入目录的 adapter 快捷路径直接访问 Raw；目录尚未由生产适配器绑定或启动后台删除。追加/截断预检现已下沉到 `KvMemStore`，直接调用 `runtime.store().truncate_to()` 也不能绕过原代数的保护；确认释放后，实际删除密集块时才清除墓碑。截断结果的容量在修改块表前预约，避免返回列表分配失败造成块表已删、墓碑未更新。生产操作仍应走 runtime，由它排空预取并清理 tier 槽位；store 的直接方法只处理元数据。仓库/worker、异步 writer 与 Raw 存储还需统一销毁顺序，不能让后台释放回调捕获先于其销毁的裸 `RawKvStore` 指针。

目录增加逐块 `acquire_read` / `acquire_transfer`：未托管块走原路径；已绑定块必须通过仓库对精确 handle 的原子许可申请；Writing、退休、回收完成及关闭仓库均不授予新许可，失败不能回退为位置式读取。生产 `score_retrieval` 已在 `has_block` 前取得 ReadLease，持有到该块所有层的 Mean-K 打分结束；`write_block_to_gpu` 已在检查 Raw 前取得 TransferLease，已托管块暂走同步 H2D，确保搬运完成后才释放许可。目录默认仍为空，因此这两个入口保留原运行行为。其他读取、缓存搬运、GPU 快捷执行、checkpoint/MTP 入口尚未全部接线，不能据此开启生产 GC。

新增测试把目录许可、管理器和真实 RawKvStore 串联：退休立即拒绝新读与新搬运，旧 Mean-K 读取和旧 packed K/V 拷贝可以完成；分别释放 ReadLease 和 TransferLease，只有最后一个许可退出后才看到两层 CPU payload 容量下降。另有用例覆盖未发布/关闭仓库拒绝访问、移动后的许可对象失效、回收墓碑不回退 legacy，以及直接 store 截断的状态不变与编号重用。提交 `94b2301` 已通过本地相关 11 项 CTest、远程 Linux 全部 20 项宿主 CTest，以及目录、真实 Raw 桥接、仓库和 RAM 释放共 4 项 ASan/UBSan；CUDA CLI/server 编译通过。

新增 `tests/historical-access-test.cpp`，以 `build-cuda/bin/kvmem-historical-access-test /path/to/model.gguf` 手动执行，无模型参数时 CTest 明确跳过。它加载用户提供的 GGUF 建立真实模型的 GPU KV 布局，向 Raw 写入已知的 K/V 字节及 Mean-K，并手动绑定一个完整块，直接调用生产 `score_retrieval` 和 `write_block_to_gpu`。在远程单张 CUDA GPU 上已通过 48 个注意力层的逐字节检查：Live 块可完成同步 stage-in；退休后、前台尚未解绑时，打分已跳过它、stage-in 已拒绝它；旧读者保留期间后台回收延后，退出后释放 **3,244,032 字节** Raw payload，另一个未托管块仍在，GPU 字节与槽号均未变化。该数值是受控 fixture 的 CPU payload 容量差，不是进程 RSS 降幅；K/V 是测试填入的已知数据，尚不代表正常推理的异步生产、选块、执行和后台 GC 已形成完整闭环。

同版 CUDA CLI 另用既定 recency 配置完成 356-token 提示词与 8-token 生成，日志记录 9 次 `stage_out_cpu`；这项回归仍只验证原有 GPU→CPU 溢出流程。远程所有改动和测试均位于 `/tmp` 隔离副本，原 `/home/yifan/kvmem-llama.cpp` 工作树状态未变。

### 异步 packed K/V 接线

`HarvestVJob` 现可携带提交时捕获的 writer context、payload store 和 token offset。D2H slab 完成及同步 fallback 均通过同一提交函数写回原 handle，不在完成时重新查 block ID。已退休或已回收块不会因为 legacy Raw 缺数据而重新采集。每层 K/V 分别查询连续写入前缀，只复制缺失后缀；批次逐项记完成位，后续项异常重试时不重复提交之前成功的 write-once 片段。

新 payload store 支持显式物理注意力层列表：非注意力层不分配 payload、不接受写入，也不妨碍完整块发布。`write_progress` 需要原 writer 许可；`copy_mean_k` 只读取均值，避免检索打分为了几个均值元素复制整个 packed K/V。目录绑定新存储后，生产 scorer 和 stage-in 也会使用该 owner 的 exact-generation 读接口，不回退到当前位置式 Raw。

GPU 定向测试现增加三轮：第 1 轮先写入不同长度的 K/V 前缀，异步 D2H 补齐后检查 48 层字节、打分及 GPU 恢复；第 2 轮在队列仍有任务时退休并解绑 Writing 块，后台必须延期，旧任务完成后才归还真实 payload；第 3 轮复用同一物理槽位为新 generation，旧 handle 不能读取新数据。三轮全部通过，分别使用 generation 1/2/3，托管数据没有写入 legacy Raw。最终代码另通过本地 11 项、Linux 20 项宿主测试，5 项 ASan/UBSan，以及 CUDA CLI/server 构建。同版再次通过 356-token 提示词、8-token 生成的 recency 回归，记录 9 次 `stage_out_cpu`；原远程项目工作树状态未变。

提交 `48469a1` 完成上述 packed K/V producer 与对应读取端接线；该提交的 Mean-K 仍由 fixture 直接填入。后续 Mean-K 接线见下节。目录与仓库仍由 fixture 创建，自动创建/发布身份、partial tail、请求预约和全部 ActiveId 入口尚未完成，后台策略仍未在正常推理中启动。

### Mean-K 采集与解码累加接线

`prepare_ubatches` / `note_ubatch_pos` 现在把原始位置、连续片段、原 writer context 一起捕获成 `CaptureBatch`。片段按位置连续性和块边界拆分；已退休、已回收和已发布的托管块关闭新写入，不回退 legacy Raw。每个 D2H slot 保存自己的 batch，worker 不再把自己的位置覆盖到共享的 `cur_pos_`。CUDA D2H、同步 fallback 和多设备批量返回使用相同的原代数提交接口；失败回退仍保留原队列中的 batch。

Mean-K 与 packed K/V 一样只写原代数尚缺失的连续后缀；重复采集不会重复累加，缺少前缀或维度不符会拒绝提交。D2H 逐层记录完成状态，后续层失败重试时不会重复提交已完成的层。绑定存储的 Mean-K 不再通过当前位置式 Raw 写回。

解码累加器保留原 writer，直到实际接受的 token 汇总写回；丢弃 draft 只移除未接受批次，不能丢掉已接受的累加结果。不同代数或不连续范围不能混入同一累加器。退休前已接受的片段可以写回，退休后新提交的范围不能继续追加到旧代。托管解码在开始累加前校验必需注意力层、维度及 token 范围；缺层不能冒充完整覆盖。另修正了已有的 host→GPU 输入切换问题：一旦累加退回 CPU，后续输入继续并入 CPU sum，避免用新的 GPU sum 覆盖此前结果。

测试将前述手填 Mean-K 改为真实 GPU/CPU 张量输入，调用生产采集与解码函数。使用已知数据逐层核对 Mean-K sum、token 数及 packed K/V；并验证跨块且不连续的原始位置、重复采集、D2H 在途退休、解码累加器持有退休 writer、只接受部分 draft、CPU/GPU 输入切换、回收后拒绝重新采集，以及同一物理槽位连续 5 代复用。并发时序由明确的提交/提交完成屏障控制，没有用 sleep 猜测完成时机。张量是 fixture 构造的数据；这仍不等于正常推理已经自动创建托管历史并运行 GC。

真实模型检查还发现：提供的 GGUF 是 `qwen3moe`（30B.A3B），现有补丁仅给 `qwen3` / `qwen35` 图加了采集入口。未修复时，容量足够的 retrieval 推理能够完成，但日志的 decode Mean-K 为 `n_tok=0, n_miss=8`，不能算采集验收通过。新增 `patches/0006-qwen3moe-capture.patch`，在 Q/K 归一化后、RoPE 前注册采集，并注册 V；`scripts/apply-patches.sh` 同步纳入该补丁，在隔离副本连续执行两次已验证只应用一次。其他模型架构仍须逐一确认入口，不能从某个模型能生成文本推断它支持完整 historical KV 采集。

本轮最终验证：本地相关宿主 CTest **11/11**、Linux 宿主 CTest **20/20**、CUDA CLI/server 构建及 GPU 定向测试全部通过。新增补丁在固定上游 `7fe450e19` 的 LF/CRLF 文件上均通过正向应用与反向检查；Windows CUDA 构建脚本通过语法解析，CUDA/HIP 两个 Windows 入口均已接入补丁，但未执行 Windows GPU 构建。

实际 GGUF 在 `retrieval --kvmem-budget 512` 下完成 **356-token 预填充 + 64-token query replay + 8-token 生成**。日志记录 3 次异步预填充采集，每次有 48 层 K；最终 decode 为 `n_tok=8, n_flush=1, n_gpu=384, n_miss=0, last_layers=48`。同版 `recency --kvmem-budget 96` 完成 356-token 预填充与 8-token 生成，并记录 9 次 `stage_out_cpu`。这验证了实际模型的采集入口和既有推理路径；仍不证明生产 GC 删除、长期有界或较小 retrieval 容量已通过。原始日志保存在隔离副本 `/tmp/kvmem-gc-99bef27.3CD9MU/` 的 `historical-mean-gpu-test.log`、`model-mean-retrieval-capture.log`、`model-mean-recency.log` 与 `historical-mean-host-test.log`。

**GC 托管身份尚未在正常推理中自动启用。** 块表查找、语义/媒体分组及托管采集已支持显式原始范围，但 legacy `RawKvStore`、容量快捷证明及部分会话/MTP 路径仍依赖对齐密集位置；检索选择、GPU stage、异步传输、checkpoint、session 和 MTP 路径未统一持有逐块许可。封存时 `truncate`、`clear` 和 snapshot 也会拒绝操作。短尾块的受控接线见下节，NVMe 与完整会话生命周期仍待支持。由此不能打开生产删除开关，也不能把单测中的释放字节数当作实际模型运行的内存收益。

生产接线的首个受限配置应是单会话、单 lane、文本、RAM 权威存储、MTP 关闭且所需注意力层与 packed V 均明确可用。目录、packed harvest job 与 Mean-K batch 已携带 `block_id → (handle, original span)` 及原 writer；正常请求的自动资源预约和身份创建仍需接入；采集任务现已在最后一个必需分量完成时自动尝试完整发布，详见后文。`valid_tokens` 在预约时固定，当前实现以固定 span + 新块切换保持该约束；不能把它当作可无限扩展的尾块。`score_retrieval` 与直接 stage-in 已取得 Read/Transfer 许可；prefetch、同 query、全驻留、keep-selected 及压力换块等执行入口仍需接线，旧许可应保留到 GPU/异步 D2H 真正完成。

### 短尾块与非对齐追加

新增 `register_historical_append(repository, storage, writers, producers)`：调用方先取得已有 writer 和实际 payload，该入口验证连续原始位置、逻辑编号、仓库/代数和有效长度，再把块表与身份目录一起登记。所有可能失败的元数据分配先完成，成功才移动 writer；中间一个 span 无效不会留下半张块表或消耗先前 writer。它不分配历史 payload、不代替请求准入、不启动 GC，也不在失败时由前台回收历史存储。退休前已取得的 writer 仍可登记其获准范围，但不能因此重新获得历史可选资格。

已绑定的短尾块不再随 `register_append` 增长；Writing、Live、Retired 和 Reclaimed 都保持原范围。新块从旧范围末端继续，原位置不补齐、不重排。例如 `[0,3)`、`[3,8)` 后，后一块退休，后续可写入新 handle 的 `[8,35)`。当前选择是绑定即固定范围，尚未实现同代可增长尾块的优化；小批次可能产生更多块，必须由后续整请求预约计入真实碎片上界。

块查找改为按 `orig_pos_start` 二分定位，必需前缀、语义分组与媒体组均使用实际范围。追加容量检查按实际新块数计算，先检查原位置溢出和申请元数据容量，再修改尾块；不能以 `ceil(total_tokens/B)` 漏算短块。Mean-K 捕获同时记录逻辑块号和范围末端，decode 在真正的末端 flush；一个非对齐 span 穿过旧 `pos/B` 边界时不会被误拆到别的块。固定短块也进入异步 packed harvest，不再必须等到 B 个 token。

未经绑定的非对齐数据不会回退到 legacy 采集、打分或直接 stage-in。托管目录存在时，旧 `can_append` / `selection_fits` / query replay 容量快捷证明拒绝通过，MTP 不再猜测尚未登记位置的 GPU 槽；这些入口等待整请求预约和对应执行许可接入。这里只关闭未实现的快捷证明，不代表其后所有普通执行路径已经安全支持 GC。

新增宿主测试覆盖批量登记失败的原子性、退休中的短尾块、迟到 writer 完成但不能重新发布、非对齐位置/分组、容量碎片与位置溢出。GPU 定向测试扩展到 48 层的短 span 和跨 32-token 边界，验证真实生产采集与按句柄恢复；测试所用 K/V 和 Mean-K 输入仍为已知 fixture 张量，不能当作正常模型托管 GC 闭环。

本轮本地相关 CTest **12/12**、Linux 宿主 CTest **21/21**、宿主 ASan/UBSan **9/9** 通过，CUDA CLI 与定向测试完成编译。短尾块 GPU 用例验证 `[3,8)` 在异步采集中退休，原写入完成后释放 CPU payload，再以同一物理槽的新 generation 存储 `[8,35)`；逐层检查所有 GPU 字节，包括退休块和未使用单元，均未被 CPU 回收改变。完整块原有 5 代复用用例继续通过。ASan/UBSan 覆盖宿主模块，未覆盖 CUDA adapter。

实际 GGUF 的 retrieval（预算 512）再次完成 356-token 预填充、64-token query replay 和 8-token 生成，Mean-K 为 48 层、`n_miss=0`；recency（预算 96）完成同一提示词和 8-token 生成，记录 9 次 `stage_out_cpu`。这是默认未托管路径的回归，不是 GC 效果对照。日志保存在远程隔离副本 `/tmp/kvmem-gc-99bef27.3CD9MU/` 的 `historical-tail-{host,gpu,asan}-test.log` 和 `historical-tail-model-{retrieval,recency}.log`。远程原工作仓库未修改。

### 请求的实际 RAM 预约

仓库新增请求私有的 `Reserved` 状态和可移动 `Reservation`。`try_reserve_slots` 以有界扫描一次取得全部槽位；不足就归还已取得的私有额度，不等 GC。Reserved 不进入策略快照，不授予读/执行许可，也不计作 Ready 或待回收 pending；范围观察单列 reserved 数量。`begin_reserved_write` 直接消费已经占有的槽位，不再扫描或申请槽位内存。未用额度在取消/结束时归还，已转成 Writing 的额度继续按原 writer/退休/回收协议结算。仓库关闭、外来 reservation、非法原始范围均不能消耗额度。

`HistoricalKvRawStore::prepare_payload` 在准入阶段真正分配各注意力层的 packed K/V 和 Mean-K 向量。`install_payload` 只把已准备的内存交给确切 writer/handle；短 span 缩短有效长度，但保留并计量所占容量，物理回收才归还容量。全池同时存在的私有 payload 与已安装 payload 合计受固定槽位数限制，计数包含二者。错误存储 owner、错误范围或已占用槽位不会吞掉准备好的 payload。未用 payload 可以随预约安全取消；历史 payload 仍由后台释放，在槽锁之外销毁后才归还计数。

新增 `HistoricalKvRamReservation` 组合真实槽位、full-slot payload 与实际 scratch 分配。设总输入/最大接受输出 token 为 T，最多非空追加次数为 k=`min(T,max_append_boundaries)`，固定 span 所需槽位的最坏上界为 `k + floor((T-k)/B)`，另加未来额外 in-flight 分配；它不借用旧尾块剩余位置，也不把未来 GC 回收算进可用空间。测试穷举小规模 token 分拆与公式对照，每 token 删除尾块时会按 T 个槽位预约。未确定输出上限、缺少追加次数上限和 streaming 均由该 API 明确拒绝。

这只是**整请求准入的 RAM 资源组件**，尚未接成 CLI/server 的 `try_reserve_request` 总入口。调用方还必须预留 GPU 页、目录/队列元数据及各后端实际 scratch/rollback/extent，限制资源预算与 token/追加次数，并统一失败清理。不能据此宣称正常模型请求已不依赖 GC，或整个进程的元数据已经有界。目录永久持有 creation writer 的问题已在下节修正；正常请求仍须把这些 producer 交接规则接入完整的请求 owner 和任务调度；已捕获任务的完成发布已在后文接入。

新增预约测试通过逐分配点故障注入检查全量回滚；已存在历史没有被回收，槽位与 payload 容量没有泄漏。另在禁止后续普通堆分配、暂停物理回收的条件下，消费 8 个预约槽位完成每 token 退休的短尾块写入；请求结束后在途 writer 仍持有 payload，最后由后台归还。覆盖 move、跨 owner、短 span 容量、关闭及对象寿命，并在两个申请线程和一个观察线程之间完成 400 次预约/取消/写入/回收。该长循环仅验证固定仓库资源，不替代完整生产目录至少 10×容量的验收。

本轮预约实现通过本地相关 CTest **13/13**、Linux 宿主 CTest **22/22**、宿主 ASan/UBSan **10/10**。独立 TSan 构建的 repository、GC、reservation 三项 **3/3** 通过：主机默认随机地址布局会在测试启动前触发 `unexpected memory mapping`，使用 `setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-tsan-reservation --output-on-failure -R 'historical_kv_(reservation|repository|gc)_test'` 仅调整测试进程即可正常运行，未改变系统设置。这些检查不覆盖 CUDA adapter 的完整并发执行。

CUDA CLI 和定向测试重新编译通过；实际 GGUF 布局上的完整块 5 代复用与短尾块 GPU 测试均通过。本轮没有再次运行默认模型生成对照，因为普通路径未使用新增 RAM 预约组件；这也不构成生产请求预约验收。原始日志位于同一远程隔离目录的 `historical-reservation-{host,asan,gpu}-test.log`、`historical-reservation-tsan-test.log`（初始运行时失败）和 `historical-reservation-tsan-noaslr.log`（成功运行）。

### Producer 随实际任务归还

绑定单块接口新增必填的空 `Producer` 输出，批量追加新增必填的空 `vector<Producer>` 输出；成功时才移走调用方 writer，输出已被占用或其它验证失败均不覆盖原任务。目录内部只保存弱引用与独立的 repository/storage owner。请求方必须在任务全部捕获原 handle 后归还自己的 producer，取消时也必须归还；后续写入使用任务捕获的上下文，不能通过新块号猜测原存储。`find()` 在生产任务结束后可以返回没有 writer 的身份描述，Mean-K / packed 采集已处理这种情况，不会解引用空指针或回退 legacy 写入。读取和搬运从独立 storage owner 获得真实 lease，因此已发布内容不会因 writer 退出而失去可读性。

`unbind_retired` 不再释放 writer，也不是后台回收的前置步骤。目录可能暂时仍记录 Bound，实际选块与读/搬运准入继续以仓库的精确代数状态为准。`confirm_reclaimed` 可直接接受仓库的同代释放确认；旧物理槽已变成新 generation 时也不会开放旧块。确认后才允许删除旧密集块，固定尾部范围的校验不再依赖 writer 是否还活着。

新增宿主并发用例：分别在 Writing 分量采集和 Live 读/搬运尚未结束时启动真实 GC worker；退休立即拒绝新请求，旧任务在独立线程完成，最后一个引用归还后 worker 自动释放全部 CPU payload。主线程没有解绑、没有 manual tick、没有下一轮 active，释放回调也不在主线程执行。随后复用原槽并验证旧代身份仍关闭。GPU 用例保留原有完整块 5 代复用和短尾块检查，并增加真实 CUDA Mean-K 迟到完成后，由 GC worker 自主释放的检查；测试输入仍是已知 fixture 张量。

本轮本地相关 CTest **13/13**、Linux 宿主 **22/22**、宿主 ASan/UBSan **10/10** 和独立 TSan 的 directory/append/legacy-bridge **3/3** 均通过。TSan 仍采用仅限测试子进程的 `setarch x86_64 -R`。CUDA CLI/定向测试重新编译通过，48 层 GPU 验收包含 `PASS managed task lifetime: late CUDA Mean-K completed, background CPU release, no unbind/next active`。原始日志位于远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU/` 的 `historical-producer-{host,asan,tsan,gpu}-test.log`。没有再次运行普通模型生成对照；本次目录仍由受控验收显式绑定，默认路径未启用托管 GC。

这一步消除了目录的永久 writer 依赖，尚不等于正常 CLI/server 已具备完整请求调度。完整生产 owner 仍需接入初始身份和请求资源、协调错误取消并持有实际 GPU 完成许可；已捕获 Mean-K/packed 任务的交接与完整发布见下节。RAM 预约仍须与元数据/GPU 预约统一。密集块表及墓碑长期有界和至少 10×容量的完整生产验收仍未完成。

### 执行前捕获 packed 任务与自动发布（2026-10-17）

`CaptureBatch` 的每段现在同时记录 producer、原始范围、token 偏移和 GPU 槽。prefill 的 `harvest_pending` 在图完成后、Mean-K 队列消费前，为这批已执行 token 提交 packed K/V；decode 在 `decode_mean_commit(n_keep)` 只提交接受的前缀。任务直接使用原捕获身份，即使准备后、实际提交前该代已经退休，仍可完成原来获准的范围。晚到的全块扫描/普通 stage-out 不再从目录发现托管写者，只处理 legacy 数据。新任务仍不能通过退休代获得权限。

托管 packed 采集按每个分量的已写前缀加上在途队列前缀计算缺口。重复提交不重复写入，跨微批次不会把已经登记但还未执行的后续行提前复制到 CPU；decode 也不把拒绝的 draft 行写入。K 入队可能提交旧 slab，因此处理 V 前重新读取进度。预留 job 向量容量先于登记 DMA 源，避免内存分配失败后已提交的源没有对应 writer。旧 slab 的 host commit 成功后才转移下一批 job 所有权，异常不会直接销毁下一批在途任务。这会改变 slab 的 CPU/GPU 重叠时序，后续性能对照必须包含该变化，不能只凭功能测试宣称吞吐不变。

Mean-K、decode Mean-K 和 packed 分量每次成功写入后调用完整性发布检查。只有全部必需层的全部分量覆盖固定 span 才成为 Live，未完成块继续 Writing，已退休块始终 Retired。发布不依赖下一轮请求或测试手动调用；请求的 producer 可以在全部微批次完成身份捕获后立即归还，实际任务完成后不留下目录写者引用。

为避免最后一次分量写入碰上 refcount CAS 冲突后永久漏掉发布，仓库内部 phase 编码预留一个 publication bit：持有 writer 时只能处于 Writing(2)、Live(3)、Retired(7)，对该位执行一次原子 `fetch_or` 只会使 Writing→Live，其余两种状态保持不变，也不改变高位引用计数。原 writer 排除了 Releasing/Ready/Reserved 及槽位复用；并发退休无论先后都不会被覆盖。配置仍要求 64 位原子 lock-free，退休与新许可继续使用有界 CAS。此编码仅在进程内使用，没有更改持久化格式。

新增并发测试覆盖 200 轮引用变化下的最后一次发布和 200 轮发布/退休竞争，并逐轮检查引用数、重复发布、退休保持关闭和同槽复用。GPU 验收增加前 16 行采集与在途去重、准备后退休再提交 packed、两个已准备微批次交接、无请求 producer 时自动发布及全部任务退出后的零 writer 引用。输入仍是已知测试张量；这证明生产采集入口的具体行为，不代表正常 CLI/server 已自动创建托管身份。

本轮最终源码通过本地相关 CTest **13/13**、Linux 宿主 **22/22**、宿主 ASan/UBSan **10/10**，以及独立 TSan 的 repository/GC/reservation/directory **4/4**。CUDA CLI 与定向测试构建通过；最终 GPU 测试继续通过完整块 5 代复用、上述微批次/自动发布检查和短尾块逐层字节验证。ASan/UBSan 与 TSan 的数字仅覆盖宿主组件，不覆盖 CUDA adapter。全部六个变更源码文件的本地/远程 SHA256 一致。

普通模型 retrieval 回归使用同一 GGUF、356-token 提示词、预算 512，完成 64-token query replay 和 8-token 生成，Mean-K 为 48 层且 `n_miss=0`。同版 recency 使用预算 96，完成同一 356-token 提示词和 8-token 生成，记录 9 次 `stage_out_cpu`。两者都是默认未启用托管 GC 的回归，不是 GC 的效果或质量对照。原始日志保存在远程隔离目录的 `historical-capture-completion-{host,asan,tsan}-test.log`、`historical-capture-completion-final-gpu-test.log` 与 `historical-capture-completion-model-{retrieval,recency}.log`，并复制到本地 `build-local/capture-completion-validation/`。

在正式绑定异步 writer 前，已修正两个现有错误路径：D2H 提交失败时保留位置队列供同步回退，并在回退读图张量前完成 scheduler 同步；CUDA stage-in 的直接 H2D fallback 等待 stream 拷贝完成才返回，避免调用者的局部 packed 缓冲区先析构。这些修复不构成 GC 生产接入。

实际启用还要拒绝没有注意力层或没有模型采集入口的配置。`Writing` 块尚未封存，不能作为当前 legacy Raw 桥接的删除候选；独立按句柄存储已验证 Writing 期间退休及在途 packed/Mean-K writer 完成，正常推理仍待接入。退休和确认回收后的目录墓碑都必须由 adapter 读取端识别为关闭状态，禁止按“找不到绑定”退回旧位置读取；宿主选块预过滤已经接线，但不等于全部读取和执行许可已完成。`seq_rm`、清空、截断、回放和快照要在修改 GPU/Raw 前完成统一退休与异步排空；退休的 GPU 块再次 stage-out 时也不能重写已释放的 Raw 位置。

### 本阶段验收记录

提交 `53930ff` 在本地 MinGW 环境通过 11 项 Historical KV 及相关宿主 CTest；另外两项会话快照/迁移测试在把临时目录指向工作区后通过。Windows 会话缓存生命周期测试仍受当前沙箱的符号链接权限及 Python 子进程输出编码限制。远程隔离 Linux 宿主构建通过全部 20 项 CTest，仓库代数、影子目录和 Raw RAM 回收的 ASan/UBSan 三项测试通过，CUDA CLI 与 server 均完成构建。

使用提供的 GGUF、单张 CUDA GPU、`--kvmem-method recency`、`--kvmem-budget 96`、`--kvmem-cpu-gb 1` 和 356-token 提示词生成 8 个 token 成功；`KVMEM_TRACE` 明确记录 9 次 `stage_out_cpu`。这是与上一轮 `97c1672` 相同的 recency 模式，只验证**原有 GPU 工作集溢出到 CPU 的路径**及新代码未破坏该路径；影子目录仍为空，后台 GC 未启动，也没有实测历史块的物理删除或 CPU 释放量。另一次使用 CLI 默认 retrieval 模式的运行在 query replay 阶段报告找不到 GPU slot；该配置与上一轮 recency 验收不同，尚未做基线对照，不能据此归因于本轮改动或声称检索模式已通过。

## 已实现的迁移基础

1. 以 Apache-2.0 来源文件为基础移植 `KvMemGc`（许可文本见 `kvmem/LICENSE.kvmem-qw3`）：后台增量扫描、按分数排队、观察期、shadow、块代数、不可变目录视图、工作集 lease、迟延释放回调及指标。`max_live_blocks == 0` 表示不开启；开启时由调用方创建 worker 并提供带分配代数校验的非阻塞释放回调。
2. `KvMemStore` 可在追加历史前挂接 GC；追加/截断/换层/IO 状态同步到 worker。top-k、prefill pressure、semantic group 与媒体约束在新目录中跳过已删除的块；显式要求的已删除块会失败，检查点可检查其必要前缀是否完整。
3. 单测覆盖策略优先级、all-fit 分支不复活“洞”、旧目录仍可读、延迟释放，以及晚到 lease 撤销退休。普通构建不创建 worker，不会改变既有选块结果。

### 有限请求的实际模型采集入口（2026-10-17）

新增 `HistoricalKvCaptureSession`，把此前各自验证的 RAM 预约、固定 span 登记和 producer 交接接到正常请求。CLI 在分词和 batch 限制确定后，以输入长度、有限的 `-n` 和最坏追加次数申请真实仓库槽位与 full-slot payload；容量不足在首次 decode 前拒绝。输入与输出额度分别扣减，不能把未用输出额度冒充输入额度；每次非空追加扣减一次边界额度。结束输入前必须用完输入额度，允许 EOS 提前结束输出。短块不借用上代尾部，身份与原位置保持不变。

`prepare_working_set` 消费预约并登记新身份；`prepare_ubatches` 为全部微批次捕获身份后归还请求持有的 producer。recency 没有 Mean-K 图节点，因此 packed 任务接过身份后也必须消费对应位置批次；否则多微批次会重复读取第一批并留下 writer。结束入口排空采集，检查每个已创建块均为 Live 且任务引用为零，才报告成功并归还未用私有 payload。这个归还量不是 GC 删除历史块的收益。

实际模型暴露并修正了短块的 GPU 容量判断问题：原逻辑只用 resident token 数判断压力，多次单 token 追加可能已经占满物理槽。托管目录存在时，追加还会核对所需新 GPU 槽与实际空槽，不足时走原有工作集换出规划。该检查服务于正常 GPU 工作集调度，不把 CPU GC 改成 GPU 回收策略。

此 owner 明确限定为单次采集基线。它预留了 RAM、释放回调和请求记录，并预留有限行位置表；目录节点、全部 GPU 页、staging scratch、后端工作区和 rollback 仍未统一预约。登记或执行失败后上下文必须退出，不能通过截断尚有任务的历史块来伪造回滚；未用的私有 payload 可立即取消，已安装 payload 保留到任务及所属域退出。CLI 的该模式在提前返回时也会析构上下文并排空采集。跨 owner 的异步任务保留原存储，后台完成后不存在回调引用环。这里尚未实现生产 GC 的后台取消队列及所有设备错误恢复。

入口拒绝 retrieval、MTP、hybrid、跨设备 KV、CPU spill arena、NVMe、非连续位置及多序列输入。续接、reset、truncate、replay、snapshot/restore、位置平移和切换会话也不能用于此基线。后续须补齐对应协议后逐项开放，不能将当前限制作为完整交付范围。

命令示例（`MODEL` 是本机 GGUF 路径；N 是历史物理槽位数，不是 GPU token 预算）：

```sh
build-cuda/bin/llama-kvmem-cli -m "$MODEL" -f prompt.txt -n 8 -c 1024 \
  -b 128 -ub 16 --kvmem --kvmem-budget 512 --kvmem-method recency \
  --kvmem-historical-capture 32 --tokens-only
```

宿主测试新增容量拒绝、输入/输出/边界额度、未交接 producer、未发布及仍有任务时禁止完成、EOS 归还私有容量、分配点故障注入、失败状态关闭和取消后异步 writer 完成。本轮相关本地测试 **14/14**，Linux 宿主 **23/23**，宿主 ASan/UBSan **11/11**；补充后台完成用例后，该项在三套构建分别重跑通过。独立 TSan 的新增 capture-session 用例 **1/1** 通过，沿用只对测试子进程设置的 `setarch x86_64 -R`。这些 sanitizer 结果不覆盖 CUDA adapter。

普通 CLI 已使用提供的 Qwen3MoE GGUF、Q8_0 KV 和 356-token 提示词生成 8 个 token，自动登记 20 个历史块，结束记录 `live=20 references=0 rows=364 payload_bytes=33423360`。没有手工绑定 writer 或调用发布接口。该运行删除关闭，不证明 CPU 历史自动回收，也不是质量或性能验收。

同模型、同提示词、同 `-n 8 -c 1024 -b 128 -ub 16`、recency 预算 512 的原路径对照，8 个 greedy 输出 token 与开启采集时完全相同（原始 token 文件 `cmp` 返回 0）。这仅是一个全历史可驻留样例的功能对照。生成 wall time 本次原路径为 487.98 ms、采集路径为 567.64 ms，采集增加了工作；单次短样例不能代替稳定吞吐、长上下文或 agent 质量评测，也不能当作 off/shadow/delete 策略对比。

GPU 验收在原有已知张量用例之外，新增真正调用 `llama_decode` 的两个场景（F16、Q8_0）。每个场景输入 160 token，使用 16-token 微批次并生成 3 token，自动建立 9 个 span；逐层比较新块 CPU payload 与实际 GPU K/V，分别检查 **432 个块×层** 的字节一致性。主动覆盖 GPU 物理槽压力换出，最后重新读取冷历史并比较保存的内容 hash；所有块均 Live、引用归零，超出已完成请求的下一次追加被拒绝。最初运行准确复现短块物理槽不足，修复后两种格式及原有 GC/异步 fixture 全部通过。该真实图测试仍未开启策略退休，不能把已有 fixture 的 CPU 释放量计入它。

2026-10-17 补充了真实图数据进入新 LRU 回收器的端到端验证：上述 F16、Q8_0 两个模型场景在请求完成并排空设备任务后，对同一按句柄仓库启动后台 LRU，分别退休并物理归还 2 个真实历史块。F16 释放 **6,291,456 bytes**，Q8_0 释放 **3,342,336 bytes**；Ready 增加，已回收代拒绝新 CPU 读取，已有 GPU 工作集字节保持不变。测试使用真实模型自动采集的 payload，没有手工写入或绑定。它验证了真实数据的回收闭环，但删除发生在请求完成后；普通 CLI/server 仍未启动生产 GC，也尚未证明执行中的 active 许可、长期复用和已接纳请求的完整资源保证。GPU 原始日志保存在本地 `build-local/capture-session-validation/historical-real-gc-postrequest-test.log`，以及远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-real-gc-postrequest-test.log`。

### 微批次执行许可与请求中退休（2026-10-17）

托管 recency 的正常 `llama_decode` 在每个微批次应用 KV cell 后、构图前为仍驻留的历史块取得新的 `ExecLease`。已退休块不能借旧 GPU 槽绕过许可：适配器从后续 attention cell 视图移除该块并归还槽位；成功提交模型计算才更新这些获准块的 LRU `last_use`，许可持有到图同步与采集回调完成。无托管采集时该 hook 保持原路径。执行入口补丁为 `patches/0007-historical-active-hook.patch`。

模型实测暴露了一个生产者时序：刚完成 GPU 计算的块可能仍在异步 packed K/V 的 Host 拷贝中，仓库状态仍是 `Writing`。首版受限采集模式在进入下一微批次前结算这批 packed 拷贝，确保该块已发布 `Live` 后才授予新的执行许可。这增加 Host 采集等待，尚未做稳定性能对照；后续可设计区分 GPU 已可执行与 CPU payload 已完整发布的许可，以恢复重叠，而不能直接把 `Writing` 当作 `Live` 放行。

远程真实模型测试保留了原 F16、Q8_0 的请求后 LRU 回收检查，并新增 F16 请求中退休：第一段 48 token 之后退休一个当时 GPU 驻留的块，旧 CPU `ReadLease` 持有期间仓库拒绝物理释放，新 CPU 读立即关闭；继续第二段真实 `llama_decode` 时，新 active 跳过该块并从 GPU 工作集去掉它；旧读者退出后后台 worker 归还对应 Host payload，余下 160-token 输入和 3-token 生成正常完成。请求结束校验现在允许确实已退休、释放中或同代已回收的块，不再错误要求每个创建块仍为 `Live`。本地 capture-session 测试及远程五项相关宿主 CTest 均通过；隔离远程日志为 `historical-active-midrequest-gpu-test.log`。此用例的退休发生在两次 `llama_decode` 之间，**尚未验证退休与已提交的 GPU kernel 真正重叠**。

### 受限 CLI 的 off/shadow/delete（2026-10-17）

CLI 的 `--kvmem-historical-capture N` 现在可配 `--kvmem-historical-gc-mode off|shadow|delete` 和成对的 `--kvmem-historical-gc-low/--kvmem-historical-gc-high`。默认 off；显式 shadow/delete 在请求仓库上启动唯一后台 manager，退出请求时先停止 worker，再检查发布与任务引用。阈值要求 `0 < low < high <= N`；未给时取容量的约四分之一与二分之一。首版只使用独立的按句柄 RAM 存储，不能用于 server、检索、MTP、NVMe、CPU spill arena、hybrid、多请求或会话恢复。

在同一 GGUF、356-token 提示词、8-token 生成、容量 32、低/高水位 14/16 下，delete 模式由普通 CLI 请求的 worker 提交并回收 **6 个**历史块：初始预留 payload 36,765,696 bytes，结束时剩余 23,396,352 bytes，`policy_failures=worker_failures=0`，请求正常完成。off 模式结束时剩余 33,423,360 bytes。差值为真实按句柄 CPU K/V payload，不是 RSS 降幅；请求完成时的未用私有额度也会归还，不能全部算作 GC 收益。原始日志：远程隔离目录的 `historical-cli-gc-delete.log` 与 `historical-cli-gc-off.log`。

shadow 不退休块、不释放 payload；相同配置结束为 `committed=0, reclaimed=0, live=20`，生成的 8 个 token 与 off 的文件逐字节相同。首轮实测重复提出同代候选 6508 次；修正后使用每槽代数模拟已提交的待回收额度，同一请求只提出 **6 个**不同候选，`shadow=6`，输出仍与 off 完全相同。对应日志是 `historical-cli-gc-shadow2.log`；宿主 GC 去重测试在 Windows 与远程 Linux 均通过。这是短请求功能对照，尚未完成稳定延迟/吞吐与 Agent 质量验收。

原始日志保存在远程隔离副本 `/tmp/kvmem-gc-99bef27.3CD9MU/` 的 `historical-capture-session-*` 文件，并复制到本地 `build-local/capture-session-validation/`；最终逐层模型检查使用 `historical-capture-session-quant-gpu-test.log`。首次失败日志也保留。9 个变更源码/构建文件的本地与远程 SHA256 一致，远程原工作仓库未改动。

## 剩余生产接入与完整验收

以下是导师方案仍未满足的交付项，不是当前已支持功能：

1. **生产身份创建与请求 owner。** packed K/V、Mean-K D2H capture、同步 fallback 和 decode accumulator 已使用执行前捕获的同代 writer/范围，最后一个必需分量写完后自动发布；短尾块、非对齐范围和批量身份登记已有接线。CLI 的单请求 recency 路径现能自动预约 RAM、创建身份、向任务交接 producer，并显式运行 shadow/delete；仍需接入 server、多请求生命周期及错误取消。这个受限入口没有承诺整请求的全部资源，更未覆盖所有调用路径。
2. **所有 active 与资源依赖。** 给 reselect、prefill、same-query、all-fit、commit-resident、keep-selected、attach 等入口建立新的 ActiveId/ExecLease；保护已经提交的 GPU 输入到实际完成。真实执行才更新 LRU；评分、预取与单纯选中不计实际使用。进一步拆分 Host/GPU/metadata 依赖，GPU 只用独立副本时允许 Host 提前释放，并协调 pinned cache 的槽位失效。
3. **请求预约与长期有界。** RAM 槽位、payload、scratch 预约及碎片上界已有资源组件；还需在请求进入 active 前统一取得 GPU、元数据、各后端和 rollback 等全部资源，并落实额度约束及失败处理。不能等 GC 补给；RAM API 已拒绝 streaming，生产参数入口也须相同。将生产 dense block 表及版本、评分事件、墓碑、pending decisions 一并改为有界；累计追加至少 10×仓库容量验证完整链路复用与计数守恒。
4. **会话、检查点与支持组合。** reset/truncate/关闭统一结算旧代任务，保证 worker 和存储的销毁顺序；闲置 checkpoint 失效不能复活历史，已承诺 rollback 保留实际依赖。NVMe allocation/IO、MTP、hybrid、query replay、refresh、多 lane 等按实际使用组合逐项接入与验收，未验证组合明确列出。
5. **策略开关、观测与效果。** 将受限 CLI 已有的 off/shadow/delete、LRU 压力与 pending credit 扩展到完整生产入口，提供独立新策略模板及版本前置条件；策略决定不受 manager 隐含年龄/驻留/引用门槛限制。保存 retirement/release/ActiveId/延期原因 trace、allocator 与 metadata 长轨迹、入口拒绝和前台 GC 等待计数；补独立 TSan 结果。固定模型、seed、任务和容量，对 off/shadow/delete 及 LRU/新增策略做延迟、吞吐和真实 agent 质量对照，保留原始日志。质量阈值未约定时只提交比较结果，不宣称通过。

单 GPU、单 lane、文本、RAM、MTP off 是首个生产闭环的阶段验收范围，不是完整交付的替代定义。目前 repository 与按句柄 payload store 已使用固定物理槽位，但生产 `KvMemStore::blocks_`、位置映射及相关队列仍需改造；当前结果不证明长期总内存有界，也不证明已接纳请求完全不依赖 GC。

# Historical KV 后台 GC 迁移方案

> 2026-10-21 范围调整：本分支按导师当前 `kvmem-vllm` 的基本后台管理交付，
> 当前范围、实现差异及验收见 [CPU 历史 KV 后台管理](background-history-gc.md)。
> 下文保留此前设计演进和测试记录；其中扩展生产化待办不再全部作为本轮交付门槛。

## 结论与边界

`kvmem-qw3` 的 legacy GC（`include/qw3/kvmem_gc.hpp`、`src/kvmem_gc.cpp`、`docs/kvmem_gc_v1.md`）会在后台选择**单个 historical block**，先在新目录视图中将其逻辑删除，等旧读者与 lease 退出后再释放该块的原始 K/V 权威存储。它是永久缩减可检索历史，并非把 GPU 块挪到 CPU 的缓存淘汰。它的策略也不是纯 LRU：最近使用、指数衰减频率、检索分数、重复/覆盖证据、anchor 和保护状态共同参与决策。

本项目原有的 `PinnedKvTier` LRU 管理 CPU 缓存，session cache 管理整段会话；两者都不会删除任意 historical block。早期移植的 Qwen3 `KvMemGc` 未接入服务器生产删除路径。新的 repository/manager 已有**受限 CLI 与 server 显式入口**，可在 recency/RAM 配置下运行 off、shadow 或 delete；默认行为仍是 off。server 已接入单 lane 文本请求的有限预约、精确前缀延续、新会话和健康取消；跨前缀回退、检查点及其它配置的接线仍未完成，不能把这个入口称为整套生产 GC。

2026-10-17 新增 CLI 的显式实验入口 `--kvmem-historical-capture N`：普通模型请求可以自动预约 RAM、创建按句柄历史块并完成 packed K/V 采集。该参数单独使用时删除关闭；配合 `--kvmem-historical-gc-mode shadow|delete` 才启动新管理器。当前仅支持单请求、单 GPU、单序列文本、recency、flash attention、RAM，不等于整请求资源准入。默认不加该参数时仍走原路径。

后续实施以导师的 `historical-kv-gc-implementation-lru-policy-template.md`（2026-10-15）为目标：策略可以退休 Writing、未封口、active 和在途块；机制只延后物理释放，不能以年龄、驻留或引用数否决退休。旧 `KvMemGc` 的保护/rescue 行为仅留作原始移植与回归参考，完整交付使用新的 repository/manager 协议。完整交付还要求长期元数据有界、真实请求预约和实际模型/agent 对照证据，不能用受限完整块测试替代这些要求。

## 当前执行计划（2026-10-19 调整）

用户已确认采用：**冻结首批支持配置 → 做长时间、内存上界和真实负载验收 → 修复验收暴露的问题 → 再扩展策略与配置**。本节是当前执行顺序；下文按日期记录的成果保留为历史证据，文末完整待办不再表示所有项目立即并行开发。

保留现有策略接口、repository/manager、代际身份、资源许可和独立释放协议。策略决定退休哪些历史块，退休立即关闭后续选用，机制等待已获准的使用结束再实际归还相应存储。此次调整不改变 GPU 工作集策略，也不重新设计一套 GC 框架。

### 第一步：冻结首批支持配置

首批验收目标配置固定如下；“纳入验收”不表示已经证明满足生产要求：

- **环境与模型：** 当前远程 Linux/WSL、单张 CUDA GPU，使用已经验证采集入口的 Qwen3-30B-A3B GGUF。模型文件为 `/mnt/d/ollama/blobs/sha256-e9183b5c18a0cf736578c1e3d1cbd4b7e98e3ad3be6176b68c20f156d54a07ac`；每轮保存实际模型指纹、源码提交、构建选项及设备信息。
- **请求入口：** server 单 lane、串行文本会话，CLI 做必要回归。每个请求有明确的输入和输出上限；SSE 仅改变传输方式，仍预约整个有限请求。纳入精确前缀延续、显式 reset、新会话替换、prefill/生成取消及已有的明确重算路径；并发会话、通用 checkpoint/rollback 不在首批支持范围。
- **推理与存储：** recency、flash attention、现有 f16 KV 布局；历史权威存储使用 RAM。关闭 MTP、独立 historical pinned 副本、legacy CPU spill、NVMe 和磁盘 session cache。模型结构仍是上述已验证模型，不将其 MoE 结构与待扩展的混合注意力/状态模型支持混淆。
- **GC：** 首批只验收现有 LRU，固定其参数；使用 off/shadow/delete 做对照。默认开关不因本次计划调整而改变。未经验证的组合须明确拒绝或保持关闭，不能静默落入缺少保护的路径。
- **容量与负载：** 分为小容量压力和实际工作负载两套固定参数。现有 HTTP fixture 的 32 个历史槽、32-token 块和 4-token 输出只用于快速生命周期回归；不能直接作为真实编码负载配置。实际负载先用基线试跑确定上下文、输出上限、历史容量和工作集，再保存完整命令与输入并冻结。更改这些参数后必须重新开始对应对照，不能混合统计。

已有 pinned/NVMe 代码和回归证据保留；暂不追加其公开组合、优化或新功能。其它 active 入口仅在首批配置实际可达、或可能绕过关闭条件时纳入当前审计，不先把所有可选路径全部接完再验收。

### 第二步：长时间、内存上界和真实负载验收

先检查首批可达路径和观测数据，补齐运行验收必需的最小接线、计数或保护。发现缺陷即进入第三步修复，然后回到本步复验；不要求把所有失败测试跑完才开始修复。

1. **持续运行与复用。** 首轮工程验收同时满足持续运行至少 1 小时、同一 historical domain 累计创建至少 10 倍物理槽位容量的历史块。该时长是本轮拟执行的下限，不是已有测试结果或长期可靠性承诺。分别记录同域持续复用与反复 reset/cancel；后者不能替代前者。旧域未结清时创建新域的重叠也必须覆盖。保留请求、块、回收、拒绝、取消和错误的完整计数，不能靠不断重启进程获得“内存稳定”。
2. **内存与元数据上界。** 在负载开始前给出容量计算和准入规则，覆盖历史 payload、块表/目录/版本/墓碑、评分事件、待处理决策与任务队列、搬运临时空间、后端对象及 GPU 资源。即使关闭可选 pinned 副本，实际 D2H/H2D 暂存仍需计入。新旧 domain 的总占用统一核算，不能只看单域额度。核对分配、预约、实际释放和可用额度守恒，并同时采集进程 RSS、设备占用和高水位。allocator 保留内存与仍被使用的内存分别解释；一次 RSS 下降或短时间曲线平稳不构成上界证明。
3. **准入与回收独立性。** 容量不足在接纳前明确拒绝，短尾碎片和失败回退计入预约。受控暂停或延迟 GC worker 后，已经接纳的有限请求仍能在已承诺资源内完成；新请求可以因容量不足拒绝。验证退休后下一次选用被关闭，旧执行和旧搬运能安全结清；不能等待 GC 补给、重复返还额度，或把重释放转移到前台。暂停 worker 的故障注入与正常吞吐测试分开报告。
4. **真实请求与性能。** 使用模型原生模板和可重复的多轮编码/工具调用负载，覆盖延续、上下文增长、取消与会话切换。固定模型、seed、输入、容量和生成上限，多轮交错运行 off/shadow/delete；记录首 token 延迟、token 间延迟、吞吐、CPU/GPU/RAM 峰值，以及成功、拒绝、取消和失败数。相同容量的 off/shadow 可能耗尽，必须报告这个结果；额外的大容量质量参考单独标注。既有强制短输出 fixture 继续作回归，不能用于证明任务质量。
5. **真实 agent 与质量（2026-10-20 按用户要求暂缓）。** 用户尚无老师所指 DeepSWE 的链接或配置，明确要求这一项先不做，待其补充资料后再继续。当前不安装、下载或配置 DeepSWE，也不把同名模型/agent 与评测基准自行选定为验收对象。继续 GC 工程验收和当前 Qwen3 模型的请求回放；它们不能替代真正的 agent 执行或宣称 DeepSWE 验收通过。后续实际 agent 若需要不同模型或配置，再作为有明确理由的验收依赖重新冻结并复验。

每轮产出须包含版本/配置指纹、命令、输入、原始日志、资源时间序列、结果和未通过项。延迟、吞吐与质量阈值尚未约定时报告实测差异和限制，不把建议值当成已批准的通过线。当前已有模型/HTTP 回归、sanitizer 和有限连续请求证据；上述完整长测、总量上界及实际 agent 质量验收仍未完成。

**预约范围澄清（2026-10-20，已列为导师确认项）：** 导师文档 §5.5 要求真实 payload/extent、必要的搬运和 scratch 等资源在接纳前落实，已接纳请求不能等待 GC 补给；文档没有直接要求整个推理框架的所有临时 C++ 分配都清零。此前日志里的“完整接纳后零分配”比这条原文更强，不能单凭调用次数非零认定 payload 预约失败，也不能因为 RSS 暂时平稳认定总量已受限。当前保留所有原始分配证据，优先验证 GC 所管理资源的实际预约、跨域总量和失败路径；通用调度器/图优化容器的全面预分配暂不扩展，等待确认其是否属于首批硬门槛。该澄清不豁免首批可达临时空间的上界核算，不把未完成的资源验收标记通过。

### 第三步：仅修复首批验收暴露的问题

当前开发队列只接收首批配置的安全性、正确性、容量约束、回收等待或实际负载阻塞问题。此前偶发 CUDA 非法访问仍列为未关闭问题：保存首次失败证据，核对版本与触发路径，进行针对性复现和定位；后续未复现不能直接将其注销，也不能直接认定由 GC 导致。

每项问题记录复现条件、影响范围、修复及必要回归。优先复跑失败场景，再重跑受影响的长测或对照；不因文档或测试外围调整反复运行全部昂贵测试。若首批实际可达入口缺少身份/执行许可、统一预算或有界元数据，则补齐它们；不以“冻结范围”为理由跳过已承诺路径的缺口。

首批阶段通过需要：配置和命令可复现，长期资源账目及上界成立，正常与取消/失败路径无未解决的安全问题，已接纳请求不依赖 GC 补给，真实负载结果及性能/质量差异有证据。只完成工程部分时标记“工程验收完成、agent/质量待验收”，不能标记整套生产 GC 完成；在首批门槛未满足前不启动可选扩展。

### 第四步：通过首批门槛后再扩展

依据实际使用需要，每次只增加一个策略或配置维度，独立冻结和验收。后续仍保留导师方案要求的冗余策略实验，以及所需的检索/重选入口、checkpoint/rollback、pinned/NVMe、MTP、混合模型、多 lane 等工作；不默认把所有可能组合都实现一遍。增加策略仍只修改策略及特征接口，不改回收生命周期协议。

**当前位置（2026-10-20）：** 首批范围已冻结。此前一小时同域长测通过：11,448 次请求、45,792 个块经过 8 个槽位；这是当时版本的证据。`18eaddd` 修复有限请求内跨微批次的 GPU 权限，43 个模型场景及 198 次 reset/cancel 短测通过；**最新六轮编码压力仍未通过：off 两轮截断、delete 五轮截断，只有第一轮内容相同。** 容量不足自动重选是否开启新 active，已列为请导师确认的行为边界。本轮先完成独立并发排障：确认旧 GCC 11 检测运行库的条件变量误报，并修复 RawKvStore 关闭通知可能丢失的问题；修复后 GCC 12 TSan 与 ASan/UBSan 各 30/30，受影响用例重复 20 次通过。一小时 reset/cancel 长测现已完成，终态见下文；完整资源预约/上界、编码质量和独立 CUDA 非法访问仍未关闭，DeepSWE 按用户要求暂缓。首批阶段验收不替代“完成整套生产 GC”的最终目标。

**最新分配审计：** 96 次固定请求的调用栈与计数器完全对应；复用固定物理层目录后，前台普通 C++ 分配从 249,921 次降至 228,417 次，其余来源计数不变。候选首次完整模型回归出现 CUDA 非法指令，基线及候选后续完整对照各退出 0；首败原因仍未确定，不能据重跑通过注销。详见文末“接纳后分配来源审计与层目录复用”，生产验收仍未完成。

**最新 CUDA 定向复现与长测：** 在 `4cb3f16` 产品代码上完成 124 次定向场景执行，以及同一模型/context 的一小时 reset/cancel 长测；长测完成 13,303 次请求、13,302 个旧域全部结清，进程退出 0，原始日志复核通过。它们仍未定位原始 CUDA 错误。用户已明确“暂时不改主机设置”，因此暂缓需要 Windows 调试接口的检查器，不执行管理员辅助脚本。长测的版本、资源采样和范围限制见文末验收记录，不能据此宣称整套生产 GC 完成。

**最新宿主检测诊断：** 在独立 ASan/UBSan 构建上，使用进程级 CUDA 兼容参数完成原失败前缀的 8 个场景、34 次解码，未报告 Host 访问或未定义行为错误。进程仍因两个共 300 bytes 的 `libcuda.so.1` 泄漏报告退出 1；无模型、无 KVMem 的 CUDA 小对照也报告完全相同的大小和库内分配位置。因此不能写成 sanitizer 全通过，也未关闭原始设备错误。没有修改 Windows/驱动设置或抑制泄漏检查，详见文末新增记录。

**最新跨域容量验收：** 新增真实模型 reset 用例，在实际 payload 全部释放后暂停 GC worker，使旧 runtime/目录/manager 继续存在。30 个旧域加当前域占用 62/64 个清理 ticket 时，下一次 reset 在接纳前拒绝；payload 为 0、其拒绝计数未增加，当前域未被替换。恢复 worker 后旧域全部销毁、下一请求完成。该用例验证了清理容量独立限制零 payload 的旧域积累；产品库保持不变，整进程资源上界仍待完成。

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

模型实测暴露了一个生产者时序：刚完成 GPU 计算的块可能仍在异步 packed K/V 的 Host 拷贝中，仓库状态仍是 `Writing`。现在增加专用 `acquire_gpu_ready_writing` 执行许可：适配器在上一微批次图同步完成、历史 span 已驻留且不与本轮写入范围重叠时使用；它允许旧 GPU 数据参与本次执行，不授予 CPU 读取或搬运权。普通读/搬运仍要求 `Live`，退休则关闭全部新许可。这样去掉了每个微批次都等待 packed Host 发布的额外结算；图完成同步本身仍存在，尚未测量稳定性能收益。

退休块退出后续 GPU attention 视图时，其旧槽位可能仍被已获准的 packed D2H 任务读取。`b69332f` 先排空全部 packed 任务再归还槽位，保证源数据安全，但增加了退休触发的等待。后续实现改为按物理 GPU 槽位登记实际排队的 D2H 读取数：下一 active 立即移除退休块的 attention cells；若该槽位仍被读取，只标记延期归还，不等待或提交额外搬运。原有采集流程完成最后一个对应任务后，才把该槽位归还一次。其它槽位的待完成任务不会阻止这个槽位归还。两个状态数组在请求开始前按 GPU 槽位数分配，reset/free 路径检查未完成引用，请求结束检查全部清零。回收策略和后台 worker 始终负责 CPU 历史资源，不直接操作 GPU 槽位。

专用 Writing 执行许可在 `b69332f` 通过 Windows/Linux 的 repository、ActiveSet 测试与远程真实模型 F16/Q8_0 回归。首版提交后退休用例确认进入“全局存在未结算 D2H 任务”的排空分支，**并未证明这些任务读取的是退休块自身**；日志为 `historical-writing-exec-drain-gpu-test.log`。普通 CLI delete 日志 `historical-writing-exec-final-cli-delete.log` 显示 6 次 `retired=1 pending=96`，随后回收 6 个 CPU 历史块，policy/worker failure 均为 0。shadow 日志 `historical-writing-exec-shadow.log` 为 `proposed=6, committed=0, shadow=6, reclaimed=0`，8 个输出 token 与先前 off 文件逐字节相同。上述记录保留为上一版功能证据，不代表长期内存、延迟或 agent 质量验收。

新增精确源槽位场景在 Writing 块最后一段 GPU 计算提交后、其 packed D2H 采集前退休它；下一微批次检查该块已退出 attention、对应槽位仍有自己的 D2H 读取数且不在空闲表，普通采集完成后再检查只归还一次。该测试针对真实源槽位生命周期；原有已发布历史块的提交后退休场景同时验证不会因其它块的 D2H 而延期归还。此设计去掉的是 active 退休入口的额外 D2H 同步，现有采集同步、请求结束资源清理、域退出和全资源预约等仍需分别审计，不能据此宣称整套前台无 GC 等待已通过。

最终 CUDA 构建和全部八个顶层模型场景通过。`historical-deferred-source-gpu-test.log` 记录新场景的 `historical_gpu_defer slot=2 readers=96` 与一次 `historical_gpu_return slot=2`；新断言在下一微批次提交时直接检查同一槽位仍被保留。F16/Q8_0 各 432 个块×层的真实采集字节校验、各回收 2 块，以及两种旧退休场景也继续通过。这证明待结算 D2H 的所有权跨过了退休边界，不声称退休瞬间 DMA 或 kernel 一定仍在硬件上执行。本地和远程三个变更源码文件的 SHA256 完全一致，原始日志保存在远程隔离目录及本地 `build-local/capture-session-validation/`。

同版 CLI 在固定模型、Q8_0 KV、356-token 提示词、8-token 输出、容量 32 和低/高水位 14/16 下顺序运行 off、shadow、delete。off/shadow 均保留 20 个 Live 块、33,423,360 bytes payload；8 个输出 token 文件 `cmp` 返回 0。delete 的 `proposed=committed=reclaimed=6`，剩余 14 个 Live 块、23,396,352 bytes，`gpu_deferred=gpu_returned=6`，任务引用为 0，policy/worker failure 为 0。六次自动策略退休均记录同槽位的 `readers=96` 延期与随后归还，期间没有调用已移除的 active 退休排空入口。对应原始文件为 `historical-deferred-source-cli-{off,shadow,delete}.{log,tokens}`。这仍是短请求功能验证，不是延迟、长期容量或质量验收。

远程真实模型测试保留了原 F16、Q8_0 的请求后 LRU 回收检查，并新增 F16 请求中退休：第一段 48 token 之后退休一个当时 GPU 驻留的块，旧 CPU `ReadLease` 持有期间仓库拒绝物理释放，新 CPU 读立即关闭；继续第二段真实 `llama_decode` 时，新 active 跳过该块并从 GPU 工作集去掉它；旧读者退出后后台 worker 归还对应 Host payload，余下 160-token 输入和 3-token 生成正常完成。请求结束校验现在允许确实已退休、释放中或同代已回收的块，不再错误要求每个创建块仍为 `Live`。本地 capture-session 测试及远程五项相关宿主 CTest 均通过；隔离远程日志为 `historical-active-midrequest-gpu-test.log`。此用例的退休发生在两次 `llama_decode` 之间，**尚未验证退休与已提交的 GPU kernel 真正重叠**。

随后增加一个默认空的测试回调，位于真实图计算成功提交、LRU 使用序列已记录之后，但在 `harvest_pending` 的显式设备同步之前。新增 F16 模型场景从当次实际取得的 `ExecLease` 中选一个 GPU 驻留历史块，在这个确定的边界退休它；仓库仍报告该执行许可有效，回收尝试只能延期。紧接着的微批次跳过退休块，旧许可退出后后台 worker 归还 CPU payload，完整请求继续生成。`historical-submitted-retire-gpu-test.log` 同时保留了两种请求中退休场景及 F16/Q8_0 旧回归。该回调为空时不影响普通请求；测试证明了**提交至显式同步期间的 lease 生命周期**，但没有量到 CUDA kernel 在退休瞬间是否仍实际运行。

### 受限 CLI 的 off/shadow/delete（2026-10-17）

CLI 的 `--kvmem-historical-capture N` 现在可配 `--kvmem-historical-gc-mode off|shadow|delete` 和成对的 `--kvmem-historical-gc-low/--kvmem-historical-gc-high`。默认 off；显式 shadow/delete 在请求仓库上启动唯一后台 manager。请求结束检查现在不再停止或等待 worker，具体边界见下节。阈值要求 `0 < low < high <= N`；未给时取容量的约四分之一与二分之一。首版只使用独立的按句柄 RAM 存储，不能用于 server、检索、MTP、NVMe、CPU spill arena、hybrid、多请求或会话恢复。

在同一 GGUF、356-token 提示词、8-token 生成、容量 32、低/高水位 14/16 下，delete 模式由普通 CLI 请求的 worker 提交并回收 **6 个**历史块：初始预留 payload 36,765,696 bytes，结束时剩余 23,396,352 bytes，`policy_failures=worker_failures=0`，请求正常完成。off 模式结束时剩余 33,423,360 bytes。差值为真实按句柄 CPU K/V payload，不是 RSS 降幅；请求完成时的未用私有额度也会归还，不能全部算作 GC 收益。原始日志：远程隔离目录的 `historical-cli-gc-delete.log` 与 `historical-cli-gc-off.log`。

shadow 不退休块、不释放 payload；相同配置结束为 `committed=0, reclaimed=0, live=20`，生成的 8 个 token 与 off 的文件逐字节相同。首轮实测重复提出同代候选 6508 次；修正后使用每槽代数模拟已提交的待回收额度，同一请求只提出 **6 个**不同候选，`shadow=6`，输出仍与 off 完全相同。对应日志是 `historical-cli-gc-shadow2.log`；宿主 GC 去重测试在 Windows 与远程 Linux 均通过。这是短请求功能对照，尚未完成稳定延迟/吞吐与 Agent 质量验收。

原始日志保存在远程隔离副本 `/tmp/kvmem-gc-99bef27.3CD9MU/` 的 `historical-capture-session-*` 文件，并复制到本地 `build-local/capture-session-validation/`；最终逐层模型检查使用 `historical-capture-session-quant-gpu-test.log`。首次失败日志也保留。9 个变更源码/构建文件的本地与远程 SHA256 一致，远程原工作仓库未改动。

### 请求完成检查与后台回收分离（2026-10-17）

先前 `finish_historical_capture()` 为避免 GC 的短期元数据 guard 干扰 `references==0`，会先调用 `HistoricalKvGc::stop()` 并 join worker。现在 repository 单独计数对外授予的 Writer/Read/Transfer/Execution 许可，内部的观察/退休保护不计入这个操作数；全部控制引用仍留在原有 `control` 中，后台物理释放条件没有放宽。移动、拒绝申请与同槽代数复用均覆盖到计数测试。

新 `completion_state(handle)` 只读取原子 phase/generation，判断这个已创建 writer 是否仍未发布、已 Live 或已退休；Releasing 及已回收代也属于已退休。该查询不取得数据许可、不持有 worker 的锁，也不能据此释放内存。请求方停止建立新任务并排空设备/搬运后，`finish()` 检查操作许可为零和每个已创建块的完成状态，直接返回；后台可以继续观察或释放。`live_observed`、回收计数和 payload bytes 是各自时刻的观测，不构成冻结的最终快照。`task_references=0` 不表示后台元数据 guard 数为零。

宿主新增屏障测试把本请求一个实际 RAM 块的释放回调停在 Releasing：请求完成检查在屏障仍关闭时返回，打开后由同一个后台线程归还 payload。真实模型新增暂停 worker 的场景：前 48 个输入 token 完成后把 worker 停在受控策略屏障，余下输入与全部 3-token 输出额度继续执行；请求 finish 返回后才打开屏障。5 秒 watchdog 只负责将回归转换成测试失败，成功条件是明确的屏障先后关系，而非靠延时碰撞。

本轮 Windows 相关宿主测试 11/11、Linux 全部宿主测试 23/23、repository/capture-session 的 ASan/UBSan 2/2 及独立 TSan 2/2 均通过。最终 CUDA 构建及九个顶层真实模型场景通过，包含前述暂停 worker 用例与所有旧回归；sanitizer 数字不包含 CUDA adapter。八个变更源码文件本地/远程 SHA256 一致。远程隔离目录中对应记录为 `historical-request-finish-{host,asan,tsan}-test.log` 和 `historical-request-finish-gpu-test.log`，本地保存在 `build-local/capture-session-validation/`。

同版普通 CLI delete 继续使用既定模型、356-token 提示词、8-token 输出和容量/水位 32/14/16，正常完成；本次结束观测为 `committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 为 0。日志为 `historical-request-finish-cli-delete.log`。这些数值是本次实际观测，调用方不应要求每次请求结束时所有 Retired 都已经变成 Ready。

这只证明当前受限请求的计算与完成检查不依赖 worker 继续推进。未使用的私有 RAM 预约仍在 `finish()` 中销毁，上下文销毁仍需停止 worker；CLI 进程退出不等于上述请求完成边界。还需把私有预约清理和域退出交给后台、有界管理结束中的 owner，并补全 GPU/metadata/backend/rollback 预约及取消恢复，才可宣称完整生产请求没有 GC 等待与前台释放。实际 GPU 提交/完成失败的取消协议也尚待验收。

### 未用预约与 CPU 域退出的有界后台清理（2026-10-17）

上一节列出的两个前台清理点现已迁移：`HistoricalKvCaptureSession::finish/fail` 将未用 RAM 预约移交后台；受限采集上下文退出时不再 join GC，而是发出停止请求，在原 GPU owner 完成既有图、采集和设备资源清理后，将 CPU owner 整体交给后台。GPU 完成等待仍由正常设备生命周期负责，不能把这次改动解释成“不等待任何 GPU 工作”。

新增 `HistoricalKvCleanup`，采用固定槽位和一个 CPU worker。默认进程服务有 64 个清理槽；当前适配器在请求进入 active 前预约两个槽，分别用于未用预约和上下文退出，并提前分配对应任务对象。入队只移动所有权并发布原子状态，不申请内存、不获取 worker 锁，也不在调用线程销毁任务。槽位满时在入口拒绝新请求，没有同步释放或无限扩张队列的兜底。实际销毁结束后才归还清理槽；未用 payload/scratch 也先销毁，再归还仓库 Reserved 槽。这是清理任务数的有界保证，尚不是跨请求总 RAM 预算或全部生产元数据的有界证明。

`HistoricalKvGc` 分开提供停止请求和 worker 退出观察。清理 worker 对尚未退出的 GC 域只检查就绪状态，继续处理其他就绪任务；不在一个暂停的策略上 join 并阻塞整条清理队列。确认 GC 已退出后，才销毁 manager、会话、Raw store、历史目录和相关 CPU 记录。当前 RAM-only runtime 的析构不会调用适配器的 GPU backend；该限制不能直接推广到 NVMe、混合模型或未排空的异步设备任务。清理服务自身在进程/测试服务退出时仍会排空并 join，因此“请求结束、上下文销毁不等 GC”和“进程关闭绝不等待”是不同的承诺。

未用预约在 `finish()` 返回时可能尚未释放，所以结束日志中的 `payload_bytes` 是即时观测，不能当作最终历史存量，也不能把私有预约归还量计入 LRU 回收收益。模型测试只有在测量后续 LRU 释放量时，才显式等待私有预约完成归还；生产请求完成入口没有添加这项等待。

新增验收证据：

- 清理单测验证预约饱和与取消、禁止分配时发布、发布线程零 `delete`、未就绪任务不阻止其他任务，以及 4 个生产线程在 4 个槽位上完成 320 次唯一移交。这个循环只验证清理组件，不替代 P4 的完整生产长轨迹。
- 请求单测用显式屏障暂停清理 worker，再调用真实 `finish()`；其返回时没有新分配或 `delete`，未用 payload 仍占用真实容量，且第二个请求因清理槽已满而在入口失败。打开屏障后，未用 payload/scratch 和 Reserved 槽才归还，已安装历史仍可读取。既有分配故障、取消后旧 writer 完成等用例也通过。
- 实际 CUDA 模型新增第 10 个顶层场景：完成模型请求后，将策略 worker 停在屏障内，调用真实 `llama_free` 销毁上下文。只有该调用返回才打开屏障；期间弱引用确认 CPU 仓库和存储仍被持有，恢复后两者均过期，验证既不提前释放也不泄漏。5 秒 watchdog 仅用于让“错误地等待 GC”的回归失败，正常通过不依靠延时碰撞。

本轮 Windows 相关测试 **12/12**、Linux 全部宿主测试 **24/24**、cleanup/capture-session/GC 的 ASan/UBSan **3/3** 与独立 TSan **3/3** 均通过。CUDA CLI 与模型测试构建通过，全部 **10 个顶层模型场景** 通过，保留 F16/Q8_0 的 432 个逐层 span 比较、Writing/已提交 active 退休和精确 D2H 源槽归还等回归。Sanitizer 结果覆盖宿主组件，不覆盖 CUDA 适配器。十二个变更源码/构建文件的本地与远程 SHA256 一致。原始日志为 `historical-cleanup-{windows,host,asan,tsan}-test.log` 和 `historical-cleanup-gpu-test.log`，保存在本地 `build-local/capture-session-validation/`；远程验证始终使用 `/tmp/kvmem-gc-99bef27.3CD9MU` 隔离目录。

普通 CLI delete 使用相同模型、356-token 提示词、8-token 输出、容量/水位 32/14/16，生成与进程退出正常；本轮 `committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 为 0。完成时 `payload_bytes=26738688` 尚包含两个未用的私有预约块，符合异步清理语义；它不是 14 个 Live 块的最终占用。原始输出和日志为 `historical-cleanup-cli-delete.tokens/.log`。

目前仍是单请求、单 GPU、recency、RAM 的阶段实现。设备提交/完成失败时的取消与源数据保护尚需专门验收；server、多请求、所有 active 入口、GPU/metadata/backend/rollback 的完整预约、长期有界和实际 agent 效果均未完成。

### Packed K/V 搬运失败的完成证明与资源保护（2026-10-17）

发现并修复 reverse slab 的两个错误路径：原先部分 `cudaMemcpyAsync` 或 event record 失败会直接返回 `-1`，没有检查先前已提交工作是否排空，并可能把原始 GPU 指针留在 `outs` 队列；event wait 失败则可能立即转同步读取并归还源槽。API 报错本身不能证明先前工作已经结束，这不满足导师文档第 5 节“已提交任务必须真实完成或确证取消”的要求。

现在 stageout 记录未确认的设备工作、对应事件和可读结果。提交失败时，只有检查 stream 排空成功并移除该批原始指针后，才返回可安全回退的 `-1`。等待旧事件失败时，同样先确认排空，再允许同步回退，但保留下一批尚未提交的队列。等待成功前不能取得输出 pin 地址。清空队列、释放 staging buffers 或继续申请 staging 也不能绕过未解决的失败。

无法确认排空时抛出错误并保留原始任务、writer、CPU 存储及 GPU 源槽引用；适配器将当前请求关闭，后续追加被拒绝。随后即使能够确认旧任务完成，也只结算已获准的工作，不重新开放失败请求，不把 Retired 代发布成 Live。物理 CPU 回收仍须等待 writer 等真实依赖退出。持续无法排空的上下文析构选择显式终止进程，阻止 C++ 成员展开提前销毁 GPU 源或把 CPU owner 移交后台；这不是任意设备故障的服务恢复方案。

新增按 execution domain 安装的内部故障钩子，生产默认空，不使用全局环境变量改变错误行为。`kvmem-stageout-failure-test` 使用实际 CUDA 缓冲区和搬运，共检查 8 个场景：descriptor copy、gather launch、packed copy、第二次直接 copy、event record，event wait 的可排空/不可确认两种结果，以及提交后始终无法确认排空。验证失败后新批次读到新数据、旧等待不丢下一批队列，以及未确认期间 enqueue/clear/free/resize 均被拒绝。故障点为确定性注入，不代表实际诱发或恢复硬件损坏。

实际模型由 10 个增加到 **13 个顶层场景**。新增加的 record/wait 可恢复场景均完成 160-token 输入和 3-token 输出，**432 个 layer span** 的 CPU/GPU 字节比较通过。不可确认场景在实际图采集后关闭请求，直接检查每个未完成 job 的原 CPU payload、GPU 源槽和 reader 引用仍存在；对该 Writing 代退休后 `try_reclaim` 返回 Deferred。解除故障后原任务收尾，退休不复活，失败请求不被算作成功，最终上下文退出后 CPU owner 的弱引用过期。此处不依赖总 payload bytes 的瞬时相等，因为未用私有预约可同时在后台归还。

最终 CUDA 故障测试 **1/1（8 个内部场景）**、全部 **13 个模型场景**、Windows 相关宿主 **12/12**、Linux 全部宿主 **24/24** 通过。六个变更源码/构建文件的本地/远程 SHA256 一致。原始记录为 `historical-stageout-native-final.log`、`historical-stageout-final-gpu-test.log`、`historical-stageout-{windows,host}-test.log`，保存在本地 `build-local/capture-session-validation/` 和既定远程隔离目录。此前宿主 ASan/TSan 结果不能作为本轮 CUDA 代码的 sanitizer 证明。

额外运行 CUDA compute-sanitizer **未取得有效检查结论**：该 WSL/WDDM 环境报告 debugger interface 初始化失败、设备不受支持，工具以 2 errors 退出，该次进程也报告数据比较失败；原生 CUDA 故障测试随后独立重跑通过。完整失败日志保存在 `historical-stageout-cuda-memcheck.log`。不能把该项记为通过或宣称显存错误已被 sanitizer 排除，需在可用的 CUDA 调试环境补测。

普通 CLI delete 的最终回归同样正常完成生成与退出：既定模型、356-token 输入、8-token 输出、容量/水位 32/14/16，观测 `committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 为 0。日志和输出为 `historical-stageout-cli-delete.log/.tokens`；完成时 payload 仍可能包含后台尚未归还的私有预约。

上述阶段只处理 packed K/V reverse slab；Mean-K/Q capture worker 的后续进展见下一节。其它设备提交/完成路径及更完整的取消协议仍需核对与验收，随后还须补齐整请求资源预约、全部 active 入口、server、多请求和长期有界等原始交付项。

### Mean-K/Q 后台采集的错误传递与取消（2026-10-17）

单 GPU Mean-K/Q 采集线程原先没有顶层异常捕获：事件等待、Host 转换或写入抛出异常时，会离开线程入口并终止进程；前台等待条件也没有“后台已失败”分支。现在 worker 在受同一互斥量保护的错误槽中保存异常并唤醒等待者；前台在采集等待、flush、追加和 active 入口观察错误，关闭该采集域并传递异常。已经失败的上下文不能继续追加或通过 reset 重新开放，正常释放上下文仍可完成经过确认的设备清理。空任务队列也要观察 worker 初始化成功或失败，不能把尚未完成的线程启动误报为成功 flush。

任务队列改为固定两个位置，对应已有两个实际搬运槽。每批 item 列表在任何 DMA 提交前预留空间，提交队列不再分配内存；槽位完成标志改为原子发布，统计值由 worker 在锁内提交、前台汇总，消除原先后台写统计与前台重置/打印统计的竞争。这只限制当前采集流水线队列，并不是整请求 payload、元数据和临时空间均已预约或有界的证明。

发生后台错误时，不伪造 inflight 完成，不丢弃出错批次或后续队列的 producer/writer，也不重试已经写入的 Q/Mean-K。上下文退出先停止并 join 采集线程，再检查真实 CUDA stream 排空；已标记失败的请求还须确认设备上的源计算结束，防止 compute → capture 依赖未能建立时只等待空的采集 stream。确认完成后才取消未提交的 CPU 工作、释放任务引用。已经在 CPU 写入中的一次任务可先完成；尚未开始的排队任务可以取消。持续不能确认设备完成时保留引用并拒绝释放；最终析构沿用显式终止进程的安全边界，尚未实现任意设备错误后的服务恢复。这里等待的是采集线程及实际设备工作，不是 GC worker。

部分 D2D、D2H 或完成事件提交失败时，只有检查 stream 排空成功，才允许使用保留在原位置队列中的输入做同步回退；排空失败则关闭采集域并保留任务。生产同步回退还会先完成前一批采集，避免前后台同时修改 Q，或后一段 Mean-K 越过尚未写入的前缀。已退休块即使还有旧任务完成，也不能重新发布成 Live。

新增确定性故障场景使用实际 CUDA 搬运和模型的层/头维度，Mean-K/Q 输入为可逐项校验的人工张量，不冒充自动生产接入。覆盖 worker 设备选择失败、事件等待失败、写完 Q 与第一层 K 后的 Host 写入失败、第二次 D2D 失败、D2H 失败、完成事件失败及无法确认排空。双批次测试将第一批停在明确屏障，确认第二批已经排队后再触发错误；检查错误唤醒、后继任务不被执行、同代 writer 保留、Retired 回收延期、Q/K 精确前缀，以及取消后 CPU payload 可释放。另一个场景通过生产 `harvest_pending` 触发第二批同步回退，验证它等待第一批采集并最终只累计 16 个 Q/Mean-K token。

最终 CUDA 模型验证 **21/21 个顶层场景** 通过：保留原有 13 个场景，新增 7 个故障场景和 1 个生产同步回退顺序场景。Windows 相关宿主回归 **12/12**、Linux 全部宿主回归 **24/24**、既有 packed stageout 故障测试 **1/1（8 个内部场景）** 通过。顺序测试初次因夹具没有提交第一批 packed KV 而被正确拒绝；补齐真实依赖后通过，失败日志单独保留，未修改产品的前缀检查。日志为 `historical-worker-final-gpu-test.log`、`historical-worker-{windows,host,stageout}-test.log`，本地保存在 `build-local/capture-session-validation/`，远程继续使用 `/tmp/kvmem-gc-99bef27.3CD9MU` 隔离副本。

同版 CLI delete 完成 356-token 输入、8-token 生成和上下文退出：容量/低高水位仍为 32/14/16，`proposed=committed=reclaimed=6`，`live_observed=14`，`task_references=0`，`gpu_deferred=gpu_returned=6`，policy/worker failure 均为 0。瞬时 `payload_bytes=26738688` 仍包含尚未后台归还的私有预约，不能直接解释成 Live 历史的最终占用。原始记录为 `historical-worker-cli-delete.log/.tokens`；三个代码文件的本地与远程 SHA256 一致，记录在 `historical-worker-source-sha256.txt`。这仍是短请求功能回归，不是长期内存或性能验收。

这一步仍不开放生产 retrieval 的历史 GC，也未覆盖 multi-GPU 采集、decode accumulator 的全部错误分支、MTP 或 NVMe 的设备失败恢复。CUDA sanitizer 的环境限制仍在，不能将本轮原生测试或此前宿主 sanitizer 结果当作 CUDA 适配器通过 sanitizer 的证据。

### 解码 Mean-K 累积、部分接受与失败取消（2026-10-17）

逐 token 解码及接受草稿前缀时，Mean-K 使用独立 GPU running sum。原实现有三个失败风险：清零与等待错误被忽略；累加 kernel 报错返回 false 后会转 CPU 重算，无法证明 GPU 是否已经修改过部分和；D2H 直接写入调用方临时向量，等待失败时该向量可能先于 DMA 释放。另一个顺序风险是接受后立即丢弃输入批次，未确认独立 stream 上读取这些图张量的累加 kernel 已结束。

现在 GPU accumulator 记录 pending/failed 状态。已提交操作的错误抛出异常并关闭累积器，不能当作“尚未执行、可以同步回退”的 false；正常 false 只用于不支持的输入或提交前分配不足。读取结果改用该 execution domain 持有的一层 pinned staging（`n_embd_k × sizeof(float)`），确认完成后才复制到调用方，失败时调用方缓冲区保持原值。清零、累加、读取、等待都检查结果；取消只确认设备任务结束并保留失败状态，释放和重新初始化是显式边界。无法确认完成时，取消、free 和外层 staging free 均不能释放相关分配。

适配器在 ingest/commit/discard/flush 错误时关闭请求，保留原 pending batch、writer 及部分结果。失败后的 commit、flush、普通草稿 discard 和新追加均被拒绝，不能靠重试把同一前缀加两遍。已接受的累加任务在丢弃输入批次前完成；下一次 graph 才能复用相应张量。尚未接受或拒绝的草稿不能被下一次 ingest 覆盖。托管历史缺少任一必要层或完全没有采集节点时明确失败，不再静默跳过。正常草稿拒绝仍只丢弃未接受范围；历史块退休后，旧 writer 可以完成已经接受的前缀，但不能重新发布为 Live。

上下文退出不再重试已失败的部分和：正常 flush 出错也转入取消路径。先通过原 GPU owner 确认源计算及其它采集任务结束，再检查 accumulator stream 排空，随后释放 decode task 的引用；packed K/V 已提交任务仍按原完成协议结算。持续无法确认设备结束的最终析构仍采取显式终止进程，不能把该实现解释成任意硬件错误后的服务恢复。新增同步属于当前设备数据依赖，不依赖 GC 继续回收。

新增独立 `kvmem-meank-failure-test` 使用实际 CUDA 缓冲区和 kernel，检查接受片段的精确和、完成后复用源张量、逐层清零、初始化及清零失败、实际提交后模拟累加结果报错、读取/等待失败和无法确认排空；验证失败后调用方缓冲区不被晚到 DMA 改写，以及取消后不能直接重开累积器。模型上下文测试另覆盖第二层累加失败、接受后等待失败、读取和 Host 写入部分失败、覆盖未决草稿、缺层/无节点，以及“全部拒绝，再接受 4/8 token，退休后完成旧前缀”。故障用例在 packed 任务已结算后单独验证 Mean-K writer 仍使 CPU 回收延期，确认取消后 CPU payload 才可释放。它们使用模型维度的已知张量，不等于真实 MTP 接受率或完整生产 retrieval GC 已验收。

最终模型上下文验证 **30/30 个顶层场景** 通过，保留上一轮 21 个，新增 8 个故障场景和 1 个正常 accept/reject 场景。独立 CUDA CTest **2/2** 通过，其中 Mean-K 测试执行 7 个内部场景，packed stageout 保留原有 8 个。Windows 宿主回归 **12/12**、Linux 宿主回归 **24/24** 通过。七个代码/构建文件的本地与远程 SHA256 一致。原始记录为 `historical-decode-final-gpu-test.log`、`historical-decode-native-final-test.log`、`historical-decode-native-detail.log`、`historical-decode-{windows,host}-test.log` 与 `historical-decode-source-sha256.txt`。

普通 CLI retrieval 使用同一 GGUF、356-token 提示词、8-token 生成、Q8_0 KV、预算 512，正常完成生成与退出。真实解码日志为 `n_tok=8, n_flush=1, n_gpu=384, n_host=0, n_miss=0`；48 个 attention 层的 8 个接受 token 均在 GPU 累积，随后一次 flush 写入。这验证真实生成经过本轮修改路径；该运行未开启 historical capture/GC，不证明生产 retrieval GC 已接入，也不是前后性能对照。原始日志和输出为 `historical-decode-cli-retrieval.log/.tokens`。

同版 recency CLI delete 使用既定 356/8 token 和容量/水位 32/14/16，正常生成并退出，`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`，policy/worker failure 均为 0。本次 `gpu_deferred=gpu_returned=2`：这两项只统计退休时仍有源读取依赖而延期归还的 GPU 槽，不要求每次等于 CPU 回收数；实际值受退休与搬运的时序影响。结束时 `payload_bytes=26738688` 仍是包含待归还私有预约的瞬时观测。原始记录为 `historical-decode-cli-delete.log/.tokens`，上述日志同时保存在远程隔离副本与本地 `build-local/capture-session-validation/`。

这一修改仍未完成全请求 GPU/pinned/metadata/scratch 预约，也未覆盖其它 stage-in、layout、多 GPU、NVMe 和 MTP 全链路的所有设备错误。新 staging 的大小有明确上界，但仍须纳入后续请求统一预约；当前结果不能证明长期总内存有界或前台没有任何动态分配。CUDA sanitizer 的环境限制继续保留，不用宿主 sanitizer 结果代替适配器证据。

### 请求入口准备 packed K/V 搬运资源（2026-10-17）

本轮推进导师方案中的“先取得实际资源，再接纳请求”。原来 recency historical capture 主要提前准备 RAM payload，packed K/V 的 GPU scratch、双 pinned slab、事件、搬运描述符和适配器任务记录仍在运行时分配；任务交接通过移动赋值还会丢掉另一批队列的容量。CPU 提交前会再构造一个临时字节向量，同步回退也逐 job 分配临时缓冲区。

现在 `begin_historical_capture` 在发布请求 owner 前准备这些搬运资源：一个至少 32 MiB 的 GPU packed slab、一个块大小的 F32 scratch、两个各 32 MiB 的 pinned CPU slab、3 个事件、8192 项设备描述符及对应 Host 描述符、两批各最多 8192 项的适配器任务，以及容纳一个最大 K/V packed slice 的同步回退缓冲区。执行 stream 在准备阶段建立，仍归 execution context 所有。工作区只接纳全新的 staging 域，不能覆盖已有队列、缓冲区或未完成的任务。新增 `KVMEM_CAPTURE_WORKSPACE` 日志报告实际准备量；模型 F16 场景为 device 33816576 bytes、pinned 67108864 bytes、Host 队列 2424832 bytes、fallback 32768 bytes。这不是模型总显存或整请求内存预算。

准备阶段任一 CUDA 分配、事件或 Host 队列准备失败，均不发布 historical owner；本轮工作区已取得的资源在返回前释放，C++ 分配异常同样走清理。释放检查设备完成和 CUDA 返回值；不能确认完成或释放失败时保留尚未释放的指针，不把失败伪装为完成回滚。`gpu_ready` 对已经准备好的工作区只检查尺寸，不再扩大；队列达到数量或 slab 字节上限时，由前台完成/提交当前搬运批次后继续使用原空间。这是当前任务的数据依赖，不等待 GC 提供空位。

任务交接改为交换两个已分配的队列；上一批清空后保留容量供下一次使用。`HistoricalKvRawStore` 增加指针与长度形式的 packed slice 接口，在调用期间把数据直接复制到同代已预约 payload，不保留外部指针；既有 vector API 仍可用并复用相同校验。提交继续检查 writer、仓库、代际、范围及连续前缀，旧 writer 退休后只能完成原写入，不能重新发布。同步回退使用入口准备的缓冲区；它只能在既有协议确认设备工作结束后执行，不改变失败任务和 GPU 源槽的保护规则。

新增证据：

- 独立 CUDA 工作区测试逐个注入 **11 个准备失败点**，检查 GPU/pinned/event/Host 队列容量清零并允许重试；另外真正禁止 C++ 分配，使 HostItems 准备抛出 `bad_alloc`，检查此前 CUDA 分配全部回滚。
- 同一测试把队列限制为 2 项，在禁止 C++ 动态分配时完成 **24 批**真实 CUDA 搬运，检查饱和拒绝不改变偏移、双 pinned slab 交替复用、输出逐字节一致、超出准备尺寸被拒绝，以及工作区占用不变。另验证已有 legacy 工作区不会被新的预约破坏。
- 宿主 raw-store 测试在禁止 C++ 分配时直接提交栈上的 K/V 数据，覆盖空指针和错误长度拒绝；修改原缓冲区后，payload 保持正确，确认没有保留借用指针。
- 模型场景由 30 个增加到 **31 个**：新场景在第二个 pinned 分配前注入失败，确认没有历史 owner/行记录及工作区残留，然后重试。实际执行时把适配器队列限制到 8 项，多次饱和交接后完成 160-token 输入和 3-token 输出，**432 个 layer span** 的 CPU/GPU 字节比较通过；每批还检查两个队列的地址/总容量及 fallback 地址/大小未变。原有 F16/Q8_0、退休、设备错误、同步回退及取消场景全部保留。

最终 Windows 相关宿主 **12/12**、Linux 全部宿主 **24/24**、独立 CUDA CTest **3/3**、模型上下文 **31/31** 通过。首次独立 CUDA 回归中，既有 stageout 故障夹具报告过一次数据不一致；检查发现它用默认 stream 上传 pageable 数据后，没有建立到独立 nonblocking 搬运 stream 的完成边界。夹具新增显式的上传完成检查，随后三项测试全部通过，未放宽输出比较。首次失败日志保留为 `historical-workspace-native-initial-failure.log`，最终日志为 `historical-workspace-native-test.log` 和 `historical-workspace-native-detail.log`。

其余记录为 `historical-workspace-{windows,host}-test.log`、`historical-workspace-gpu-test.log`、构建日志及 `historical-workspace-source-sha256.txt`，保存于本地 `build-local/capture-session-validation/` 与远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。11 个源码/构建文件的本地与远程 SHA256 一致。CUDA sanitizer 环境限制继续保留；本轮没有新的 sanitizer 通过结论。

普通 recency CLI delete 同样完成生成与退出：既定 GGUF、356-token 输入、8-token 输出、Q8_0、容量/水位 32/14/16，`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 为 0。结束时 `payload_bytes=26738688` 仍可能包括等待清理的私有预约；不把该瞬时数当作 Live payload 的最终占用。原始记录为 `historical-workspace-cli-delete.log/.tokens`。本轮没有独立前后性能对照，不能据此宣称加速。

**范围仍是组件级预约。** RAM session 当前在外层 GPU 准备前建立；如果后续工作区准备失败，未用 RAM 仍通过已有有界 CPU cleanup 归还。这尚不是全部资源在一个事务中的立即回滚。实际 GPU 工作集、layout scratch、目录/producer 元数据、backend、Mean-K/Q、rollback 等仍有未纳入统一预约的分配。原生测试中禁止 C++ 分配只覆盖所测搬运/提交区间，不是整个模型请求零动态分配的证明。当前仍处于 P3，下一步优先处理 layout scratch 与目录/producer 等实际分配，并逐项接入统一的请求准备和失败回滚。

### 主模型布局临时区预约与搬运完成边界（2026-10-17）

继续推进请求资源准备，处理主模型 `layout_gpu_slots_by_orig_pos`。布局整理会把仍在 GPU 的历史块按原始位置排成连续槽位，它属于前台工作集管理；后台 GC 的回收对象仍是 CPU historical payload。本轮没有让 GC 修改 GPU 工作集。

原布局每次临时 `cudaMalloc`，每层还建立六个搬运向量。现在历史请求入口准备 `n_slots × block_tokens × (packed_k_row_bytes + packed_v_row_bytes)` 大小的 GPU 布局临时区，以及有界的项目、槽位检查、索引和 gather/scatter 描述符数组。各层完成后复用同一个临时区，因此大小不乘层数。准备量包含全部物理 GPU 槽位，支持最坏情况下全部驻留块需要移动；尺寸计算检查溢出，执行时验证槽位唯一、token 数及真实 tensor 边界。工作区归 execution owner 持有，历史请求不能在执行中扩大；legacy 调用可在开始复制前准备空间，分配不足时才允许原有 Host 回退。

同时发现并修正了完成边界：原主模型布局的 batched copy 在独立 nonblocking stream 上提交，但调用方等待的是 `cudaStreamPerThread`。这并不证明搬运完成；每次 `cudaFree` 的隐式影响不能作为长期复用缓冲区的协议。新 `kvmem_layout_copy` 在实际提交 stream 上逐批检查完成，再允许更新 Host 描述符、开始 scatter 或进入下一层；超过 8192 项的描述符分批处理，不扩大描述符存储。布局开始移动前结算已有 packed/Mean-K capture，避免覆盖仍被读取的 GPU 源位置。

已提交的 descriptor/kernel/wait 错误会抛出并关闭布局及请求，不能当作“尚未执行”转入 Host 重做或继续使用半更新缓存。主模型的槽位元数据在所有层复制成功之后才重排；错误后不提交新映射。若无法确认设备完成，临时区和共享 staging 分配保持所有权，free/复用均被拒绝；确认取消只解除设备依赖，不重新开放已失败的上下文。最后析构仍保留“持续无法确认完成则终止进程”的边界，未实现任意硬件错误后的服务恢复。

新增和扩展验证：

- 预约故障点由 11 个增加到 **12 个**，加入 layout 分配；在 layout 已分配后制造真实 Host `bad_alloc`，检查所有工作区资源释放后可重试。
- 新独立 layout 测试在禁止 C++ 动态分配时完成 **30 轮循环换位**，分别覆盖已预约的 gather kernel 路径与 legacy direct-copy 路径，逐字节检查移动结果和未选中尾部。另将 **8195 条描述符**拆成多批，验证没有丢失或覆盖未完成记录。
- 同一测试覆盖 descriptor 提交、实际 kernel 提交后的结果错误、完成等待错误，各分能确认/不能确认排空两种情形，共 **6 个故障场景**。检查失败后不能重放，未确认时 free 不改变工作区占用，取消后仍不能复用失败缓存，最终显式 free 后分配归零。
- 模型上下文场景由 31 个增加到 **35 个**。新成功场景在实际生成 48 个输入 token 后交换两个 GPU 驻留块及其位置记录，再调用生产布局函数；所有注意力层的 K/V 与原 CPU payload 逐字节一致，入口准备的十组 Host 数组地址和 GPU 占用不变，然后继续完成 160-token 输入、3-token 输出和原有 432 个 layer span 检查。该场景主动安排布局以覆盖搬运路径，不等于生产 retrieval GC 已自动接入。
- 另三个模型场景分别在 gather kernel 后、scatter 完成等待时和无法确认排空时注入错误；检查请求关闭、不能继续追加、旧槽位元数据未被新映射替换、未确认时 scratch 保留，取消和实际上下文销毁后 CPU owner 最终释放。

最终 Windows 宿主 **12/12**、Linux 宿主 **24/24**、独立 CUDA CTest **4/4**、模型上下文 **35/35** 通过。新增场景使用的 F16、7 个 GPU 槽位配置，layout 显存为 458752 bytes，布局 Host 数组为 882 bytes；整个已准备 staging 的 device 总量为 34275328 bytes，pinned 仍为 67108864 bytes。八个源码/构建文件的本地与远程 SHA256 一致。记录为 `historical-layout-{windows,host}-test.log`、`historical-layout-native-test.log`、`historical-layout-native-detail.log`、`historical-layout-gpu-test.log`、`historical-layout-source-sha256.txt` 及构建日志；远程继续只使用 `/tmp/kvmem-gc-99bef27.3CD9MU` 隔离目录。

普通 CLI retrieval 使用同一模型与 356-token 提示词、Q8_0、预算 128，正常完成 8-token 生成和退出。实际 trace 为 `layout_writeback move=4 raw=0 d2d=1`，确认生产检索流程真正经过本轮布局代码；统计还记录 64-token query replay，因此总 prompt eval 为 420 token。这次运行未开启 historical capture/GC，验证的是普通检索兼容性，未证明生产 retrieval GC 已接通。原始记录为 `historical-layout-cli-retrieval.log/.tokens`。

同版 recency CLI delete 使用预算 512、356-token 输入、8-token 输出、容量/水位 32/14/16，正常生成和退出；`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=2`，policy/worker failure 为 0。该配置额外预约 layout 835584 bytes、Host 布局数组 3024 bytes。CPU 回收数与延期归还 GPU 源槽数含义不同，不要求相等；结束时 `payload_bytes=26738688` 仍可能含待后台归还的私有预约。记录为 `historical-layout-cli-delete.log/.tokens`。两次 CLI 均为功能回归，没有前后性能对照，不据此宣称加速。以上原始日志及源码摘要同时保存在本地 `build-local/capture-session-validation/`。

当前已预约的是主模型单 GPU、非转置 K/V 的布局 scratch 和搬运记录。`occupy_in` 的 cell/ubatch 元数据、冷块 restore、完整目录/producer、backend、GPU 工作集统一计费及 rollback 仍未全部纳入请求事务；MTP 自身的布局、多 GPU/转置 V 的 Host 回退和其它 stage-in 错误也须继续验收。不能把所测复制区间零 C++ 分配推导为整个布局函数或整个请求零动态分配。下一步优先接入目录、producer 和 cell 元数据的实际预约，并统一入口失败回滚；完整 P0–P6 范围继续保留。CUDA sanitizer 环境限制未改变，本轮没有新的 sanitizer 通过结论。

### 历史目录与 producer 实际预留、请求准备/接纳分离（2026-10-17）

本轮继续落实导师方案中“接纳前取得真实资源”的要求。之前 RAM payload 已提前分配，但每次追加仍会建立目录 map 节点、producer 对象和 shared_ptr 控制块，并可能扩大 dense block 表；另外，RAM session 在外层 GPU 工作区准备完成之前就被标记为已接纳。GPU 准备失败时，未用 RAM 会进入后台 cleanup，连续重试可能占着清理队列名额。

现在按同一请求碎片上界提前分配真实目录节点、producer 对象及控制块、dense block 表容量和批量校验临时数组。`AppendReservation` 持有这些私有资源；提交后通过 C++17 map node handle 迁移节点，直接交接已经存在的 producer，不再在成功追加路径创建这些对象。先校验整个批次，再消费 writer 并提交行记录；后面的 span 错误不会让前面的 span 部分登记。预留额度固定，不能通过 legacy append/bind 绕过。目录仍只持 producer 的弱引用，任务取得的 writer 独立存活；销毁未用预留不取消已交接任务，也不把退休块重新发布成 Live。

请求入口拆为 `try_prepare()` 和 `admit()`。前者只准备私有 RAM/元数据及 cleanup 名额，不改变公开目录或块表容量，不允许追加或执行；外层继续准备队列、布局及 GPU/pinned 工作区，全部成功后才提交接纳。提交前再次检查 runtime 仍为空且没有待执行计划。未接纳请求失败或取消时同步退还私有资源及清理名额，不依赖 GC 或 CPU cleanup worker；接纳后的完成/取消继续使用此前的后台清理规则，避免把未用 RAM 的释放搬到请求完成的前台路径。

新增宿主验证包括：

- 对入口逐个注入 C++ 分配失败，检查公开目录/块表及清理名额不变；准备后修改 runtime，接纳会拒绝且保留新数据。零 token 请求仍可正常接纳和结束。
- 禁止 C++ 动态分配时完成接纳与一次 9-token、3-span 追加，检查没有前台释放、块表容量不变且元数据额度精确消耗。
- 对批量 span 间隙、错误 runtime、输出数组容量不足及额度耗尽分别拒绝；修正后可在禁止分配时成功登记。已经退休的 writer 可继续原写入，不能重新发布；实际任务退出后可以回收，证明预留池没有额外长期持有 producer。
- 暂停独立 CPU cleanup worker，反复准备后销毁或显式取消请求，检查 payload 归零、repository 所有者消失、块表不变及名额立即返还。另保留“已接纳请求完成时后台清理暂停”的原有测试，确保两种生命周期没有混淆。

实际模型的既有饱和队列场景扩展为：先暂停 CPU cleanup，再让第二个 pinned 分配失败，检查 GPU/pinned/event/Host 工作区、目录及两个请求清理名额均退回，随后重试并完成真实推理。每批执行额外检查 dense block 表容量保持入口准备的值；逐场景完成记录便于核对全部 35 个顶层场景。

Windows 相关宿主 **12/12**、Linux 全部宿主 **24/24**、独立 CUDA **4/4**、宿主 ASan/UBSan **5/5**、独立 TSan **5/5** 通过。TSan 首次因默认地址随机布局在启动时触发 `unexpected memory mapping`；仅对子进程使用 `setarch x86_64 -R` 后全部通过，未修改主机全局配置。保留首次启动失败日志。这些 sanitizer 结果只覆盖宿主组件，CUDA sanitizer 的既有环境限制仍在。

最终模型验证 **35/35** 通过，进程退出码为 0，包含暂停 cleanup 的 GPU 准备失败与重试、每批 dense block 表容量不增长，以及既有 F16/Q8_0、退休、异步写入、设备错误和布局场景。首次模型运行的外层脚本在记录退出码时引号错误，不能据此认定进程正常退出；保留初次日志，修正脚本后完整复测，最终证据为 `historical-metadata-final-gpu-test.log/.exit`。八个源码/测试文件本地与远程 SHA256 一致；摘要、各项构建/测试及首次 TSan 启动失败日志均保存在本地 `build-local/capture-session-validation/`，远程仍只使用 `/tmp/kvmem-gc-99bef27.3CD9MU` 隔离目录。

同版 recency CLI delete 完成 356-token 输入、8-token 输出并正常退出；入口实际准备 **22 个元数据位置**，容量/低高水位仍为 32/14/16，`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 均为 0。结束时 `payload_bytes=26738688` 仍可能包含未用私有预留，不能当作 Live 历史的最终占用。原始记录为 `historical-metadata-cli-delete.log/.tokens/.exit`；没有前后性能对照，不宣称提速。

**本轮仍是有限单请求的资源准备。** 预留节点数随该请求的最大追加需求增长；这没有完成多请求累计追加时的目录/墓碑复用，也不是所有资源的 byte 计费或长期内存有界证明。cell/ubatch、ActiveSet、backend、冷块恢复、GPU 工作集和 rollback 等仍有执行中分配，后续继续逐项纳入入口事务。旧动态目录 API 保留给未启用这一请求模式的路径。当前仍处于 P3，完整 P0–P6 交付范围不变。

### ActiveSet 及块 cell 临时记录的入口预留（2026-10-17）

上一轮完成历史块创建记录的预留后，本轮处理执行窗口。原 `begin_active_ubatch` 每次新建三个候选数组，并创建新的 ActiveSet 对象及 lease/handle 数组；`occupy_in` 每次重建一个块的槽位、位置、序列、token 等临时数组。现在在请求私有准备阶段，按物理 GPU 槽位数预留 active 候选、待移除记录、槽位校验数组和一个 ActiveSet；另按 `block_tokens` 准备单块 cell 临时数组。外层 GPU 准备失败仍同步回滚这些私有资源，接纳后不能扩大它们；退出时随已确认设备完成的 CPU 所有权一起交给后台 cleanup。

ActiveSet 增加 `prepare`、`try_acquire` 和 `release`。复用的是对象和数组，每个新窗口仍逐块取得新的 ExecLease，不继承上一窗口的权限。已经退休的可选块仍被跳过，缺少必需块时退回本次已经取得的全部许可；超过准备上限会在获取许可前拒绝。已有窗口未结束时返回 Busy 并保留它，空选择也标为一个已接纳窗口，不能用空数组绕过重叠检查。仅在实际提交后更新 LRU；放弃未执行计划不更新。调用 `release` 前仍由设备 owner 证明执行完成，该方法本身不等待设备。旧按次分配的 `acquire` 接口保留并复用同一套校验，避免两套许可规则分歧。

生产适配器在既有 `harvest_pending` 确认执行依赖完成之后释放许可，保留 ActiveSet 容量供下一窗口使用。若采集抛出错误，完成调用不会执行，许可继续保留至原有取消/设备确认路径。选择阶段检查 GPU 槽位范围及重复映射，防止错误 residency 突破预留数组上界；登记失败时本次许可由作用域回滚。cell 临时记录只在同步 `apply_ubatch` 调用期间借用，没有被提交给异步图或跨调用保存。

验证结果：

- 新增 ActiveSet 准备分配故障测试；禁止 C++ 分配时在同一物理槽位上完成 **64 代**“新许可 → 退休 → 旧执行保护 → 退出 → 回收”。检查数组地址/容量不变、没有前台释放准备数组、重叠拒绝及必需块失败不会泄漏许可。该结果仅证明组件复用，不代表生产目录已通过长期追加门槛。
- 模型测试在实际完成 48-token 输入后，禁止前台 C++ 分配并连续进行 **32 次**新 active 接纳及放弃，检查许可最终归零且 LRU 没有更新。原有 35 个模型场景全部通过，进程正常退出；每批推理和实际布局移动还核对 active 对象与 15 组数组地址、容量不变。入口 pinned 分配失败场景同时检查新准备空间没有发布或残留。
- Windows 宿主 **12/12**、Linux 宿主 **24/24**、独立 CUDA **4/4**、宿主 ASan/UBSan **4/4**、独立 TSan **4/4** 通过。TSan 沿用仅对子进程禁用地址随机化的配置；宿主检查不覆盖 CUDA 适配器的全部并发行为。

模型场景的 7 个物理 GPU 槽位对应 active 数组容量 **791 bytes**，单块 cell 临时数组 **1468 bytes**；这是容器持有的数据容量，不含对象自身、分配器和其它执行资源开销。本轮没有修改 GC 的回收对象，GC 仍回收 CPU historical payload，GPU 源槽仍遵循原有前台设备完成协议。

同版 recency CLI delete 正常完成 356-token 输入、8-token 生成和退出：32/14/16 的容量及水位配置下，`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 为 0。24 个 GPU 槽位的 active 数组容量为 **2712 bytes**，cell 临时数组仍为 1468 bytes；结束时 `payload_bytes=26738688` 仍可能包含等待清理的私有预留。普通 retrieval（未启用 historical GC）也正常完成 8-token 生成，实际移动 4 个驻留块，trace 为 `layout_writeback move=4 raw=0 d2d=1`，输入 356 token 加 64-token replay，总 prompt eval 420 token。它的退出诊断仍显示 compute buffer 实际 15.9652 MiB、预估 12.2700 MiB；对照上一轮 `historical-layout-cli-retrieval.log` 已有相同数值。这是后续 backend 工作区预算必须覆盖的现有缺口，不能据本轮功能通过宣称严格资源承诺已成立。

六个源码/测试文件的本地与远程 SHA256 一致。原始日志为 `historical-active-workspace-{windows,host,asan,tsan,native,gpu}-test.log`、`historical-active-workspace-cli-{delete,retrieval}.log/.tokens/.exit` 及构建日志；模型测试退出码另存于 `historical-active-workspace-gpu-test.exit`，三次模型/CLI 运行均为 0。资料保存在本地 `build-local/capture-session-validation/` 和远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。本轮没有前后性能对照，不宣称提速；CUDA sanitizer 的环境限制保持不变。

本轮检查还明确了下一处实际分配：上游 `llama_kv_cells` 的 `used` 和 `seq_pos` 是节点容器，`apply_ubatch` 的插入、移除仍会分配/释放节点。完整 `llama_ubatch::data_t`、slot_info 列表、memory context、采集范围及 rollback 快照也没有全部准备。不能把外层 cell 临时数组复用或 active 接纳零分配解释为完整 ubatch/模型请求零动态分配。下一步须继续处理这些真实节点及上下文所有权，覆盖 prepare、apply、取消与恢复路径，再推进其它 active 入口、server、多请求及完整 P4 长轨迹。P0–P6 总目标保持不变。

### 上游 cell 索引的真实节点预约与复用（2026-10-17）

上一轮只准备了重建单块 cell 所需的外层数组，上游 `llama_kv_cells::used` 和 `seq_pos` 的 `std::set` 仍会在插入时分配节点、删除时释放节点。本轮增加 `ReservedNodeSet`，在请求准备阶段实际分配并提取标准容器节点，执行时在“备用池”和“有序索引”之间转移节点。删除、清空和恢复均复用原节点，容量不足明确失败；未启用预约的实例保留普通有序集合行为。复制整个索引仍会分配，不能把它当成已预约的快照接口。

维护补丁 `0008-historical-cell-index-reservation.patch` 将这一机制接入真正的上游 cell 类。当前受限历史请求只允许一个序列，按实际 cell 数 N 准备 used 和该序列的位置索引各 N 个节点，其他序列的预约容量为零。插入、删除、位置变更、连续/非连续恢复及同尺寸 reset/resize 都使用该池；不同尺寸增长及未预约序列的写入/恢复在改动原状态前拒绝。新请求只接受无在用 cell、无旧预约、无待处理 position shift 的缓存。重复位置仍以 `(position, cell index)` 区分，查询顺序保持原行为。

准备分为 `prepare → install → accept`：节点先由私有 token 持有，在已有 RAM/元数据 session 接纳之前安装；如果 runtime 已变化导致接纳失败，token 析构撤销安装并归还全部节点。安装与接受之间由同一个前台 owner 串行处理，不允许应用 cell。准备失败不会发布 historical owner，也不等待 GC 提供资源。接受后节点池归 cell 缓存持有。日志 `KVMEM_CELL_INDEX` 报告真实节点数量；模型测试为 **448 个**，本轮 CLI 为 **1536 个**。节点数量不是 allocator 字节数、RSS 或完整请求内存预算。

补丁同时将 `LLAMA_KVMEM` 及所需头文件路径作为 libllama 的公共构建属性，确保使用内部 cell 头文件的调用方与库采用相同结构布局。Linux、Windows PowerShell 和 Windows HIP 构建脚本均纳入补丁；后两者同时补齐先前未自动应用的 `0007` active hook。Linux 重复应用检查及三文件的撤销/重放/逐文件比较通过，PowerShell 脚本语法检查通过；本轮未执行 Windows CUDA 或 HIP 完整构建。

新增测试及结果：

- 节点容器在准备阶段逐项注入 C++ 分配失败；安装后禁止 `new`，执行 **512 次**插入、删除、清空及有序查询，对照标准集合并检查没有节点分配/释放、节点数量守恒。另验证容量耗尽、零容量、复制后的备用容量及移动所有权。
- 实际上游 cell 测试完成 **128 轮**插入、移除、位置调整、清空和两种恢复。测试在这些操作期间禁止 C++ 分配并统计释放；还覆盖 RAM 接纳失败后的节点回滚、准备后缓存变化、待处理 shift、普通多序列共享 cell，以及预约非零序列的恢复。
- 真实模型布局测试在调用真正的 `occupy_block_cells` / `apply_ubatch` 时禁止 C++ 分配；每批推理检查节点总数固定、已用及备用节点数量守恒。全部 **35 个模型场景**通过，保留 CPU payload 退休/释放、旧任务保护与设备失败回归。
- Windows 相关宿主 **13/13**、Linux 全部宿主 **26/26**、专项 CTest **5/5**（1 个实际 cell 宿主测试、4 个 CUDA 测试）、宿主 ASan/UBSan **4/4**、独立 TSan **4/4** 均通过。后两组覆盖节点池、实际 cell、capture-session 和 GC，不是 CUDA sanitizer 证据。

同版 recency CLI delete 完成 356-token 输入、8-token 生成和退出。容量/低高水位为 32/14/16，`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 均为 0；结束时 `payload_bytes=26738688` 仍可能包含后台尚未归还的私有预留。本轮不改变回收对象：GC 回收 CPU historical payload，GPU 源槽继续由前台确认设备完成后归还。

普通 retrieval（未开启 historical GC）也正常生成 8 token 并退出：`layout_writeback move=4 raw=0 d2d=1`，输入 356 token 加 64-token replay，prompt eval 共 420 token，解码 Mean-K 为 `n_gpu=384, n_host=0, n_miss=0`。退出时仍有此前已记录的 compute buffer 实际 15.9652 MiB、预估 12.2700 MiB 差异，后续 backend 统一预算须处理，不能当作严格资源承诺已经完成。模型测试、delete CLI、retrieval CLI 的退出码均为 0。

原始日志使用 `historical-cell-nodes-` 前缀，保存在本地 `build-local/capture-session-validation/` 和远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。13 个源码/构建文件的本地与远程 SHA256 一致；节点补丁重放后的三个上游文件与实际测试版本逐文件一致。仅新增节点预约及复用的功能证据，没有前后性能对照；CUDA sanitizer 的既有环境限制保留。

下一处明确缺口仍在 `init_batch` / `prepare_ubatches`：完整 `llama_ubatch::data_t`、slot_info 列表、memory context、采集位置/范围和 rollback 快照仍会动态分配。必须继续准备这些真实存储及其所有权，并纳入 backend、冷块恢复及 GPU 资源的统一计费。本轮 cell 恢复测试只证明已经存在的快照可复用索引节点，不代表快照创建、整个模型请求或完整生产 GC 已达到零动态分配和长期有界；P3 仍进行中，P4–P6 交付范围不变。

### 请求入口预约采集位置与范围（2026-10-17）

前一轮模型验证后，`prepare_ubatches` 仍为每个微批次创建位置向量；`capture_batch` 动态增长原始范围向量，D2H 槽位复制这些范围时还会再次分配。现在受限 historical 请求在 `begin_historical_capture` 的私有准备阶段取得固定字节 arena，并按输入加最大可接受输出的 token 总数预留位置队列。每个实际批次的位置和最多同数目的范围都从该 arena 取得；在已接纳请求中，arena 耗尽会明确失败，不回退到普通进程分配器。直接使用微批次的位置指针，删除了原先每批中间位置向量的复制。

一个批次可移动至同步采集、异步 D2H 槽位和后台清理 owner。`CaptureBatch` 持有 arena 的共享所有权，使尚未完成的异步任务不会拿到已释放的内存；批次结束时，范围中的 writer 引用依旧按原生命周期析构，固定 arena 仅把字节容量留到请求安全退出。新请求要求旧位置队列及待处理批次为空。准备在 RAM/元数据接纳前完成，失败不会发布 arena 或队列；接受以后交换所有权，不新增分配。入口及完成日志分别记录预留字节、实际用量和累计位置数。

最终真实模型的 **35/35** 个顶层场景通过，其中 160-token 输入加 3-token 输出的请求在入口准备 **30,016 字节**、163 个队列位置，完成时 arena 实际消耗 **9,792 字节**、累计位置数 **163/163**。提交前的测试钩子在禁止 C++ 新分配时复制真实 `CaptureBatch`，检查其仍由固定 arena 持有。原有逐层 GPU/CPU 字节比较、Writing 退休、D2H/Mean-K 故障、未确认设备完成及 CPU 回收回归继续通过。准备失败夹具确认 arena 和队列均未发布。

固定 arena 的独立测试覆盖无新 C++ 分配的增长、复制与移动、所有权及容量耗尽。Windows 相关宿主 **14/14**、Linux 全部宿主 **27/27**，宿主 ASan/UBSan **3/3** 与 TSan **3/3** 通过；另有上轮 5 项 cell/CUDA 专项测试通过。Sanitizer 范围仅为宿主组件，不是 CUDA 适配器的 sanitizer 结论。

同版 recency CLI delete 完成 356-token 输入、8-token 生成并正常退出。arena 入口预留 **66,976 字节**，完成时实际用去 **21,872 字节**，捕获位置 **364/364**；`proposed=committed=reclaimed=6`、`live_observed=14`、`task_references=0`，policy/worker failure 为 0。结束时的 payload 字节数仍可能包含后台待归还的私有预约，不能解释为最终 RSS。普通 retrieval（未开启 historical GC）也正常生成 8 token，实际移动 4 个驻留块并完成 64-token replay；退出时仍有上一轮已确认的 backend compute buffer 实际 **15.9652 MiB**、预估 **12.2700 MiB** 差异。模型及两条 CLI 的退出码均为 0。六个代码/构建文件与远程测试副本的 SHA256 全部一致。原始数据与上述模型、宿主、sanitizer 日志保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`，文件前缀为 `historical-capture-arena-`，本地副本放在 `build-local/capture-session-validation/`。

这一容量只覆盖采集位置、原始范围以及可能被 D2H 持有的一份副本；整个 `llama_ubatch::data_t`、上游 `llama_batch_allocr`、每批 `slot_info`、`llama_kv_cache_context` 和 backend 计算图仍会分配。当前入口只用于已支持的单序列 recency/RAM 请求，尚未实现跨请求 arena 复用、完整请求严格预约、server 或长期元数据有界，P3–P6 的原定交付范围保持不变。

### 请求入口预约并复用真实 `slot_info` 存储（2026-10-17）

此前 `init_batch` 每次解码都会创建 `slot_info_vec_t`，`prepare_ubatches` 再为每个微批次创建槽位对象和 cell 索引向量。现在受限 historical 请求在 RAM/元数据接纳前，按 `min(请求 token 上限, n_batch)` 和 `n_ubatch` 准备实际的外层列表、stream 列表及各微批次 cell 索引容量。已接纳请求填充原对象；若微批次超过预约边界，明确关闭请求。普通非 historical 路径保持原行为。

上游补丁 `0009-historical-slot-info-return.patch` 为 `llama_kv_cache_context` 增加只在 `LLAMA_KVMEM` 下启用的槽位列表交还接口。适配器的解码上下文从请求取走这批缓冲区，在虚析构期间交回，下一次解码复用同一块存储；请求准备失败时，私有缓冲区不会被发布。Linux、Windows PowerShell 和 Windows HIP 补丁入口均纳入 `0009`。Linux 连续两次应用检查通过，第二次识别为已应用；PowerShell 构建脚本语法检查通过，未在本轮执行 Windows CUDA/HIP 全构建。

真实模型的 **35/35** 个顶层场景在最终版本通过。新增检查覆盖请求准备失败后的槽位容量归零、每次 `llama_decode` 后外层和嵌套缓冲区地址/容量不变，以及在禁止 C++ 新分配期间填满一个已准备槽位的索引向量。测试用的 `n_batch=64, n_ubatch=16` 在入口取得 **4** 个槽位对象、容器持有容量 **592 字节**；正常 CLI 的 `n_batch=128, n_ubatch=16` 取得 **8** 个、**1184 字节**。这些字节数只计相应容器容量，不含对象、分配器或其它模型工作区。实际 GC 仍回收 CPU historical payload，槽位列表属于前台执行元数据，并不改变 GPU 工作集回收协议。源码、测试及补丁的本地与远程测试副本 SHA256 一致。

最终版本的 recency CLI delete 完成 356-token 输入、8-token 生成并正常退出：`proposed=committed=reclaimed=6`、`task_references=0`、`gpu_deferred=gpu_returned=6`，policy/worker failure 均为 0。未启用 historical GC 的普通 retrieval CLI 同样正常退出，完成 `layout_writeback move=4` 和 64-token replay。模型及两条 CLI 的退出码均为 0；原始日志前缀为 `historical-slot-info-final-`，保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU` 及本地 `build-local/capture-session-validation/`。

这一步只消除了已接纳 historical 请求中 `slot_info` 列表的反复分配。上游 `llama_batch_allocr`、完整 `llama_ubatch::data_t`、每次创建的执行上下文对象、rollback 快照及 backend 计算图仍可能分配；统一资源预算、完整 active 入口和长期有界复用也未完成。因此 P3 仍在推进，P4–P6 的原定交付范围不变。

### 预备微批次数据并保留图缓存依赖（2026-10-17）

继续检查上游 `llama_batch_allocr` 后发现：`split_simple` 每次创建索引向量，`ubatch_add` 为每个微批次 `make_shared<data_t>` 并增长 token、position、序列和输出向量；适配器自己的 `ubatches` 外层向量也每次重新增长。受限 historical 请求现在按最多一个 batch 的微批次数量，在接纳前实际取得这些 `data_t`、其内部向量、索引暂存区和外层向量。上游补丁 `0010-historical-microbatch-pool.patch` 只在单序列、单位置轴、纯文本 simple split 下从此池取对象，其他路径沿用原实现；执行上下文析构时把外层列表归还请求。

第一次模型回归暴露了复用对象的序列列表没有清空，第二次暴露了图缓存仍持有旧 `data_t`。最终设计在每次填充前原地重建序列列表，并且只取引用计数证明空闲的数据对象，不覆盖图持有者。当前固定上游的 `llama_context` 至多保留 8 个宽度专用解码图结果和 2 个普通图结果，因此入口准备“本次最多微批次数 + 10”个数据对象；图缓存容量发生变化时必须同步审查此上界。容量不够或旧引用未释放时明确失败，不以运行中临时分配补位。

最终真实模型 **35/35** 个顶层场景通过。新增检查覆盖请求准备失败后不发布微批次容量、每次解码归还原外层和内部缓冲区、连续三次禁止 C++ 新分配地复用同一个上游 `data_t`，以及模拟 10 个图结果全部持有旧对象后，在禁止 C++ 新分配时完成一次 4 个微批次的完整拆分。模型请求 `n_batch=64, n_ubatch=16` 入口准备 **14** 个数据对象，内部向量与外层容器合计容量 **21,656 字节**；recency CLI `n_batch=128, n_ubatch=16` 准备 **18** 个、**28,168 字节**。这些只统计向量容量，不含 `data_t`/shared_ptr 对象、控制块、分配器开销和其它执行资源。

最终 recency CLI delete 正常完成 356-token 输入、8-token 生成和退出，`proposed=committed=reclaimed=6`、policy/worker failure 为 0；未启用 historical GC 的普通 retrieval CLI 也正常完成，执行 4 个驻留块布局移动和 64-token replay。模型及两条 CLI 的退出码均为 0。Linux 补丁连续应用两次通过，第二次识别为已应用；PowerShell 构建脚本语法检查通过，本轮未执行 Windows CUDA/HIP 全构建。源码、测试及补丁的本地与远程副本 SHA256 相同；原始日志以 `historical-ubatch-` 开头，保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU` 和本地 `build-local/capture-session-validation/`。

这项准备只覆盖 simple split 的索引、`data_t` 及外层列表。`llama_batch_allocr::init` 的输入规范化、`used` / `out_ids`、每次执行上下文对象、后端计算图和 rollback 等真实资源还没有全部纳入同一入口预算；不能据此声称 P3 严格请求预约完成。

### 请求前预留上游输入批次整理空间（2026-10-17）

继续审计发现，上游 `llama_batch_allocr::init` 在 memory 的 `init_batch` **之前**运行。即使先前已准备微批次数据，它仍可能自动生成位置、序列、logits，增长 `used`/输出索引，并为通用序列集合建立 `std::set`、`std::unordered_map` 节点。因此本轮把受限 historical 请求的入口改成接收实际 `llama_context *`，核对它拥有当前绑定的 KVMem memory，再在发布 capture owner 之前预留 `n_batch` 范围内的输入整理向量。准备失败会释放这些新缓冲区；重复 begin 不拆除已接纳请求的预留。普通非 historical 解码仍走原上游分支。

补丁 `0011-historical-input-workspace.patch` 为单序列、单位置轴、纯文本输入设置专用整理路径。该路径检查序列 0 的连续位置和 memory 起点，以标量维护最小/最大位置，跳过通用 set/map 的动态节点；超出已接纳的 token 数或输入形状会失败，不退回动态分配。接纳前预留位置、序列 ID/指针、输出标志、`used` 和 `out_ids` 的真实容量。上游 `llama_context` 只暴露准备/清理入口，适配器仍负责请求 owner 的发布与回滚。Linux、PowerShell 和 HIP 的补丁重放入口已加入 `0011`。

最终真实 GGUF/CUDA 模型 **35/35** 个场景通过，包含受阻 CPU 清理队列下的请求准备回滚。新增测试在禁止前台 C++ 新分配时重复调用 `llama_batch_allocr::init` 三次，并检查位置、输出计数和输出索引；准备拒绝时检查上下文未留在 historical 专用模式。recency CLI delete 完成 356-token 输入与 8-token 生成，后台 `proposed=committed=reclaimed=6`，策略/worker 错误均为 0；未启用 historical GC 的 retrieval CLI 正常完成 420-token prompt eval 与 8 次输出，`layout_writeback move=4`。三项程序退出码均为 0，远程隔离目录的原始日志依次为 `historical-input-final-test.log`、`historical-input-delete.log`、`historical-input-retrieval.log`。

补丁重放第一次检查发现 `0011` 改动了 `0010` 的上下文行，导致已装 `0011` 时单独反向检查 `0010` 误判。Linux、PowerShell 和 HIP 脚本现先识别 `0011`，把它作为 `0010` 已满足的凭证；Linux 连续两次重放和从仅有 `0010` 升级的路径都通过，PowerShell 构建脚本语法检查通过。本轮没有执行 Windows CUDA/HIP 全构建。这个里程碑只消除受限输入整理路径的已识别分配。每次解码创建的 memory execution context、backend 图/调度器、rollback 快照及其它 active 路径仍未纳入统一严格预算；P3 继续推进，P4–P6 不变。

### 请求前准备并复用解码上下文对象（2026-10-17）

在上述向量已可复用后，`init_batch` 每次仍以 `new HistoricalBatchContext` 分配一个多态解码上下文。现在受限请求在 `begin_historical_capture` 的私有准备阶段为这个对象取得一块实际存储，随其它资源一起发布；请求准备失败时本地 RAII 释放它。每次成功解码在同一地址原位构造上下文，并沿原有虚析构路径把槽位和微批次向量交还请求；对象析构不释放这块预留存储，直到 memory owner 退出才释放。重复/并行占用同一槽位会拒绝，符合当前单请求、单解码上下文限制。

已接纳请求若在 `init_batch` 中失败，现直接返回终止性错误，不创建一个临时错误上下文。这样也避免上游对 `FAILED_PREPARE` 进行重试时，在旧失败对象尚未析构前再次索取同一槽位。入口日志报告实际预留对象字节；最终真实模型本次为 **160 字节**，不包含对象内已分别预约的向量容量、分配器开销或 backend 图。

最终真实 GGUF/CUDA 测试 **35/35** 个顶层场景通过。测试检查入口失败后没有发布上下文存储，以及每次实际 `llama_decode` 后仍是同一地址、相同容量且不再被占用；受阻 CPU 清理队列下的请求回滚也通过。另在请求完成后的失败解码检查槽位没有泄漏。recency CLI delete 完成 356-token 输入和 8-token 生成，`proposed=committed=reclaimed=6`，策略与 worker 错误均为 0；普通 retrieval CLI 正常完成 420-token prompt eval 和 8 次输出，`layout_writeback move=4`。三项程序退出码均为 0，原始日志在远程隔离目录，前缀为 `historical-batch-context-`。当前只消除这一受限路径的上下文对象分配；backend 调度器/输出缓冲区、图结果、rollback 快照以及非受限 active 路径仍需请求前预算和长期有界验证。

### 后端输出缓冲区和图结果请求前预留（2026-10-17）

上游上下文创建时会预约计算调度器，但输出缓冲区通常只按最初的少量输出分配，第一次实际 decode 可能扩容；两组先前用于微批次所有权上界的图结果对象（8 个宽度专用 decode、2 个普通结果）也按需创建。补丁 `0012-historical-backend-reservation.patch` 扩展受限请求的上下文准备：在发布 capture owner 前确认调度器已预约，按允许的最大输出数量取得实际 Host 输出缓冲区，创建全部 10 个图结果及其元数据 arena。图结果的 reset 使用 `ggml_reset` 重用同一个 `ggml_context` 和元数据 arena，而非每次释放并重建。请求接纳后若参数变化要求重建调度器或增大输出缓冲区，解码会失败而不会临时扩容。准备被拒绝时撤销专用复用标志，已分配但由上下文持有的后端缓冲区留在这个可继续使用的上下文中。

真实模型最终 **35/35** 场景通过。模型 `n_batch=64` 的入口日志报告输出缓冲区 **38,895,616 字节**、10 个图结果元数据 **19,778,080 字节**；新增断言逐次比较调度器、输出 buffer、图结果对象及元数据 buffer 的地址/容量，均未改变。另创建独立真实模型上下文，在接纳请求后改变 causal-attention 参数；直接请求重建调度器被拒绝，原缓冲区保持不变，请求被标记为失败。受阻 CPU 清理 worker 的请求准备回滚仍通过。recency CLI 的 `n_batch=128` 入口预留输出缓冲区 **77,791,232 字节**；356-token 输入与 8-token 生成后 `proposed=committed=reclaimed=6`，policy/worker failure 为 0，GPU 延期/归还均为 6。普通 retrieval CLI 未启用 historical GC，完成 420-token prompt eval、8 次输出与 `layout_writeback move=4`。三项程序退出码均为 0。远程隔离目录中的最终原始日志分别为 `backend-reservation-final-model.log`、`backend-reservation-final-delete.log`、`backend-reservation-final-retrieval.log`。

Linux 补丁连续重放两次及从已装 `0011` 升级到 `0012` 均通过；PowerShell 构建脚本语法检查通过，Windows CUDA/HIP 全构建未执行。本地补丁、适配器与测试源码的 SHA256 与远程实际测试版本一致。

这是后端资源中已有明确上界的部分；实际构图仍可能为 input/fused-node 对象、rollback 快照、调度器内部结构或临时执行数据分配。本轮地址稳定测试不能证明已接纳请求完全无新分配，也不覆盖 server、多请求、MTP/hybrid 等组合。P3 严格预算仍未完成，P4–P6 原交付范围不变。

### 缩小每批图输入状态并实测前台分配（2026-10-17）

上轮只检查了后端缓冲区和图结果的地址是否稳定。本轮在隔离远程构建的测试程序中，临时统计真实模型每次 `llama_decode` 的前台 C++ `new`，并按调用点归类。受限历史请求的首个 48-token 批次有 **1815 次、6,660,844 字节**；随后 32-token 批次仍有 **1425 次、228,748 字节**。这证明此前的地址稳定检查不能代表整个解码无分配。首批中约 **6,397,248 字节**来自 CUDA 图的首次节点属性分配；稳定批次的两个大对象分别来自 KV 注意力输入和输出 ID 输入，每批各创建两次，却复制了完整的 `llama_hparams` 与 `llama_cparams`。

上游补丁 `0013-compact-graph-input-state.patch` 使 KV 注意力输入只持有后续 `set_input` 实际需要的 `causal_attn` 值，输出 ID 输入只持有输出数量；不借用上一批临时图参数的引用。对照构建在相同模型、相同请求下，32-token 批次降至 **1421 次、83,740 字节**，后续 48-token 批次从 **1336 次、229,796 字节**降至 **1332 次、84,788 字节**，即每批减少 **4 次、145,008 字节**的 C++ 分配。首次 48-token 批次仍有 **1811 次、6,515,836 字节**，主要受 CUDA 图首次准备影响。计数只覆盖调用 `llama_decode` 的前台线程、C++ `new`；不是总内存/RSS、CUDA 分配或完整请求预算的测量。

最终测试为图输入对象加入尺寸断言，真实 GGUF/CUDA 模型 **35/35** 项及 backend guard 通过。recency/RAM delete 实际推理完成 356-token 输入和 8-token 生成，`proposed=committed=reclaimed=6`、policy/worker failure 为 0；普通 retrieval 实际推理完成 420-token prompt eval、8-token 生成、64-token replay 和 `layout_writeback move=4`。Linux 补丁从 `0012` 升级到 `0013` 及重复应用均通过。测量与最终运行日志保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`，文件名分别为 `foreground-site-model.log`、`compact-graph-input-model.log`、`compact-graph-input-final-model.log`、`compact-graph-input-final-delete.log` 和 `compact-graph-input-final-retrieval.log`。

本轮再次确认长期有界问题：物理 repository/payload 槽位可按代复用，但生产 `KvMemStore::blocks_` 仍按递增 ID 存储，已预留的 `AppendReservation` 用完即拒绝后续追加，即使旧 payload 已被 GC 回收。因此 P4 要求的累计追加至少 10 倍容量及元数据循环复用仍未达到。其它图输入、fused-node、调度器/CUDA 图内部结构及 rollback 仍可分配；P3 严格接纳前预算也未完成。

### 后台 GC 扫描元数据固定容量复用（2026-10-17）

后台 manager 过去每轮为回收扫描、策略观察和决策输出重新创建 `std::vector`，完成一轮后还会销毁整个快照。现在 repository 提供接收调用方预留缓冲区的 `observe_range_into`；容量不足在改变输出前拒绝。GC manager 在构造时按物理仓库容量准备扫描、整轮快照和决策缓冲区，后续 tick 清空并复用它们，重试队列和代际去重数组仍维持原有固定上界。策略发生异常时也清除本轮决策，不保留未提交命令。外部原有 `observe_range`/`snapshot_range` 接口行为不变。

新增宿主回归用 **4 个物理槽位累计追加 40 个真实 RAM payload**，通过 LRU 提交退休、精确代际 release 和后续新代复用完成 10 倍容量循环；每个 `manual_tick` 均在禁止普通 C++ `new` 的测试区间内执行。结束时 40 次实际 release、仓库全部 Ready、payload 字节归零，累计过程中快照不超过 4 条、payload 字节不超过 4 个单槽用量，旧句柄不能重新读取。另一回归验证可复用扫描缓冲区不会遗留上一轮计数，容量不足时原输出保持不变。这只证明底层 repository/RAM/GC 链路，**不证明**生产 dense block 表或多请求入口已经通过 10 倍追加。

Linux 宿主 CTest **27/27**、相关 ASan/UBSan **2/2** 通过。TSan 首次受本机已知的 `unexpected memory mapping` 启动错误阻挡，按此前已验证的 `setarch x86_64 -R` 仅调整测试进程后相关 **2/2** 通过。真实 GGUF/CUDA 模型 **35/35** 项与 backend guard 通过；recency/RAM delete 推理完成 356-token 输入、8-token 输出，`proposed=committed=reclaimed=6`，policy/worker failure 为 0。原始日志保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`，前缀为 `bounded-worker-`。

### 逻辑身份与元数据表下标分离（2026-10-17）

此前预约追加要求 repository 的 `logical_id` 等于 `KvMemStore` 的稠密 `block_id`。即使物理 repository/RAM 槽位已经按代回收，后续新代仍无法沿用一个旧表位置而拥有独立身份。本轮给块记录和 shadow directory 绑定增加 64 位 `logical_id`，预约追加按 repository 元数据写入它，并要求跨批次严格递增；`block_id` 继续作为现有表下标。旧式追加及单块绑定仍沿用 `logical_id == block_id`，保持现有选择器和适配器行为。绑定有效性检查同时比较逻辑身份，避免表项被替换后误认旧代。

测试用表下标 0、1 搭配逻辑身份 1000、1001 成功登记，倒序身份 999 在发布前被拒绝且不消耗 writer 或表容量。隔离远程宿主 CTest **27/27**、相关 ASan/UBSan **3/3**、TSan **3/3**、真实 GGUF/CUDA 模型 **35/35** 及 backend guard 均通过。recency/RAM delete 实际推理完成 356-token 输入、8-token 输出，`proposed=committed=reclaimed=6`，policy/worker failure 为 0；原始日志以 `logical-id-` 开头，位于远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。

这是后续循环复用表项的身份准备，**尚未实现**生产 `KvMemStore::blocks_` 的槽位复用、长序列位置映射或跨请求接纳。当前完整生产请求的追加上界和单请求限制仍然存在，不能据此称 P4 的 10 倍容量长轨迹已经完成。

### 条件版本删除的原子提交（2026-10-17）

此前策略可以在删除意图中填写 `expected_content_version`，但 manager 对任何有版本条件的意图一律返回 `Reevaluate`，无法执行。现在 repository 的 `retire` 在同一物理槽位保护下核对代际、当前 Writing/Live 状态和内容版本，再用原有 CAS 关闭新引用入口；版本不符返回 `VersionChanged`，manager 将其报告为 `Reevaluate`，不退休该块。无条件 LRU 沿用原接口语义。当前一个 generation 的内容版本在进入 Writing 前发布，随后保持不变；将来若支持原地修改内容，必须先设计与该退休 CAS 同步的版本更新协议。

宿主 CTest **27/27**、相关 ASan/UBSan **2/2**、TSan **2/2** 通过。新增测试确认错误版本不删除仍可取得新执行许可，匹配版本才退休；manager 在两个结果下分别计入重评与已提交。真实 GGUF/CUDA 模型 **35/35** 项及 backend guard 通过；recency/RAM delete 推理完成 356-token 输入、8-token 输出，`proposed=committed=reclaimed=6`，policy/worker failure 为 0。原始日志以 `conditional-version-` 开头，保存在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。

这补齐的是策略意图的提交前置条件，不等于已经实现冗余判定算法、完整生产入口或长时有界元数据。

### 生产位置顺序索引与已回收跨度脱离（2026-10-17）

原 `KvMemStore::blocks_` 同时充当物理表与时间顺序，位置二分查找、sink/recent、recency/top-k、语义组和窗口打包均隐含 `block_id` 递增。一旦复用较小的表下标，新块会被误当成旧块。本轮增加单独的 `position_order_`，存放按原始位置排列的物理表下标；受限 historical 请求在入口按表容量预留它。追加和截断同步维护索引，位置查询及主要选择器按原始位置而不是下标运行，跨已删除空洞的媒体/语义组不会被当成完整组。

当 repository 已确认精确代际释放、producer 不再存在且 GPU/CPU/NVMe 槽位和 in-flight/working-set 权利均清零时，前台在下一次 active 边界把该跨度从位置索引移除；这一步不等待 GC，也不释放 CPU payload。旧表项暂留为不可选的墓碑，旧位置查询返回缺失。宿主回归验证中间块删除后的空洞、紧预算下 sink/recent 选择、跨空洞的媒体/语义组、后续追加、按原始位置打包以及跨墓碑截断。宿主 CTest **27/27**、相关 ASan/UBSan **3/3**、TSan **3/3**、真实 GGUF/CUDA 模型 **35/35** 项及 backend guard 通过。recency/RAM delete 推理完成 356-token 输入、8-token 输出，`proposed=committed=reclaimed=6`，trace 实际记录 **6 次**位置脱离，policy/worker failure 为 0。未开启 historical GC 的普通 retrieval 同样正常退出，完成 420-token prompt eval、8-token 输出和 `layout_writeback move=4`；原始日志前缀为 `position-order-`，位于远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`。

此时位置索引可稀疏，**物理块表和 shadow 目录仍未循环复用**；生产请求仍按整请求一次性预约并在用完后拒绝追加。下一步必须让墓碑在旧计划/设备/任务引用全部结清后获得新代，改造剩余按密集下标寻址的适配器与快照，并完成累计追加至少 10 倍物理容量的端到端验收。

### 已回收物理表项的显式复用（2026-10-17）

预约追加现在可以在某个块完成精确代际 CPU payload 释放、producer 引用消失、GPU/CPU/NVMe 槽和执行权利归零、原跨度从位置索引脱离之后，显式准备该表下标的新 producer。准备在接纳时完成可能分配内存的工作；后续追加先用尚未使用的表下标，再选已准备的墓碑，把新逻辑身份、repository handle 和原始位置覆盖到同一物理表项。shadow directory 的节点和 `KvMemStore` 的块表/位置索引均保持固定容量；旧 handle 继续由 repository 的 generation 校验拒绝，旧位置也不能解析为新代。

宿主回归使用 **2 个物理表项、2 个 repository/RAM 槽位累计追加 20 个真实 payload**，每次追加在禁止普通 C++ `new` 的区间执行，检查元数据容量恒定、旧句柄不可读取、旧位置缺失和最后 payload 全部归还。这是宿主组件级的 **10 倍循环**，不是 CLI 或 server 的生产长轨迹验收。当时旧 `truncate_to` 仍按物理表的稠密尾部删除，可能删错新代，因此暂时拒绝对已复用表的截断；后续改造见下节。生产 capture session 尚未自动调用表项准备，也没有跨请求 admission；不能把此项报告成完整 P4。

另有混合批次回归：先保留一个旧块，再同时追加一个新表下标和一个已回收表下标；第二个 writer 位置错误时整个批次不提交，修正后在禁止 `new` 的区间原子提交。最终远程宿主 CTest **27/27**、相关 ASan/UBSan **3/3**、TSan **3/3** 均通过。真实 GGUF/CUDA 模型的既有 **35/35** 场景和 backend guard 通过；recency/RAM delete 请求完成 356-token 输入、8-token 输出，本次后台 `proposed=committed=reclaimed=8`、位置脱离 8 次、策略与 worker 错误为 0。该 CLI 路径目前未自动启用表项复用，模型结果是基线回归，不是生产复用验收。原始日志在远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`，文件名前缀 `recycled-table-`。

### 旧选择计划与新代表项的隔离（2026-10-17）

重选计划原先只保存物理表下标。若计划准备后同一下标变成新代，旧计划的 stage-out、stage-in 或 RoPE 布局可能作用到新数据。本轮给块表增设身份/跨度修订号：追加、截断、确认释放后的位置脱离及墓碑复用均推进修订号。运行时给每个待执行计划单独编号；在 adapter 开始 GPU 搬运前，以及 runtime stage-out/stage-in 前，必须同时核对编号、表修订号及 pending 状态。尚未执行的计划不能被第二个计划静默替换，回收跨度脱离和表项新代准备也会在 pending/IO 未清时拒绝。旧计划失效后只能丢弃并重新规划，不能把同一个物理 ID 当成原 generation 继续执行。

宿主测试覆盖计划准备后表发生追加而拒绝执行、pending 期间不许脱离/准备复用、以及复用后选择器仍按原始位置排序；完整宿主 CTest **27/27**、相关 ASan/UBSan **4/4**、TSan **4/4** 通过。这个保护覆盖 runtime 的计划执行入口，不等于已经把 server checkpoint、跨请求缓存的裸表下标全部升级为精确代际引用；那些入口在启用生产复用前仍需逐项改造。

最终 CUDA 构建及真实 GGUF 测试 **35/35** 个场景和 backend guard 通过。相同模型的 recency/RAM delete 推理完成 356-token 输入、8-token 输出，后台 `proposed=committed=reclaimed=8`、位置索引脱离 8 次、策略及 worker 错误为 0；这个结果验证现有受限生产路径没有被计划检查误拒，仍不是跨请求表项复用验收。原始日志位于远程隔离目录，前缀 `plan-generation-final-`。

### 已复用物理表的安全后缀截断（2026-10-17）

旧截断从 `blocks_` 的物理尾部弹出，而复用后较小的表下标可能对应更新的原始位置。预约历史表现在改按 `position_order_` 找到要回滚的后缀：只有仓库确认该后缀每个精确 generation 的 CPU payload 已释放、producer 消失、GPU/CPU/NVMe 槽与执行/IO 权利均归零时，才把后缀从位置索引移除并将对应物理表项变成墓碑。块表和 shadow directory 不缩小，下一次接纳可以预备这些墓碑并用更高逻辑 ID 在回退后的位置再次追加。截断前验证整个后缀并预留返回列表，失败不改变块表；未执行的选择计划存在时，runtime 直接拒绝截断。

宿主用例从物理顺序 `1,2,0` 的混合新旧表项出发，确认活着的代际、pending 计划、仍占用的 GPU 槽各自阻止截断；两段后缀精确释放后，回退到位置 4 保持 3 个物理表项、保留位置 2 的旧块，旧位置和旧句柄不复活，再用表项 0 接纳位置 4 的新代。宿主 CTest **27/27**、相关 ASan/UBSan **4/4**、TSan **4/4** 通过。这是宿主预约表的回滚能力；当前 CLI adapter 仍主动拒绝 historical 请求中的截断、reset、会话续接和 rollback，相关执行资源及检查点协议未接入，不能称生产回滚完成。

最终 CUDA 构建和真实 GGUF 的 **35/35** 个模型场景及 backend guard 通过。recency/RAM delete 推理完成 356-token 输入、8-token 输出，本次 `proposed=committed=reclaimed=8`、位置脱离 8 次，策略/worker 错误为 0。该 CLI 不调用新增的截断路径，因此这里只证明既有推理未回归；原始日志在远程隔离目录，前缀 `recycled-truncate-`。

### 持久仓库域与跨请求 RAM 预约的宿主入口（2026-10-18）

原 `HistoricalKvCaptureSession` 同时拥有仓库、payload store、元数据预约和单次请求的 RAM 预约；请求对象析构就关闭仓库，后续请求无法使用 GC 归还的 Ready 槽位。本轮新增 `HistoricalKvCaptureDomain`：它先私下准备按物理容量固定大小的表，首个请求也可在表尚未安装时完成私有 RAM 预约；请求的 `admit` 在复核 runtime 后原子安装域，或由域直接 `admit` 供宿主调用。仓库、payload store 和表元数据存活到域结束。新请求由 `try_prepare_next` 只取得自己上界内的真实 Ready 槽位、payload、scratch、writer 回调和 CPU 清理队列票据。一个域同时只允许一个准备中或活动的请求。请求完成后，未使用的私有分配交给后台 CPU 清理 worker，已安装的历史块、仓库和表继续存活。请求失败则关闭域，避免部分安装后的继续接纳。

下一请求接纳前，前台仅在没有未执行计划/待完成 IO 且精确代际已经物理回收、producer 与 GPU/CPU/NVMe 使用权全部离开时，脱离旧位置并为墓碑预备新 producer。预约只计真实 Ready 槽位，不把 Retired/Releasing 当作未来信用；元数据槽不足时拒绝。请求准备与 `admit` 之间检查原始位置、物理块数和表修订号，防止使用已变化的运行时状态。新逻辑身份及 LRU 使用序列跨请求单调递增，物理表下标仍可重复。

新增宿主用例用 **2 个物理槽位连续完成 24 个有界请求**，累计追加 96 token，即 12 倍于这两个槽位的 8-token 容量。测试逐次完成真实 RAM payload 写入/发布、请求结束、精确 retire/reclaim、下一请求接纳，并在禁止前台 C++ 分配的区间验证实际追加，同时检查固定表容量、旧 handle 不复活、仓库在请求结束后保持开放。另用单槽位验证未执行计划阻止 `admit`，取消准备后 RAM/元数据归还，Live 槽位和仍有执行租约的 Retired 槽位不能作为接纳信用，完全回收后才允许新代写入。隔离远程宿主 CTest **27/27**，相关 ASan/UBSan **4/4** 和 TSan **4/4** 均通过；最终增补的零分配用例亦分别通过 ASan/UBSan 与 TSan 复验。小容量清理队列在旧请求私有析构尚未完成时会拒绝新预约；测试等后台清理完成后再继续，这符合非阻塞接纳的背压契约。

此轮先把 llama 适配器的**首个**受限请求接到持久域：域和请求 RAM 都在准备阶段保持私有，与现有 GPU、cell、图和输入 workspace 一起检查，最后由请求 `admit` 安装元数据；适配器在关闭时把域与 capture、runtime、GC 一同交给 CPU 清理对象，保证 worker 与销毁顺序。真实模型回归增加断言，确认失败准备不发布域，成功请求的域与 capture 指向同一仓库，`finish` 后仓库仍开放。最终 GGUF/CUDA **35/35** 场景和 backend guard 通过，原始日志为 `/tmp/kvmem-domain-adapter-final-model.log`。recency/RAM delete CLI 真实推理完成 356-token 输入、8-token 输出，`proposed=committed=reclaimed=8`、policy/worker failure 为 0，GPU 延期/归还各 8 次；日志 `/tmp/kvmem-domain-adapter-delete.log`。此外，CPU 清理队列固定 64 格；适配器域最多同时占用域元数据、请求私有 RAM、适配器销毁三个格位，因此最坏同时接纳量不超过 21 个域，耗尽时拒绝新请求而不等待。

截至首请求接线时，适配器仍拒绝第二次 `begin`、cached-turn continuation、reset/truncate，server 也未调用多请求入口；因此当时宿主级 12 倍轨迹还不是生产多请求 P4 验收。后续同一上下文的有界连续请求见下一节。已接纳请求的其余构图和后端资源仍未全部纳入 P3 预算。

### 同一真实模型上下文的连续请求与固定行元数据（2026-10-18）

适配器现在允许在前一有限请求成功 `finish` 后，使用相同 capacity 和同一个 GC manager 再次 `begin`。新请求沿原始位置继续追加，单独预约其输入、输出和追加次数上界。入口要求没有未执行计划、旧执行租约、采集队列、GPU source reader 或仍使用采集 arena 的对象；Ready 槽位或清理队列不足就返回拒绝，不在入口等待回收。请求失败后域仍关闭，不能以重新 `begin` 恢复半完成的请求。此入口尚未接到 server 的请求调度、缓存命中或分支恢复流程。

`row_positions_` 原来按累计 token 位置增长，现在在域准备阶段按 `capacity × block_tokens` 分配。通过位置索引取得当前物理表项，再定位该项内的行；表项换代后使用新代行数据，旧原始位置不再映射到新代。采集 arena 和 slot/microbatch workspace 同样按域容量一次准备，在所有持有 arena 的旧对象退出后才重置分配游标。`seq_pos_max` 使用域内累计原始位置，避免最后一个 GPU 驻留块退休后把下一请求的位置错误地回退。请求边界只清理已经退休且无设备使用者的 GPU 映射；CPU payload 的释放仍由后台 GC 完成，这不是让 GC worker 驱逐正在执行的 GPU 工作集。

已完成的旧请求对象交给固定容量的 CPU 清理队列销毁。修正了一个域所有权问题：旧请求析构过去会无条件清除 `request_claimed_`，从而可能取消新请求的独占标记；现在每个请求显式记录自己是否仍持有该标记，`finish`/`fail` 后不再改动下一请求的所有权。宿主回归特意在新请求接纳之后才销毁旧请求，再验证第三个重叠请求仍被拒绝。arena 测试则验证旧 owner 存活时拒绝重置，完全退出后在禁止 C++ 分配的区间复用同一地址。清理队列仍固定 64 格，但连续请求会额外暂存旧请求和未用预约；前节首请求的三格估算不能视作连续请求并发容量保证，实际接纳由所有未归还票据共同约束。

真实 GGUF/CUDA 长轨迹现在调用生产 `begin → decode → end_prefill → finish` 入口，始终保留同一个模型上下文和仓库域：

- 36 次请求，每次 64-token 输入，累计 **2,304 token / 72 块**，仅 **6 个物理槽位**，达到 token 容量的 **12 倍**；本次实际回收 70 块。
- 48 次请求交替使用 64/48-token 输入，分别生成 3/1 token；每次均预约最多 3 个输出，覆盖提前结束、短尾块和未用预约归还。累计 **2,784 token（含 96 个生成 token）/ 192 块**，仅 **8 个物理槽位**，超过 token 容量的 **10 倍**；本次实际回收 191 块。
- 每轮断言块表与 shadow directory 不超过固定槽数、行 buffer 地址和容量不变、采集 arena 地址不变、实际 payload 字节不超过物理容量，旧代退休句柄不能重新取得读取权。重叠 `begin` 和旧 arena owner 尚存时的 `begin` 必须拒绝。测试在两个已完成请求之间可重试接纳；请求入口本身和已接纳请求不依赖这种等待。

宿主 CTest **27/27**，涉及 arena、capture session 和 shadow directory 的 ASan/UBSan **3/3**、TSan **3/3** 通过。最终版本的真实模型 **35/35** 场景、两条长轨迹和 backend guard 全部通过，日志 `/tmp/kvmem-repeated-confirm-model.log`；两条长轨迹单独运行也通过，日志 `/tmp/kvmem-repeated-generation-model.log`。回收计数随后台执行时序变化：包含生成的轨迹单独运行回收 191 块，完整回归中回收 190 块，均满足至少释放 `192 - 8` 个旧块才能完成整条轨迹的断言。测试程序新增 `--repeated` 和 `--scenario 0..34`，便于单独重现连续请求及故障注入场景。

CLI 兼容性验收也通过：recency/RAM delete 以 capacity=32、Ready low=12/high=16 完成 **356-token 输入、8-token 输出**，`proposed=committed=reclaimed=8`，policy/worker failure 为 0，GPU source 延期与归还各 8 次，日志 `/tmp/kvmem-repeated-pressure-delete.log`。同样容量的默认 low=8/high=16 用例正常推理但没有触发回收（预约后的 Ready 恰为 8），该结果仅计作兼容性检查，日志 `/tmp/kvmem-repeated-final-delete.log`。未启用 historical GC 的普通 retrieval 完成 **420-token prompt eval（含 64-token replay）、8-token 输出**，日志 `/tmp/kvmem-repeated-final-retrieval.log`。所有模型验收均在远程隔离副本 `/tmp/kvmem-gc-99bef27.3CD9MU` 中进行；本地待提交的 10 个源码/测试文件与远程受测文件的 SHA-256 完全一致。

本轮首次完整模型回归曾在旧场景 11/35 的 CUDA stream 同步处报告一次非法访问，日志 `/tmp/kvmem-repeated-model.log`；后续相同场景单独运行和两次完整回归均通过，原因仍未确认。尝试 CUDA memcheck 时，远程 WSL/WDDM 调试接口初始化失败，不能据此声称 GPU 内存检查通过，诊断保留在 `/tmp/kvmem-repeated-stageout-memcheck.log`。这项异常仍需追踪，不能用成功重跑替代根因结论。

上述结果是**单 GPU、单 lane、文本、recency、RAM 的适配器连续请求**验收，证明所列存储和元数据可以跨请求复用；它不是 server 接入、全部 CPU/GPU 内存有界或完整 P3/P4 已完成的证明。其余构图/后端分配、reset/checkpoint、支持组合和效果对照仍在下列待办内。

### 新会话接纳、取消与旧 incarnation 后台退出（2026-10-18）

server 的普通会话流程会在缓存不匹配或请求取消时调用 clear/truncate/rollback；此前 historical capture 必须销毁整个 llama context 才能重置，不能直接接入这些流程。本轮新增明确的 `llama_kvmem_historical_capture_reset_begin` 入口：在同一个已准备的上下文中，先确认旧设备计算完成并结清普通采集任务，再私下准备新的 runtime、仓库域、真实 RAM 预约、active owner 和后台清理票据；全部准备成功才关闭旧 incarnation、清空 GPU cell 映射并发布从位置零开始的新请求。GPU buffer、后端图存储、固定行记录和采集 arena 继续复用。容量必须保持不变；申请过大、分配失败或清理队列无空间时拒绝，新域不会覆盖旧会话。发生执行/设备错误的上下文仍须 teardown，不能用 reset 擦掉错误。

这一入口同时允许取消**健康但尚未完成输入配额**的旧请求。旧请求的未用预约交给 CPU 清理队列；旧 GC 只收到停止请求，不在前台 join。旧 CPU 读者继续持有原 handle 和原 payload，新仓库使用不同 incarnation，因此旧句柄不能取得新会话的读取或执行权。旧 runtime、仓库、payload store、active 和 GC 保存在后台清理对象里，直到旧 GC 退出且所有 lease/预约都释放后才析构。新域默认 GC off，调用方须在首个 decode 前显式启动所选策略；server 参数和自动配置接线尚未完成。

新增 `HistoricalKvRepository::cleanup_ready`，不仅检查 operation 计数，还要求关闭后的内部 shared state 没有 lease 或 reservation owner。原因是 lease 会先递减 operation 计数，再放下自己的 shared_ptr；仅看计数归零可能使后台提前交出最后一个 owner，导致历史存储最终由前台析构。该判定仅在关闭、GC/元数据生产者停止后用于后台清理，不是策略的删除限制，也不把 CPU 读者变成“不能退休”的块。

宿主新增用例验证关闭后旧读者和未用 reservation 各自阻止最终清理，最后一个 owner 退出后才允许析构。宿主 CTest **27/27**、相关 repository/capture session/cleanup 的 ASan/UBSan **3/3** 和 TSan **3/3** 通过。真实模型专门用例已验证：64-token 输入只执行 48 token 后取消；超量/清理队列不足时旧请求仍可继续；旧 GC 暂停且旧读者存活期间，新域从位置零执行并完成；旧读者内容保持一致、旧句柄不能复活；最后一个旧 lease 的释放没有触发前台 C++ delete，旧域最终由后台销毁。初次模型日志 `/tmp/kvmem-reset-model.log`。测试程序新增 `--reset` 独立选择。

最终还增加了禁止 C++ 分配的准备失败用例和正常 `finish` 后再次重置的用例。server 构建成功；完整真实模型的 **35/35** 旧场景、新 reset 场景、两条连续请求长轨迹和 backend guard 均通过，日志 `/tmp/kvmem-reset-final-model.log`。本轮两条长轨迹分别回收 69/190 块，仍满足实际复用和固定容量断言。本地七个待提交源码/测试文件与隔离远程受测版本的 SHA-256 一致。本轮没有 HTTP 请求验收，不能将 server 编译成功报告为接口接线完成。

这是 server 接线所需的显式会话切换机制，**不是 server 已启用 historical GC**。普通 `llama_memory_clear`、缓存前缀截断、检查点恢复、多会话绑定和设备故障恢复仍须按各自协议接线；不会因新 reset 用例通过就把它们记成已完成。新旧域在切换期间各自持有实际资源，仍需纳入完整 P3 总预算；固定清理队列只保证待退出域数量受限，不替代总 bytes 预算。

### 单 lane server 的有限请求、HTTP 拒绝与健康取消（2026-10-18）

server 现可显式使用 `--kvmem-historical-capture N` 和 `--kvmem-historical-gc-mode off|shadow|delete`，low/high 的校验与 CLI 相同。开启此入口时要求 recency、单 lane/单会话、文本、flash attention、无 MTP/CPU spill/NVMe/session disk；不支持的组合在模型加载前拒绝。`/props` 返回容量、模式、水位和是否启用有限请求接纳。不加 capture 参数保留原路径。

原 server 默认把文本也交给带 checkpoint 的 multimodal 路径，会在缓存失配或取消时调用 historical 尚不支持的 clear/truncate/restore。本轮增加独立的 historical 请求流程：先按未缓存输入、有限输出上限和追加次数预约 RAM，成功后才能进入 decode。上一轮成功提交的 token 序列只有在新输入**完整包含它且有新后缀**时才继续原域；同样的 prompt、前缀回退、内容变化、显式 `cache_reset` 和取消后的新请求均使用 `reset_begin`，从位置零重新计算。这里是完整重算回退，不是已经实现检查点恢复。

预约在设置 SSE 响应之前执行，因此普通 JSON 和 SSE 请求都能在容量不足时返回 HTTP 503。拒绝不会替换旧 incarnation，原成功请求的缓存仍可供下一次精确延续使用。所有请求都按解析并限制后的 `max_tokens` 取得有限额度；HTTP `stream:true` 只是分批发送响应，**不等于**导师文档中的“依赖后续 GC 补给的分段预约”。后者仍未实现。

新增 `HistoricalKvCaptureSession::cancel` 和 `llama_kvmem_historical_capture_cancel`：在批次边界同步已提交的设备工作、结清采集任务，即使输入配额尚未用完也可安全结束健康请求，将未用预约交给已经预留的 CPU 清理任务；不等待 GC，也不在前台释放其 payload。已安装的历史仍归旧域所有，下一新会话再由 reset 的后台退出协议结清。取消不等于执行错误恢复；设备/执行失败仍关闭上下文，server 对后续请求明确返回需重启的错误。正常响应只有在 capture finish 成功后才发布可复用缓存；所有取消入口共用幂等清理，绕开旧 checkpoint 回滚。

宿主测试覆盖部分输入取消、在途 producer/lease 阻止过早完成、失败请求不能被 cancel 恢复，以及清理线程暂停时取消不分配/不释放前台 RAM、也不提前返还 Ready 信用。宿主 CTest **27/27**、repository/capture session/cleanup 的 ASan/UBSan **3/3**、TSan **3/3** 和 server 参数测试通过。真实模型的 `--reset` 场景新增显式取消及取消后新请求，和 backend guard 一起通过，日志 `/tmp/kvmem-http-reset-model.log`。

新增可重复运行的 `scripts/test_server_historical_gc.py`，只启动其自己的 loopback 服务并保存命令、响应、断言和 server 日志。delete 模式已验证 Chat Completions/Responses 的 JSON 与 SSE、首次请求、精确前缀复用、普通/SSE 预约拒绝、拒绝后复用旧缓存、显式重置、prefill 与生成阶段断线，以及后续新请求。实际 prefill 取消发生在 **385-token 输入只完成 64 token** 时，剩余 321 token；生成取消时已到 6 行、输入配额为零。四次压力请求每次完成 **389-token 输入、1-token 输出**，各有 `committed=14`、完成边界观察到 `reclaimed=13`，policy/worker failure 为 0。剩余释放可继续在后台执行，不能把完成边界计数当作要求前台等待全部 GC。

第一轮脚本使用默认日志级别，HTTP 行为通过但找不到底层完成/取消信息，最终日志断言失败；原因是 pinned llama 的 info 日志映射到 verbosity 4。验收命令现在显式带 `-lv 4`，随后日志断言通过。off、shadow、delete 三种模式的上述 HTTP 生命周期用例均已通过；shadow 四次压力请求各有 `shadow=14`、`committed=reclaimed=0`。原始证据分别保存在 `/tmp/kvmem-http-off/`、`/tmp/kvmem-http-shadow/` 和 `/tmp/kvmem-http-delete-final/`，其中 `command.json`、`results.json`、`server.log` 可复查。自定义纯文本模板用于固定 token 前缀与 grammar 输出，属于生命周期验证，不是模型回答质量评测。

另外使用脚本的 `--native-template` 模式，保留模型原始聊天模板，两轮正常对话及一次压力/reset 请求均通过，证据在 `/tmp/kvmem-http-native/`。本地 12 个源码/测试文件与隔离远程受测副本 SHA-256 一致，清单位于隔离副本的 `historical-http-source.sha256`。

最终完整真实模型回归也通过：原有 **35/35** 场景、新 reset/cancel 场景、36 次/48 次连续请求长轨迹及 backend guard。两条长轨迹本次分别回收 70/190 块，日志 `/tmp/kvmem-http-full-model.log`。本轮未重现前节记录的偶发 CUDA 非法访问，仍不能据此认定其根因已解决。验收原始日志、响应和命令已备份到本地 `build-local/historical-http-validation/`。

此项完成的是受限 server 请求闭环。模型采样器、其余图对象、缓存元数据和新旧域切换期间的总 bytes 仍未全部纳入 P3；当时 GPU/Host 依赖拆分尚未实现，后续进展见下节。checkpoint、其它后端组合及策略效果比较也未完成。真实设备错误目前需要重启 server，而不是自动重建 context。不能将这些 HTTP 测试报告为完整生产交付或全内存有界。

### RAM Host payload 与 GPU 执行身份分离（2026-10-18）

此前 repository 只有一个整块引用计数：即使旧 active 只读已独立驻留的 GPU 副本，也会把 CPU payload 一直保留到 active 结束。本轮把许可区分为需要 Host 的 writer/read/transfer/普通 execution，以及只保留身份的 GPU execution。默认 execution 仍保留后续可能用到的 Host 数据；只有调用方能证明整个操作及其未来步骤均只依赖独立 GPU 副本时，才能使用 `acquire_gpu` 或显式 GPU-ready Writing 许可。按句柄存储拒绝使用 GPU-only 许可读取 K/V 或 Mean-K，退休后也不能升级为新 Host 许可。

阶段、Host 引用和身份/GPU 引用编码在同一个原子 control word 中；获得执行/读取许可与登记 Host 依赖是同一次 CAS，因此不会出现“已获得许可，退休却看不到未来 Host 依赖”的窗口。重复的 active 请求按最强依赖合并：同一块即使先出现 GPU-only 请求，后续 Host 请求也不能被去重过程丢掉。受限 recency 适配器对已完整驻留 GPU 的旧历史使用 GPU-only 许可；当前写入跨度仍由 producer 持有 Host 引用。

退休仍立即关闭下一次选择。最后一个 Host 依赖结束后，后台可以调用精确代际的释放回调，销毁 `HistoricalKvRawStore` 中该块的 RAM K/V/Mean-K bundle；此时旧 GPU execution 与元数据观察仍可保留身份。新增 `HostReleased` 结果和 `host_released_early` 计数，只记录部分资源释放。该块仍是 Retired，**不是 Ready**，不返还可预约槽位，也不撤销 pending slot credit。所有身份/GPU 引用退出且 Host 回调成功后，才进入完整 Releasing/Ready 并允许新代复用；失败回调仍按原句柄重试。GC 不修改 GPU 映射、mask 或工作集。

这次拆分覆盖的是 RAM Host bundle 与 GPU/身份之间的依赖。K/V 与 Mean-K 仍作为同一 Host bundle 释放，NVMe allocation/IO 和 pinned cache 尚未独立接入。旧位置式 `RawKvStore` 桥接保留整块等待条件，不能把它报告成已支持提前释放；兼容回归专门检查仅剩 GPU-only 引用时它仍保留实际 CPU payload，最后引用退出才释放。

新增宿主回归覆盖 Host read/transfer 延迟释放、GPU-only 许可禁止读 Host、回调重试、重复请求保留 Host 依赖、Host admission 与退休并发，以及部分释放后的 Ready/pending 守恒。真实 GPU 用例在已驻留的 K 张量上提交单独的 CUDA 读取操作，用测试屏障保持其完成事件未就绪；后台释放 CPU payload 时，检查旧 active/身份和 GPU 槽位仍在、槽位不能复用，随后解除屏障并逐字节验证读取结果。这个证据证明**真实 GPU 消费者未完成时 Host 可先释放**，不声称原 attention kernel 仍在运行，也没有将该等待屏障加入生产路径。

完整真实模型回归已通过原有 **35/35** 场景、reset/cancel、36 次和 48 次连续请求及 backend guard；两条长轨迹分别回收 70/191 块。原始日志为 `/tmp/kvmem-resource-split-full-model.log`，单场景 GPU 证据为 `/tmp/kvmem-resource-split-gpu.log`。此前记录的偶发 CUDA 非法访问本轮未重现，根因仍未查明。

最终宿主 CTest **27/27**，相关 ASan/UBSan **7/7**、TSan **7/7** 通过（六个核心组件加一个 legacy 桥接组件；sanitizer 不覆盖 CUDA adapter）。delete 模式真实 HTTP 回归 **33/33** 断言通过，包括 JSON/SSE、精确前缀延续、容量拒绝、reset、prefill/生成取消及后续压力请求。四次压力请求均 `committed=14`、完成边界 `reclaimed=13`，policy/worker failure 为零。本轮这些普通 HTTP 请求的 `host_released_early=0`；提前释放的受控 GPU 用例已证明可行，但没有据此声称自然请求的峰值内存或延迟已经改善。HTTP 证据位于 `/tmp/kvmem-resource-split-http-delete/`，本轮原始日志与响应备份到本地 `build-local/historical-resource-split-validation/`。

### Packed K/V、Mean-K 与存储记录的独立回收（2026-10-18）

上一节把 Host bundle 与 GPU/身份分离，但一个只需要 Mean-K 的评分 reader 仍会留住全部 packed K/V。本轮增加 `PackedKvRead`、`PackedKvTransfer` 和 `MeanKRead` 许可，分别对应只读 K/V、搬运 K/V 和读取评分均值；原有普通 read/transfer/execution/writer 保留完整 Host 权利。退休关闭所有新的资源申请，窄许可不能读另一类资源，也不能在退休后升级。阶段和四类引用计数仍在同一原子 control word 中提交，每类计数上限 32767，饱和时明确拒绝，避免溢出到其它计数或阶段位。

新增 `ResourceRelease` 回调协议，按 packed K/V、Mean-K、最后的存储 metadata 分别确认完成。一个资源的失败不妨碍其它已满足条件的资源尝试释放；成功分量不重复执行，失败分量以原句柄幂等重试。实际存储每层交换出待释放向量，在槽锁外析构，之后才扣除真实容量；`allocated_payload_bytes` 反映剩余向量容量。K/V 和 Mean-K 都释放后，只要旧 GPU/身份引用仍在，存储记录的 allocation 信用继续占用，repository 也不会变成 Ready。metadata 仅在全部旧引用结束后释放；完成确认丢失时同样可以重试。

生产 capture session 在请求接纳时预备分资源回调及实际 K/V/Mean-K bytes，追加时只移动既有回调与 payload。目录增加窄许可入口，适配器 `score_retrieval` 使用 Mean-K 许可，stage-in 使用 packed K/V transfer 许可并只复制 K/V，不再顺带复制 Mean-K。legacy `RawKvStore` 的整块回调保留旧等待条件。新增 `partial_resource_releases` 统计记录 Host bundle 尚未全部释放时的进展；该值不作为 Ready 容量或可用预约信用。

宿主新增两种对称情形：保留评分 reader 时释放 K/V，保留 K/V transfer 时释放 Mean-K；读取线程与回收线程并发，逐层校验仍被保留的数据。测试还覆盖资源或 metadata 已物理释放但确认丢失后的重试、旧代 callback 不影响新代、禁止普通 C++ 分配时回收、32767 个引用的计数边界、退休与三类 Host 许可并发，以及 manager 的 Ready/pending 守恒。首次构建暴露旧 capture 测试 hook 仍接受整块回调签名，已改为仅在 packed K/V 释放前设置原有屏障，没有改变生产取消/完成语义。

宿主 CTest **27/27**、相关 ASan/UBSan **8/8** 和 TSan **8/8** 通过；补充的 metadata 确认丢失用例在三套构建中分别再通过。真实模型的多代 CUDA capture 用例改为分资源回调，在 **48 个注意力层**逐层验证：保留 Mean-K 时剩余 **98,304 bytes**，保留 packed K/V 时剩余 **3,145,728 bytes**，其余 Host payload 已释放，保留数据和 GPU 副本字节不变。这些是实际 payload 容量，不是 RSS。先前“GPU 事件未完成时 Host 可先释放”的用例也继续通过。

最终完整模型 **35/35**、reset/cancel、36 次/48 次连续请求和 backend guard 通过；两条长轨迹本轮分别回收 69/191 块。最终日志为 `/tmp/kvmem-mean-split-model-final.log`；早期基线为 `/tmp/kvmem-mean-split-model-initial.log`。本轮未重现先前的偶发 CUDA 非法访问，其根因仍未解决。

受限 server 的 delete HTTP 回归 **33/33** 通过，证据在 `/tmp/kvmem-mean-split-http-delete/`。四次压力请求各 `committed=14`、完成边界 `reclaimed=13`，policy/worker failure 为零；prefill 取消时剩余 321/385 个输入 token，生成取消时输入已完成。普通 HTTP 用例没有触发组件单独释放，`partial_resource_releases=0`，不据此声明实际请求内存峰值或速度改善。命令、响应、构建与 sanitizer 日志备份到本地 `build-local/historical-mean-split-validation/`。

**当时的边界：** 写入 producer 及其任务仍持完整 Host writer；后续分量结算见下节。K 与 V 在 packed K/V 内仍共同释放。NVMe、pinned cache、MTP 也未因此获得支持。虽然 scorer/stage-in 的接口已使用窄许可，自动 historical capture 的公开入口仍限制为 recency/RAM，不能把接口测试报告为完整 retrieval 生产接入或吞吐改善。

### 生产 writer 按已完成分量归还 Host 权利（2026-10-18）

多个异步任务仍共享同一个 producer，但分资源 writer 现在在建立时分别取得 packed K/V 权利、Mean-K 权利和独立身份引用。`HistoricalKvRawStore::publish_complete` 在内容锁内检查整个固定跨度、全部必需注意力层的进度：某分量全部写满后，就不可逆地关闭该分量的后续 writer 访问，再归还其资源引用。完成位幂等，重复结算不会重复扣引用；原来的整块 writer/legacy bridge 继续保留原语义。现有 packed、异步/同步 Mean-K 和 decode 提交流程本来就调用该接口，因此这一结算已接到实际生产任务。

完整块尚未发布、或已被策略退休，都不妨碍已完成分量归还权利。后续任务继续使用捕获的原身份，`write_progress` 保留逐层进度，便于已经获准的重试识别已写内容；它不授予 payload 访问。写接口在内容锁内检查对应权利，已关闭的分量拒绝迟到写入。读取接口也在拿到内容锁后再次检查动态 writer 权利，避免初次权限检查与实际读取之间发生分量关闭；普通 reader/transfer 许可仍保持其原有、不可缩减的资源依赖。

某分量没有写满时不会提前归还权利。如果请求取消或任务失败，最后一个 producer 退出时归还尚未结算的权利，退休资源仍能最终释放，不会因短尾或未完成写入永久滞留。即使两份 Host payload 都已释放，producer 的身份引用仍保护进度、发布状态和存储记录；最后一个任务退出后，metadata 才可释放并返还 Ready。GC 不修改 GPU 工作集，也不会把旧完成回调转向新 incarnation。

新增宿主并发测试使用两个必需层和一个非注意力层，分别让 packed K/V、Mean-K 先完成；真实回收线程持续运行，另一分量继续写入，并发查询 writer 权利及读取均值。测试检查单层完成不能释放整个分量、保留数据与进度正确、关闭后拒绝读写、禁止普通 C++ 分配时回收、重复结算、identity-only writer 的移动，以及最后一个任务退出后的 Ready 守恒。宿主 **27/27**、相关 ASan/UBSan **8/8**、TSan **8/8** 已通过。

真实 CUDA capture 用例验证：48 层 packed K/V 已全部提交、Mean-K capture 尚未结清时，后台释放 **3,145,728 bytes** K/V，仅保留 **98,304 bytes** Mean-K 和原身份；之后迟到的 Mean-K 任务继续完成，退休状态不会恢复成 Live。首次新增断言把 decode fixture 的 8/32 个已接受行误当作完整 K/V，导致测试失败；修正后明确要求该未完成分量继续受保护，直到任务退出。本轮没有为了通过测试而放开未完成写入的资源。

最终源码的完整真实模型回归 **35/35**、reset/cancel、36 次和 48 次连续请求及 backend guard 全部通过；两条长轨迹分别回收 **69/190** 块，日志 `/tmp/kvmem-writer-split-model-checked.log`。delete HTTP 回归 **33/33** 通过，四次压力请求均 `committed=14`、完成边界 `reclaimed=13`，policy/worker failure 为零；本轮这些请求的 `partial_resource_releases` 分别为 **10/12/12/12**，说明普通请求中也实际发生了分量先释放。该计数不是释放字节数，也不能单独证明峰值内存或吞吐改善。prefill 取消时已接收 64 行、剩余 321 个输入 token，生成取消时输入已完成。先前记录的偶发 CUDA 非法访问本轮未重现，根因仍未确认。构建、sanitizer、模型和 HTTP 原始日志以及命令备份在本地 `build-local/historical-writer-split-validation/`；最终源码与隔离受测副本逐文件核对 SHA-256。

本轮仍以 packed K/V 为一个资源组，并保留完成检查和存储内容同步；这不是每条 DMA job 都各自分配新 writer，也不是完整前台无等待或全资源预算证明。NVMe、pinned cache、其它 active 入口、检查点/多会话及策略效果验收仍需继续完成。

### NVMe allocation 代际身份与真实 IO 生命周期（2026-10-18）

原 `NvmeKvTier` 只按 block ID / 物理槽位定位，`RawKvStore::IoJob` 直到 worker 执行才分配磁盘位置，runtime 预取也在线程开始后重新查 block ID。本轮给每个 arena 分配独立身份，每个固定物理槽位保存不回绕的 generation；placement 返回 `NvmeAllocationHandle {arena, generation, slot}`。`NvmeIoLease` 在同一个原子 phase/refcount 中取得使用权，最多尝试 16 次，取得后复核 generation；旧句柄和其它 arena 的句柄不能授权 IO。它保护 allocation 生命周期，不代替内容版本、读写顺序或 tier 对象自身的存活保证。

新增 `try_release_allocation(handle)`：CPU worker 尝试取得元数据锁，忙时返回 Retry；关闭后拒绝新 IO，已有读写继续完成，未退出时返回 Deferred，最后归还物理槽位并返回 Released。重复确认返回 AlreadyReleased，已复用旧代返回 Stale，不重复返还容量。未完成的 allocation 不作为 free slot；legacy LRU 也不能驱逐有 IO 的槽位。按 block ID 的 `release_block` 只允许已排空的同步 owner 使用，`clear` 先关闭各 allocation，若仍有 IO 则拒绝复用并要求稍后重试。归还的是文件内 allocator 容量，不承诺文件大小或磁盘占用立即缩小。

真实异步写回现在在入队前分配并捕获 IO lease，worker 使用原 allocation 批量写，不重新解析 block ID；IO 结束并处理完成状态后，先归还 lease，再发布 `inflight=0`，避免 `wait_writes/clear` 先行。runtime 预取也在提交线程前捕获 allocation。磁盘无槽位时，写回在入队前拒绝，RAM 原数据保留；新增单槽磁盘用例验证 K 已写入、V 无空间时仍能读取两者，clear 后可再写。legacy raw store 仍使用位置式 metadata 和已有错误传播协议，因此 historical GC bridge 继续拒绝 NVMe，不能据此认为新历史域已经支持磁盘回收。

自查发现旧的同名临时文件 `open → unlink` 有并发窗口：两个 arena 可能打开同一个 inode。临时主文件及 overlay 改为 `mkstemp` 创建唯一文件后再 unlink，归档主文件保持原路径。测试同时建立 16 个同名 arena 及 8 个共享只读归档的独立 overlay，逐个读回不同数据；overlay allocation 释放再绑定后不继承旧 overlay 字节，原始归档保持不变。read-only arena 必须显式使用 durable 归档模式。

真实文件测试还覆盖分段写到一半时关闭 allocation、另一 allocation 独立释放、旧任务继续写完并按字节读回、旧/跨 arena 句柄失效、clear 与 IO 重叠，以及 **1 个物理槽位连续 64 代**读取/关闭/复用。Linux `KVMEM_ENABLE_NVME=ON` 宿主 **27/27**、相关 ASan/UBSan **5/5**、TSan **5/5** 通过，独立 OFF 构建 **27/27** 通过。首次测试因其它用户拥有固定 `/tmp` 测试目录而拒绝访问；改用各构建独立 TMPDIR 后通过，没有更改这些既有目录。

Windows OFF 构建成功，23 个测试分项复验全部通过。最初沙箱临时目录拒绝文件操作；改用项目专用临时目录后 snapshot/transfer 通过，生命周期测试在允许其创建测试链接和管理子进程的环境复验。另修复测试脚本将系统本地编码错误信息一律按 UTF-8 严格解码的问题：现在保留非 UTF-8 字节的转义表示，错误路径和失败重试断言保持不变；Windows 及 Linux 生命周期测试随后通过。

隔离 CUDA 构建已开启 `KVMEM_ENABLE_NVME=ON`。最终真实模型回归 **35/35**、reset/cancel、36 次/48 次连续请求及 backend guard 通过；两条长轨迹回收 **69/190** 块。实际 retrieval CLI 使用 128 MiB NVMe、raw-K/V 磁盘写回，完成 **420-token prompt eval（含 64-token replay）及 8-token 输出**。delete HTTP 回归 **33/33** 通过，四次压力请求均 `committed=14`、完成边界 `reclaimed=13`，policy/worker failure 为零；分量释放事件为 12/12/10/12。HTTP 仍运行受限 recency/RAM 路径，这一结果验证开启 NVMe 编译能力后既有 GC 行为未回归，不代表 HTTP historical GC 已开始回收磁盘。此前偶发 CUDA 非法访问本轮未重现，根因仍待确认。

额外用相同 CLI 参数开启已有 `KVMEM_PERF=1` 计数器，采集摘要实际记录 **1,441,792 bytes / 3 次 NVMe syscall**，证明模型路径发生了磁盘写回；这个异步采集区间的计数不作为整次请求的完整 IO 总量或性能改善结论。最终受测源码逐文件 SHA-256 与本地一致，构建、失败记录、sanitizer 详细输出、模型和 HTTP 响应及完整命令备份在 `build-local/historical-nvme-allocation-validation/`。

这一层完成的是磁盘分配与旧 IO 的安全隔离。它仍需与 historical handle 的资源绑定、内容版本、请求预算和完成回调整合；pinned cache 的失效与其它 active 路径尚未完成。元数据 map/LRU 和 legacy 查询仍使用锁，不能把 worker 的 try_lock 或 IO lease 的有界 CAS 当作完整前台无 GC 等待证明。

### Historical handle 与独立 NVMe 副本绑定（2026-10-18）

在 allocation generation/IO lease 底座上，`HistoricalKvRawStore` 现在可选配私有 NVMe arena。每份历史 payload 在请求预约阶段同时取得完整 RAM 容量、真实文件内 extent、原始 producer IO 权利；historical handle 与该 extent 的 arena/slot/generation 精确绑定。磁盘位置不按原始 token 位置计算，也不由 legacy LRU 隐式驱逐。短尾使用完整预留 extent，hybrid 布局按物理层号寻址，非注意力层的空洞也计入容量。新配置只用于独立 historical store，不改变 legacy runtime 的 NVMe 配置或公开 CLI/server 支持范围。

磁盘文件在域创建时 `posix_fallocate`，historical 模式不接受不支持预分配时静默退回稀疏文件；真实文件空间不足会在创建时失败。请求需要的 extent 不足时预约立即失败，全部退回尚未发布的 RAM、磁盘和 repository slot；不等待 GC，也不把待释放资源当 Ready。`reserved_nvme_bytes` 表示请求已经取得的完整 extent 容量，`allocated_nvme_bytes` 包含私有预约与尚未物理归还的已安装 extent；这是文件内 allocator 字节数，不包括文件系统块取整或文件系统元数据。预分配文件在进程运行期间不因单块回收缩小。

现有 packed K/V 写接口先使用原 producer lease 写入该代磁盘位置，成功后再更新 RAM 内容与进度。所有必需层的 packed K/V 完成时独立关闭 producer 的磁盘写入权；Mean-K 未完成仍可继续。若半写取消，最后一个 writer 退出自动归还未完成的 IO 权利。回收可以在安装 payload 前退休 writer，但此时不能把尚未安装的磁盘资源确认成 Absent；旧 writer 仍可按原预约安装、完成已获准的工作。失败的磁盘写入不推进完成前缀，不能发布为 Live。

repository 新增独立 NVMe 资源确认：退休后关闭新 IO，已有 IO 使用自己的 lease 继续完成。Host reader 不会无故阻挡磁盘释放；磁盘 reader 也不保留已经不需要的 CPU payload。只有 Host、NVMe 和身份引用全部结清，metadata 才能释放并返还完整 Ready 槽位。新 `NvmeRead` 同时持精确 historical 身份和 allocation IO 权利，只能在 Live 时取得；退休后旧读可完成，新读被拒绝。物理释放与确认重试分别记账，旧确认或旧 metadata 析构不能关闭已复用的新 allocation。

同时修补 NVMe 构造失败清理：临时文件在预分配前即 unlink，后续初始化异常关闭已经打开的描述符。新增真实文件测试覆盖 `fallocate` 拒绝、`pwrite` 部分写入后失败及恢复重试、半写取消、退休先于安装、CPU 与磁盘分别先释放、后台回收与旧读重叠、移动 reader、确认丢失后复用以及跨请求预约退回。一个 extent 连续复用 **32 代**；同一 capture domain 在 **2 个物理槽位上完成 24 次请求**，append 和已准备数据的 write 在禁止普通 C++ 分配期间执行。

Linux NVMe ON 全套 **28/28**、OFF **27/27**、相关 ASan/UBSan **11/11**、TSan **11/11** 通过；额外补充的真实部分写失败用例也分别在普通/ASan/TSan 构建复验。最终 Windows OFF 回归 **23/23** 通过。

真实 CUDA 新增场景 35 在同一个磁盘 extent 上运行 5 代实际 capture，覆盖异步 packed K/V、异步及同步 Mean-K、decode 累加、写入中退休和半写取消。3 轮已发布数据分别在 CPU payload 释放后，使用退休前取得的磁盘 reader 逐层核对 **48 层 / 3,145,728 bytes**；其余检查确认 GPU 数据保持不变、旧 producer 能结清，以及全部 CPU/磁盘容量最终归还。完整模型回归 **36/36**、reset/cancel、36 次和 48 次连续请求及 backend guard 均通过；两条长轨迹回收 **70/190** 块。这些连续请求仍是现有 RAM 路径，不能将其统计当作磁盘卸载效果。

服务器 delete HTTP 回归 **33/33** 通过；四次压力请求均 `committed=14`、完成边界 `reclaimed=13`、`partial_resource_releases=12`，policy/worker failure 为零。HTTP 仍验证受限 recency/RAM 入口。本轮没有重现此前偶发的 CUDA 非法访问，根因仍未确认。原始构建、sanitizer、模型、HTTP 证据及命令保存在 `build-local/historical-nvme-binding-validation/`；源码与远程隔离受测副本逐文件核对 SHA-256。

**当时的阶段边界：** 独立磁盘副本保留完整 RAM payload，并未按压力将 Live CPU payload 卸载到磁盘。当时新磁盘读取接口尚未替代实际 stage-in 来源；本项后续接线见下一节。公开自动 capture 仍限制 recency/RAM。磁盘写入仍在既有 store 内容锁范围内完成，Host 释放也保留内容同步，不能据此宣称整个生产前台无 GC 等待。

### Stage-in 的精确存储来源与设备完成确认（2026-10-18）

上一阶段已经可以从独立 NVMe 副本读取历史数据，但实际恢复 GPU K/V 的 `write_block_to_gpu` 仍只读取 RAM。本轮增加 `StageInAccess`：在搬运开始前一次取得确定的存储来源。绑定独立磁盘副本时持 historical 身份与该 allocation 的 IO 权利；RAM 路径持 packed K/V transfer 权利。来源拒绝后直接失败，不改查位置式旧存储，也不临时切换到另一个资源。许可取得后的退休关闭新 stage-in，已经承诺的旧搬运可以逐层完成。

磁盘来源不持 CPU payload 引用，因此旧磁盘搬运仍在进行时，后台可先释放 RAM K/V 和 Mean-K；IO、身份与 metadata 继续保留到该搬运结束。RAM 来源则保留实际需要的 packed K/V。新增复制到已有 buffer 的接口，避免每层创建 `LayerSnapshot` 向量；受限公开请求在接纳前准备并初始化整块 K/V staging buffers，记录 `KVMEM_STAGEIN_WORKSPACE host_bytes`。这些是 Host 暂存容量，不是 pinned cache，也不证明所有生产入口已经完成统一预约。

单 GPU 的受管理历史搬运改用检查提交结果和 stream 完成状态的 H2D 路径。传输错误后关闭后续搬运，不执行旧的失败 fallback。源许可和 Host buffer 由 adapter 成员持有，异常栈退出不释放；如果尚不能确认设备完成，重试恢复、clear 和 cleanup 都不能覆盖或释放它们。只有 checked drain 成功后才释放源许可。最终设备清理持续失败时沿用现有中止进程的处理，不假装安全复用。

宿主测试覆盖 RAM 和 NVMe 来源在退休前取得许可、退休后旧读完成及新读拒绝。磁盘 capture 域的 24 次请求循环还验证 CPU payload 先释放、旧来源不分配新 C++ buffer 即可读取、最后归还精确 extent。Linux NVMe ON **28/28**、OFF **27/27**、相关 ASan/UBSan **9/9**、TSan **9/9**、Windows OFF **23/23** 已通过；sanitizer 不覆盖 CUDA adapter。

真实 GPU 新增四个场景：退休后从磁盘恢复三 token 短块并核对 **48 层**及未写范围；RAM 完成等待失败；NVMe 实际提交后报告失败；NVMe 等待失败且首次取消无法确认完成。首次完整验收通过原有 36 个场景，但新增测试漏建 cache row 元数据，恢复在进入实际拷贝前失败；已改用正常 `prepare_ubatches` 初始化。修正后完整模型 **40/40**、reset/cancel、36 次/48 次连续请求及 backend guard 全部通过，两条长轨迹分别回收 **69/191** 块。最终日志为 `/tmp/kvmem-stagein-model-fixture-fixed.log`；首次失败及单场景诊断日志也保留。故障注入发生在真实 H2D 提交之后，但不是自然硬件故障复现；连续请求仍走现有 RAM 路径。

真实 server delete HTTP 回归 **33/33** 通过，覆盖 JSON/SSE、精确前缀延续、容量拒绝、reset、prefill/生成取消和压力请求。四次压力请求均 `committed=14`、完成边界 `reclaimed=13`、`partial_resource_releases=12`，policy/worker failure 为零。源码与受测隔离副本逐文件 SHA-256 相符；构建、sanitizer、首次失败、最终模型、HTTP 日志与响应保存在 `build-local/historical-stagein-validation/`。

**阶段边界：** 本项接通受管理来源到实际 GPU restore 的路径，仍保留完整 RAM 加磁盘副本。公开自动 capture 仍是受限 recency/RAM，未开放磁盘配置或 Live CPU 卸载。多 GPU、通用批量异步 stage-in、pinned cache 槽位失效、全部 active 入口、统一预算及无 GC 等待证明仍需独立完成。此前偶发 CUDA 非法访问本轮未重现，根因仍未确认。

### CPU cache 槽位的代际与搬运保护（2026-10-18）

此前 `PinnedKvTier` 只有 block ID 到整数槽位的映射，LRU 替换、release 和 clear 不检查是否仍有搬运使用原槽。本轮将 NVMe 已使用的 allocation 状态机提取为带资源类型的 `ResourceAllocationRegistry<Tag>`；NVMe 与 pinned 的 handle/lease 类型不同，不能互相授权。每次分配带独立 arena ID、slot 和单调 generation，取得/归还 IO 权利与关闭新 IO 使用同一个原子 control word。此状态机只保护分配生命周期，内容版本和读写顺序仍由存储 owner 保证。

`PinnedKvTier` 增加精确分配和释放入口。GC 的 try-release 关闭新访问后，即使返回 Deferred，也不返还槽位；旧 IO 退出后按原 generation 确认释放。重复确认幂等，复用后旧确认返回 Stale。clear 同样关闭全部旧代，保留仍被使用的槽位，不能直接重建空闲表。普通 LRU 只替换没有在途 IO 的槽位，淘汰失败不关闭仍有效的旧副本。精确模式没有 block-ID LRU/heat 元数据，容量在构造时固定；旧模式的映射、heat 及互斥锁仍需后续预算和无等待审计。

runtime 的 CPU 缓存写入及恢复现持有精确 allocation lease，恢复前校验当前缓存身份，完成后只结算原代，不能再按 block ID 释放潜在替换项。GPU 目标分配失败时保留 CPU 来源；同步 backend 报错时归还本次新 GPU 目标，保留可重试的来源。这里明确要求 backend 在返回或抛出异常前已结束对 Host 指针的访问，不能用这个同步接口提前交还异步 DMA。

宿主新增验证包括一个物理槽位 **128 代**复用（禁止普通 C++ 分配期间执行）、跨池/跨代拒绝、移动 lease、LRU 跳过在途来源、clear 后旧 IO 延续、重复确认及 **64 轮**读取与后台关闭交错。runtime 测试在实际复制回调期间由另一线程关闭缓存来源，逐字节核对数据，并覆盖目标分配失败、同步复制失败及旧 completion 不影响新代。另增加 HeatAware 兼容用例，防止引入 IO 保护时改变原准入策略。最终 Linux ON **28/28**、OFF **27/27**、Windows **23/23**，相关 ASan/UBSan 与 TSan 各 **4/4** 通过。

真实 CUDA 用例使用 `cudaMallocHost` 分配来源，在实际 H2D 已提交、完成事件尚未就绪时由另一线程关闭 allocation。新 IO 和复用被拒绝；确认事件完成后才归还槽位，再复用 **4 代**逐字节校验。完整模型 **40/40**、reset/cancel、36 次/48 次连续请求及 backend guard 通过，两条长轨迹分别回收 **70/190** 块；完整日志为 `/tmp/kvmem-pinned-model.log`。此后只修正旧 HeatAware 准入兼容分支并重新完成上述宿主验收；最终 CUDA 定向场景 0 及 backend guard 通过，日志为 `/tmp/kvmem-pinned-heat-gpu0.log`。全部原始日志及两个受测版本的源码指纹保存在 `build-local/historical-pinned-validation/`。此前偶发 CUDA 非法访问本轮未重现，根因仍未确认。

**阶段边界：** 本项为实际 CPU cache/runtime 路径建立槽位生命周期协议，还没有把每个 historical handle 的 pinned 副本接进 repository 的独立资源回收确认。`PinnedKvTier` 本身不拥有或注册 page-locked 内存；runtime 现有 `cpu_arena_` 仍是普通 Host vector，不能凭类型名称报告为真实 pinned 内存。公开 historical capture 入口仍未启用该缓存；后续须接内容版本、存储 owner、实际 pinned arena 预算、缓存命中与回收失效，并验证完整请求路径。

### Historical pinned 副本的独立生命周期（2026-10-19）

本轮把上一阶段的 CPU cache allocation 协议接到 historical handle。`HistoricalKvRawStore` 可选配独立 pinned 副本，每块在请求接纳前取得完整物理 extent；公开 CUDA 入口使用 `cudaMallocHost`，由存储 owner 持有真实 page-locked 内存。新增 `--kvmem-historical-pinned-gb`，CLI/server 均要求同时开启 historical capture；它与原有 `--kvmem-cpu-gb` 是不同资源。预算不足或 allocator 失败时，整次预约失败并归还已经取得的资源，不等待后续 GC。短尾仍按完整 extent 计费。

producer 在预约阶段取得原始 pinned IO 许可，并随唯一 writer 交接。因此 Writing 块即使在安装前或写入中退休，已经承诺的 packed K/V 复制仍能完成；退休不会重新开放读者。packed K/V 全部完成后归还 pinned producer 许可，不必等待 Mean-K；半写取消则在最后一个 producer 退出时归还。这个许可与后续 stage-in reader 分开，不能在退休后临时重新申请写入权。

repository 增加独立 pinned 释放确认。退休先关闭新 IO，已有读者保留原 allocation generation；其间 RAM packed K/V 与 Mean-K 可以独立释放。最后一个 pinned 使用者退出后，后台先实际释放 page-locked 内存，再返还 pinned 槽位；完整 metadata/Ready 仍等待其余身份与资源结清。请求预约回退遵守相同顺序，避免慢 deleter 尚未完成时新请求复用容量。CUDA 释放报错时停止执行，不将未确认释放的容量记作可用；这属于设备失败后的 fail-stop 行为，没有实现设备故障自动恢复。

受管理 stage-in 可直接借用 pinned 地址，持有来源许可直到检查过的 H2D 完成；提交失败或完成状态不确定时沿用已建立的保留/清理协议。GC 不修改 GPU 工作集。当前仍保留普通 RAM 副本，这不是 Live RAM 压力卸载；同时配置独立 NVMe 与 pinned 副本明确拒绝，未验证组合不静默接纳。

自查修正了两处旧草稿的缺口：预约回退曾先返还槽位再调用实际内存 deleter；pinned producer 曾在发布时才申请 IO。新增受控并发测试在 deleter 尚未结束时尝试新预约，分别覆盖 GC 和私有预约回退；另覆盖安装前退休、半写取消、packed 完成后 Mean-K 延续、一个槽位连续 32 代、实际 allocator 失败和整请求回退。Host 单元测试使用可控普通内存模拟 allocator，只证明生命周期与记账；真实 page-locked/H2D 证据来自 CUDA 测试。

最终 Linux NVMe ON **29/29**、OFF **28/28**；Windows 原有 23 项加新增 pinned 项均通过分项复验。Windows 首轮生命周期测试因沙箱禁止创建测试链接失败，在允许该测试操作的环境复验通过。历史 GC 的 ASan/UBSan **14/14**、TSan **14/14** 通过，最后仅调整测试屏障后 pinned 用例再次通过 ASan/UBSan 和 Windows。上述统计不包含工作树中另一个未提交的 GC 演示目标。

TSan 首次遇到本机地址映射冲突，使用已有的进程级 `setarch x86_64 -R` 方式复验；随后条件变量测试屏障出现 double-lock 告警。独立的两线程最小程序（不包含项目代码）复现了同样告警，故将测试屏障改为有超时的原子握手，没有关闭 TSan 检测或忽略告警。首次失败、最小复现及最终通过日志均保留。

最终真实模型 **43/43** 场景、reset/cancel、两条连续请求及 backend guard 通过；两条长轨迹分别回收 **70/190** 块。新增三个 pinned stage-in 场景分别验证正常 H2D、真实提交后注入失败、无法确认完成，每个场景逐层核对 **48 层**。这些 stage-in 是确定性构造的实际设备路径；先前普通 CLI 请求虽发生 pinned 分配和 GC，但没有触发 pinned stage-in，不能把该 CLI 结果当成自然缓存命中率或性能改善证据。此前偶发 CUDA 非法访问本轮未重现，根因仍未确认。

开启 256 MiB historical pinned 预算的 delete HTTP 回归 **34/34** 断言通过，覆盖 Chat Completions/Responses、JSON/SSE、精确前缀延续、容量拒绝、reset、prefill/生成取消和后续压力请求。四次压力请求均 `committed=14`、完成边界 `reclaimed=13`，policy/worker failure 为零；begin 日志记录实际 pinned 分配，`/props` 返回预算。测试脚本新增 `--pinned-gb`，可使用相同流程复验。预算当前按 historical domain 计，新旧域切换时的全局总 bytes 仍是待办；这些 HTTP 请求没有证明自然 pinned stage-in 命中或性能收益。

本轮受测源码指纹、构建与测试日志放在 `artifacts/historical-pinned-binding/`；远程仍只使用 `/tmp/kvmem-gc-99bef27.3CD9MU` 隔离副本。完整交付仍需下列生产接入与验收；执行顺序以文首“当前执行计划”为准，先审计首批配置可达的 active 入口、预算和长期有界性，不再先扩展全部入口及可选存储组合。该调整不删除 P5/P6 或质量验收要求。用户已确认导师指定 DeepSWE，但尚未配置运行环境，后续需核对模型、运行入口和评测工具，不能用本节模型/HTTP 生命周期测试代替实际任务质量。

### 首批长测入口与预算审计（2026-10-19）

按调整后的顺序，先扩展已有真实模型连续请求测试，没有新增存储组合或策略。`kvmem-historical-access-test MODEL --soak-seconds 3600` 在同一个 context/repository 中持续执行至少指定秒数，并至少完成原有 48 次带生成请求；不通过 reset 或重新加载模型清空历史。固定 recency、f16、单 GPU、RAM、8 个历史槽、32-token 块、96-token 工作集、16-token microbatch；交替追加 64/48 个输入 token 和 3/1 个实际贪心生成 token，每次按完整 3-token 输出上限预约。该输入是生命周期压力 fixture，不是编码任务或 DeepSWE 质量测试。

每个请求继续检查已退休旧句柄不可读、槽位/目录有界、块表和行索引地址/容量不变、capture arena 不变及 payload 容量上界；新增检查后端 scheduler、输出 buffer 和图结果/元数据地址与容量不变。长测的观察者只保留最初一组和固定数量的近期句柄，避免测试自身随运行时间增长。接纳前容量不足的重试单独计数；这些调用尚未进入 active，不能把它们记成已经接纳的请求等待 GC，也不能从吞吐结果中隐去。

每约 10 秒及结束时输出 `KVMEM_SOAK` JSON，记录同一 repository ID、完成请求/输入输出/块数、GC 统计、准入拒绝数、payload 与固定上界、目录/块表/行索引、arena/后端容量、CPU 清理队列占用、Linux RSS/高水位/线程数及 CUDA free/total。采样发生在请求完成边界，不能覆盖请求内部所有峰值；各后台计数也不是联合原子快照。进程高水位包含模型加载阶段；本机预检中 CUDA free 返回 0，保留原始结果，不据此推导可接纳请求的资源额度。

60 秒预检成功：实际运行 **60.255 秒、205 次请求、11,899 token（含 411 个生成 token）、821 个块经过 8 个槽位，回收 819 个**，记录 **97 次接纳前拒绝重试**，policy/worker failure 为零。10 秒后的 RSS 样本为 **794,348–795,148 KiB**，历史 payload 未超过 **25,165,824 bytes**；这些只是本轮观测和已列明的断言，不是完整总内存上界证明。原有两条连续请求回归也通过：36 次/2,304 token/72 块回收 70，48 次/2,784 token/192 块回收 190；后者记录 21 次准入重试。backend guard 通过。无模型仍返回跳过码 77；0、负数、非整数及超过 86,400 秒的时长均在模型加载前拒绝。

一小时任务在原隔离目录运行，运行前核对实际 GGUF 的 SHA-256 与所给 blob 名相符；外层设 4,200 秒超时，分别保存驱动、普通回归和长测退出码。**首次长测失败，未达到一小时门槛：** 完成 954 次请求、累计 55,332 token 后，下一次准入持续被拒绝，测试的 5 秒重试期限耗尽，驱动和测试退出码均为 1。最后一个完成边界记录 committed=3816、reclaimed=3815，policy/worker failure=0；这些计数不能解释拒绝原因。全过程含模型加载耗时 6 分 46.62 秒，最后一个定时样本的持续解码时间为 316.512 秒；不能把包含加载的时间当作持续负载时长。没有调大等待时间或将这次失败记为通过。

命令、源码指纹、预检和首次失败日志已备份在 `artifacts/historical-first-soak/`，远程原始输出保留在 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-first-soak/hour/`。首次长测使用 `6d8aa20` 的测试、`c1f87ce` 的生产实现；后续仅在相同基线的准入拒绝处补充 finished/failed、操作数、arena 引用、队列和 GPU 来源状态，600 秒诊断也以退出码 1 结束。诊断没有混入下节共享额度改动，也没有覆盖首次失败证据。

诊断最后的定时样本为 490.161 秒、1,605 次请求；持续拒绝时 `finished=1`、`operations=0`、`arena_refs=2`、`queues=1/0/0/0/0`，GPU reader/retired 计数均为零。根因是 GC 在 producer 查询前退休了整批 Writing 块，使该批所有采集范围变成 Closed；recency 又没有 Mean-K 图节点，旧 `harvest_pending` 只在存在 Managed 范围时移除队列项。请求能结束，但残留批次一直持有 arena，下一次请求因队列非空而永久拒绝。

修复让受管理采集在该无 Mean-K 分支中正常消费已执行的队首，即使整批范围都已 Closed。它不禁止退休 Writing 块、不放宽下一次请求的安全检查，也不延长 5 秒测试重试期限。新增 `--closed-capture` 用例直接控制“追加 → 退休 → producer 查询 → harvest”的顺序：修复前退出 1，明确报队列/arena 残留；修复后退出 0，检查 Closed 范围、队列清空、arena 引用归还、finish 及下一次准入均通过。同期 backend guard 通过。

一次早期对照因压缩包保留旧时间戳而混用旧 adapter 对象与新存储库，产生段错误；保留该日志但不将它作为 GC 复现或验收结果。重新核对源码指纹并强制更新受影响文件时间后，构建日志确认 adapter 和存储库重新编译，才取得上述有效的失败/通过对照。新一轮顺序验收保存到 `historical-first-soak/queue-fix-0949ac6/`，按 closed-capture、reset、repeated、3600 秒 soak 执行；任一前置项失败即停止。另每秒记录设备占用，补充请求边界的内存采样；尚不能将正在运行的驱动记为一小时通过。

修复后的三项前置测试退出码均为 0。两条连续轨迹分别为 **36 次请求、2,304 token、72 块、回收 70 块、0 次准入重试**，以及 **48 次请求、2,784 token（含 96 个生成 token）、192 块、回收 190 块、20 次准入重试**；旧句柄拒绝与固定工作空间断言通过。随后驱动开始新的 3600 秒 soak；时间从模型加载后的持续请求阶段计算。该轮源码对应 `1210082`，保存实际文件和二进制 SHA-256；目录中的 `0949ac6` 是建立这次验证时的父提交名，不能用目录名代替源码指纹。

**2026-10-20 终态核实：修复后一小时同域长测通过。** 驱动与 soak 退出码均为 0，结束时间为 2026-10-19 08:08:34 UTC；持续请求阶段 **3,600.148 秒、11,448 次请求、663,984 token（含 22,896 个实际生成 token）**。累计 **45,792 块经过 8 个物理槽位（5,724 倍容量），提交退休 45,792、实际回收 45,791**；最后一块尚在异步结清，不能将退休数当作已回收数。policy/worker failure 均为 0，记录 **5,139 次接纳前重试**，共享 payload 额度拒绝为 0；backend guard 通过。

保存的 358 个样本中，60 秒以后的 RSS 为 **794,052–796,932 KiB**，历史 payload 与共享 payload 样本均在 **0–12 MiB**；单域断言上界 24 MiB、跨域 allocator 额度 48 MiB。块表 1,312 bytes、目录 8 项、行容量 256、arena 47,104 bytes、后端输出 38,895,616 bytes、图元数据 19,778,080 bytes 均保持不变；清理服务占用 2–3 个槽位，容量 64。请求边界采样没有覆盖所有内部瞬时峰值；进程高水位 18,532,604 KiB 包含模型加载，不能当作历史 payload。独立每秒 GPU 采样覆盖整个前置回归和长测驱动，占用范围 594–16,077 MiB，属于整张设备而非单个 GC 模块。本机 CUDA free 持续为 0，仍不拿它推导可用额度。

原始日志、模型/源码/二进制指纹、构建配置和设备采样已保存到本地 `artifacts/historical-first-soak/queue-fix-hour-evidence.tar.gz`，摘要为同目录 `hour-summary.json`。该结果完成首批的“一小时同域持续复用”门槛，支持队列修复在此压力负载下有效；它没有替代长期 reset/cancel、完整资源上界、真实编码/agent 质量或独立 CUDA 问题的验收。

预算审计确认仍有必须处理的缺口：reset 为新域创建新的 `HistoricalKvRawStore`，旧域可因读者或未退出 worker 继续保留；每个 store 的 payload 额度独立。进程 CPU 清理服务的 64 个任务槽限制任务数量，并非新旧域共享的总 bytes 预算。因此即使本次同域长测稳定，仍需验证 reset/cancel 重叠的总量，落实共同准入约束；不能据此关闭完整内存上界验收。现有正常请求入口仍按有限输出预约，SSE 不改变该语义。

另重新读取并备份此前 CUDA 非法访问原始日志：首次失败位于旧完整回归通过第 10/35 场景后的下一场景，错误从 `harvest_pending` 调用的 CUDA stream 同步返回。该处报告不等于故障最初发生处，也尚不能归因于回收。Compute Sanitizer 当时因 WDDM debugger interface 初始化失败而不可用，不能算 GPU 内存检查通过。原始模型失败日志及 sanitizer 诊断已纳入本轮证据；仍列为未关闭问题，后续在长测结束后安排针对性复现，不与正式长测争用 GPU。

### 新旧历史域共享 RAM payload 额度（2026-10-19）

为处理上节预算审计发现的问题，新增可共享的 `HistoricalKvPayloadBudget`，并将额度绑定到 packed K、V 和 Mean-K 向量的 allocator。每次分配先按实际请求的 capacity bytes 扣额度，额度不足立即使请求预约失败；底层分配失败归还本次额度。回收先完成真实 deallocate 再返还额度，短尾缩小逻辑长度不返还仍被向量保留的容量。旧域、半完成私有预约和仍被持有的 payload 都参与同一份额度，不能仅根据退休或任务入队减账。

首批 adapter 在首次建立历史域时设置相当于 **两份完整历史容量** 的 payload 额度，reset 通过复制存储配置继承同一个预算对象。它限制 bytes，不是只允许两个 domain：多个较小旧域可以共存，但全部实际 payload 之和不能超限。新的 reset 预约失败发生在替换旧域之前，原会话继续有效；已经接纳的请求仍使用事先取得的 payload。此额度按一个 adapter 的连续历史域共享，尚不是跨多个 adapter 的进程总预算。

宿主用例验证：旧读者跨域保留、第三份 payload 拒绝、申请到一半失败完整回退、读者退出但 GC 尚未释放时仍不授予额度、后台实际释放后再准入、64 次共享额度复用、store 已析构但私有 payload 仍持额度、逐分配点失败回退、阻塞真实 deleter 时不提前返还额度，以及 320 次并发额度申请/释放。既有 K/V 与 Mean-K 独立释放用例也核对共享计数，包含丢失释放确认后的幂等重试。

Windows historical 测试 **13/13** 通过；独立 Linux 宿主构建 NVMe ON **23/23**、OFF **22/22**，相关 ASan/UBSan **14/14**、TSan **14/14** 通过。这里 Linux 使用仅包含 kvmem 的独立构建，测试数不能与包含 server/会话工具的旧全项目测试总数混算。adapter 与新增模型测试也通过实际 CUDA 构建。

真实 Qwen3 模型 `--reset` 验收退出 0：持有 **11 个旧域**的读者，加上当前域占满 **37,748,736 bytes（36 MiB）**共享 payload 额度；下一次 reset 在替换前拒绝，当前会话身份与旧数据仍有效；释放读者并等待后台真实释放后，额度归还，后续新请求成功完成。原有部分输入取消、预约失败回退、暂停旧 GC、跨域读者隔离、显式取消和 GPU/backend 工作空间复用检查同轮通过，backend guard 通过。这证明本项共享 payload 约束的真实接线，不等于所有资源已具备统一总预算。

宿主证据放在 `artifacts/historical-shared-payload-budget/`；模型日志保留于远程 `historical-first-soak/queue-fix-0949ac6/reset.log`，已打包到本地同名证据目录中的 `queue-fix-regression-evidence.tar.gz`。所有库替换均在前一个 GPU 进程终止后完成。

本项只计算历史权威 payload 的 allocator 请求字节，不包括 allocator 自身开销、元数据、模型、搬运暂存、独立 pinned/NVMe 副本和调用者的读取快照。上述资源仍须纳入对应预算与长期验收；上节连续请求准入停滞另有独立的队列清理修复，不能将共享额度改动当作它的修复。

### 首批服务器复验与原生编码预检（2026-10-20）

使用上述已验证实现和重新链接的 `llama-kvmem-server`，现有 `test_server_historical_gc.py` 在关闭独立 historical pinned、NVMe 和 legacy CPU spill 的配置下顺序复验。**off 31/31、shadow 33/33、delete 33/33、原生模板 delete 8/8** 断言通过，四项退出码及驱动退出码均为 0。覆盖有限请求预约、接纳前容量拒绝、拒绝后保留旧 cache、精确前缀延续、reset、Chat/Responses SSE、prefill/生成取消和恢复、模式统计与实际后台释放。原生模板用例验证两轮对话及压力 reset。

上述仍是强制短输出的生命周期回归，不能算编码质量结果。原始证据保留于远程 `historical-first-http-1210082/`，本地归档为 `artifacts/historical-first-http/http-evidence.tar.gz`。另准备了六轮原生模板编码请求的预检与固定输入回放工具，记录实际响应、SSE 时间、token 计数、进程内存与设备占用，不执行模型生成的代码。先用单独标注的大容量 off 参考确定输入和生成上限，再冻结压力对照参数；该预检不是 agent 或 SWE-bench 验收。

第一份预检有六次 HTTP 200，但第 3/4/5 轮只生成 128 token，均以 `length` 结束。原因是请求的 `max_tokens=256` 被服务器现有的 `gen_reserve=128` 限制；旧预检只检查 SSE 完成，退出 0 不能视为完整回答通过。完整失败证据保存为 `artifacts/historical-real-workload/native-pilot-evidence.tar.gz`。没有通过修改产品的上限规则绕过资源预约。

新增可维护的 `scripts/test_server_historical_workload.py`，将请求输出上限和生成预约统一为显式参数，检查 `/props` 发布值，保存每轮实际请求及 SHA-256、完整响应和事件时间；三种模式回放同一组请求，不在 503 后偷偷重试或重置。默认使用原生模板、greedy、seed 42、presence/frequency penalty 0、repeat penalty 1；原生聊天模板的前缀不匹配仍走已说明的完整重算路径，不能将多轮 HTTP 200 当作同域缓存延续成功。退出成功要求回答正常结束且未截断，shadow/delete 还要求日志确实出现策略决定/实际释放；即使成功，也不代表生成的代码语义正确。

第二轮配置为 768 历史槽、32-token 块、1024-token GPU 工作集预算、512-token 生成预约、4096 context。关闭 GC 的六轮中仍有两轮重复并截断，已出现负数处理、只用 O(n) 内存等要求丢失的回答，不能把这个参考标记为质量通过。同时，初设 low=32/high=64 没有触发策略，shadow 的 proposed/shadow 均为 0；这轮仅作为普通负载与失败诊断保留，不能算实际 GC 压力对照。后续压力配置固定 low=384/high=512，在相同请求与其它容量不变的条件下单独重跑。增加未启用 KVMem 的参考诊断，用于区分输出问题发生在哪条路径，不将它作为新增生产支持组合。结果以终态日志为准。

压力组已取得终态：两种模式均提交六次相同请求、均 HTTP 200、无 503。shadow 确实产生决定（各轮 39–299 个），未实际退休或释放，六份输出与 off 逐字相同，仍有两轮截断。delete 确实退休并释放（各完成边界 260–325 块），policy/worker failure 观察值为 0，但六轮全部生成至 512-token 上限，第一轮已从正常函数变为重复无关文本。**这是一项未通过的实际负载验收，不能用无崩溃、成功释放或 HTTP 200 抵消。** 原始证据为 `artifacts/historical-real-workload/pressure-reset-v2-evidence.tar.gz`；未触发 GC 的上一轮保存在 `native-comparison-evidence.tar.gz`。

初步源码审计：`begin_active_ubatch` 每个微批次重取 ExecLease，退休的驻留块会通过正常前台边界移除逻辑 cells，后续 attention 不再包含它。导师文档 §4.3 允许一个明确的 ActiveId 覆盖多个 decode token，并要求已有窗口继续；当前窗口较细，整请求预约又会先降低 Ready 水位，使压力策略可能在本次输入尚未完整处理时退休新块。该路径需要用短输出、明确 trace 和参考路径定位；现在尚不能把差异认定为 GPU 越界、错误释放或单一策略参数问题，也不能靠禁止策略删除 active/Writing 块来掩盖它。针对性诊断先于新的长期验收。

未启用 KVMem 的第一份诊断命令错误地传入 `--kvmem-historical-capture 0`，在加载模型前被参数校验拒绝，没有得到参考输出；此失败属于测试配置问题。已改为完全省略 historical 参数，并安排独立诊断，不能将尚未运行的参考组计为通过。

随后完成最小复现：同一个 88-token 原生编码输入、相同 greedy/seed，只生成 16 token；historical 配置为 32 槽、1024 GPU 工作集预算、16 生成预约、low=20/high=26。off 与完全关闭 KVMem 的参考都返回 `def take_last(values, n):` 开始的相同 16-token 前缀；delete 返回 `def {` 后重复空白。delete trace 同时记录了输入 block 0/1 等在后续微批次中被 `historical_active_skip` 移除。未启用 KVMem 的参考使用 2048 context，historical 两组使用 4096，三组输入加输出均未达到对应上限；不能据此宣称所有长上下文行为相同。此实验刻意截断，只验证早期分歧，不检验完整答案正确性。证据为 `artifacts/historical-real-workload/active-boundary-evidence.tar.gz`。当前优先明确并修正有限请求内的 active 窗口，保留新窗口不得复用退休块、旧窗口许可到实际完成、Host/GPU 独立释放的约束，不直接取消退休检查。

资源审计仍明确保留一个未关闭点：已预约的 ggml tensor 元数据与 scheduler/output buffer 不包含所有构图 C++ 对象。当前 Qwen3MoE 构图仍会创建 graph builder 及四类 input holder，图结果 reset 会清理这些对象；一小时容量稳定不能证明整个 decode 不再进行动态分配。后续须针对首批可达路径补充实际分配证据与预约处理，不能以已有大 buffer 地址不变来替代完整准入保证。

### 反复 reset/cancel 的持续验收入口（2026-10-20）

模型测试新增 `--reset-soak-seconds 1..86400`，在同一个模型/context 中交替正常完成、prefill 尚未完成时取消、生成一部分后取消；每次再建立新 incarnation。固定 8 个历史槽，输入额度 64、最大输出 3、最多 5 次追加；每个周期在 finish/cancel 后保存一个真正已发布的 CPU 读取许可，让该读者跨越下一次 reset 和新域的实际解码。GC 在请求接纳后启动，使用 low=1/high=2；六槽预约留下两个 Ready 槽，因此本轨迹不以水位触发 LRU 删除，保证外部读者的取得不依赖时序碰运气。实际旧域释放由 reset 关闭域及后台退出协议触发，抽查旧块首个注意力层的 packed K/V 内容不变，并验证旧身份不能取得新许可、旧 reader 最后释放不在前台销毁历史 payload，以及旧 domain 最终在后台退出。观察者只有有界的弱引用和单份比较快照，不能随测试时间累计保留历史。同域请求执行期间的 GC 压力由独立的 `--soak-seconds` 覆盖，本项专门验证切换时的旧域重叠。

整个轨迹检查新旧域沿用同一 payload 额度、额度不超过两份完整历史容量，行索引/capture arena/后端输出和图元数据继续复用，块表/目录与后台清理队列有界。接纳前失败重试单独统计；已接纳请求不靠测试重试补额度。结束时要求全部旧 domain 均已销毁，最终当前 domain 仍由 live context 持有。输出 `KVMEM_RESET_SOAK`，记录完成/两类取消/拒绝/旧域退出数、payload、RSS 与固定空间数据；该测试不覆盖 HTTP 断连本身，后者由既有 server 回归覆盖。

同时记录 decode、同步和采集阶段本线程普通 C++ `operator new` 次数及累计请求字节，预约阶段和测试读快照不计入。这是为后续资源缺口定位提供实际证据，**累计分配字节不是仍驻留的内存，也未覆盖 C malloc、aligned new、后台线程或 CUDA 分配**。计划先通过短预检，再执行一小时持续运行；新增入口或驱动启动本身不算长测通过。

首次短预检退出 1，报告 `reset soak finish failed`，因此驱动没有启动一小时任务。原因是测试在 finish 前保留外部 ReadLease，而现有 `HistoricalKvCaptureSession::complete` 要求 outstanding operations 已结清；不是已经证实的产品回收故障。第二份测试又尝试在 finish 后启动 GC，被已有 owner 校验拒绝；也保留为测试配置失败。最终按上述协议安排先启动非压力 GC、完成/取消后再取得读者，没有为测试放宽产品 API。

最终短预检与驱动退出码均为 **0**：持续 **60.101 秒、220 次请求**，正常完成 74 次、prefill 取消 73 次、生成取消 73 次，合计 12,039 token（其中实际生成 295 token），接纳前拒绝 0；**219 个旧域全部退出，pending 为 0**。共享 payload 上界 48 MiB、采样峰值 36 MiB、结束为 15 MiB，最后当前域仍由 context 持有。块表 1,312 bytes、目录不超过 8 项、行容量 256、arena 47,104 bytes、后端输出 38,895,616 bytes、图元数据 19,778,080 bytes 的断言通过；结束 RSS 815,164 KiB，模型加载高水位 18,531,652 KiB，线程数 5。无效时长 0/86401 在模型加载前拒绝，backend guard 通过。证据为 `artifacts/historical-real-workload/reset-pilot-final-evidence.tar.gz`。

该短测的前台 decode 区间累计观察到 **572,250 次普通 C++ 分配、44,204,172 个请求字节**，因此完整的“接纳后不再依赖新分配”要求尚未满足；这不表示泄漏了 44 MB。仍需定位调用点、核算实际保留量并处理首批可达资源。反复切换的一小时任务因实际删除输出差异暂缓排程；这里通过的是 60 秒预检。

### 有限请求内的 GPU 执行窗口修复（2026-10-20）

原生编码压力测试暴露了一个边界错误：适配器把每个 ubatch 都视为一次全新的 active。于是，同一次回答已经获准使用的历史块被 GC 退休后，会在下一批 token 的 attention view 中消失。88-token 提示、16-token 输出上限的最小复现中，off 与无 KVMem 参考均以正常的 `take_last` 函数开头，delete 则生成 `def {` 和空白。此前的模型测试把“下一微批次排除退休块”当作预期，未覆盖同一有限回答的连续执行语义。

依据导师文档 §4.3 对有限 active window 的约定，首批 recency/RAM 路径现在将一个已预约、输入/输出/追加次数均有上限的请求作为 GPU 执行窗口。首次选择旧 GPU 历史时取得权限；该请求新产生的块，在 writer 发布 Writing 的同一次状态转换中取得原始 GPU 执行权限，避免 GC 在两次独立 acquire 之间退休该块。后续 ubatch 只复用窗口内已经取得的同代权限，不能据“GPU 上还有副本”取得新权利。实际提交仅更新此次确实用到的块的 LRU，已被普通容量管理移出的块不会继续被虚假 touch。

这是 GPU-only 权限，不包含未来 CPU 读或 H2D 权限。GC 仍可退休 Writing/active 块，Packed K/V、Mean-K 等 CPU 分量仍在各自依赖结束后后台释放；元数据槽位则等旧窗口退出后才能返还 Ready。GPU 容量管理仍可按原有规则移出驻留块，保留窗口权限不代表无限保留全部上下文。请求完成、健康取消、成功 reset 或确认设备工作结束后的析构关闭窗口；未确认设备完成的失败路径保留已有依赖。过早调用 finish 不会关闭尚未完成输入的窗口。下一请求重新获取权限，退休块不能复活。

窗口数组在请求接纳前按 repository 固定容量准备，实际使用列表按 GPU 槽数准备，计入 active 空间统计；执行和关闭不扩容。trace 使用 `repository + window` 标识开始/结束。`test_server_historical_workload.py` 的物理回收活动判定同时接受“完整槽回收”和“CPU 已提前释放”的计数；单纯 proposal/commit 仍不算释放证据。Ready 信用继续只来自真实完整回收，不因这个观测口径变化增加。

新增宿主测试覆盖原始 writer/GPU 权限与退休的并发顺序、退休后拒绝新 GPU/CPU 许可、CPU 早于窗口退出释放、同槽代际校验、窗口关闭后禁止追加、无前台 payload 释放，以及仅实际提交的块更新 LRU。真实模型用例相应改为检查同一窗口连续执行，窗口结束后旧许可消失；保留 GPU 字节、后台 Host 释放及设备错误处理检查。

初版修复的最小复现已通过**前缀一致性**检查：同一 HTTP 请求体下，修复后 off/delete 与先前无 KVMem 参考的 16-token 内容逐字相同；delete 观察到 14 次退休及 14 次 Host 提前释放，窗口内 active_skip 为 0、开始/结束各一次。完整元数据回收在 finish 采样点仍为 0：窗口关闭与后台完成最后回收不是同一个时刻，不能将其误读为 CPU 尚未释放。四个针对性模型场景（5/6/7/31）均退出 0。证据为 `artifacts/historical-finite-window/probe-evidence.tar.gz`、`probe-summary.json`；两组短输出故意以 length 结束，不能称为完整回答质量通过。

最终候选版另补充提前 finish 保护、窗口标识和空间统计，源码包 SHA-256 为 `bffd19f1433861537b510bab19a8234aa51a91cb9388fe73697bcde06efaad03`。Windows 三项直接相关宿主测试通过；全套首次 22/25，另外两项临时目录权限失败在指定项目内临时目录后通过，剩余会话缓存符号链接测试受 Windows 权限限制。Linux ASan/UBSan 首次 27/30，三个 NVMe 用例因已有 `/tmp` 目录权限失败；改用本轮隔离临时目录后这三项全部通过，未修改产品代码以绕过检查。最终候选版的 43 个真实模型场景、reset、两组重复请求、全 Closed capture 及 backend guard 已通过；这次通过不单独注销此前独立记录的偶发 CUDA 非法访问问题。

最终候选版 reset/cancel 短测退出 0：60.284 秒完成 198 次请求，正常完成、prefill 取消、生成取消各 66 次，197 个旧域全部后台退出，接纳前拒绝 0；共享 payload 采样峰值 36 MiB/上限 48 MiB，结束 9 MiB，固定元数据/arena/backend 检查通过。前台 decode 普通 C++ 分配仍有 514,475 次、累计请求 40,704,096 bytes；并未解决完整资源预约缺口。

TSan 首次全套在启动阶段遇到 `unexpected memory mapping`。仅对测试进程固定地址布局后，直接相关 historical 测试全部通过，但整套为 **28/30**：未修改的 lane pool 报 `double lock of a mutex`，单独复验仍报错；通用 `raw_kv_store_test` 持续等待，保存线程等待状态后终止，单独设置 60 秒超时复验在 0.56 秒通过。两项首次异常均保留，不能将 TSan 整套报告为通过，也不将它们未经定位归因于本次修改。lane pool 的告警需核对实现与当前 GCC 11/libtsan 运行时；本轮未通过禁用检查来掩盖告警。

完整原生编码压力重放已结束，**仍未通过**。六轮 HTTP 请求体指纹逐轮一致；off 为四轮 stop、两轮 length，delete 为一轮 stop、五轮 length，仅第一轮完整内容相同。第二轮共同前缀为 151 字符，在第 44 个非空内容流式片段处分歧；随后 delete 开始生成无关函数。各轮确有退休及 Host 提前释放，policy/worker failures 为 0，因此不能归结为没有触发 GC，也不能把 HTTP 200 当质量通过。两组负载脚本均退出 1；外层驱动 0 仅表示排程结束。最终证据为 `artifacts/historical-finite-window/final-evidence.tar.gz`，摘要为 `final-summary.json`。

代码检查发现尚未贯通的第二个入口：`prepare_working_set()` 在 GPU 槽位不足时调用 `prepare_prefill_pressure()`，其 `pick_prefill_pressure_blocks()` / `set_selection()` 仍经 `is_live()` → `allows_new_selection()` 过滤已退休块。这条路径可以在同一回答内移除仍有旧窗口权限的 GPU 驻留内容。本轮已修复 ubatch 权限释放及创建竞态，**容量压力选块与该窗口的配合仍待修复**；完整压力失败保留为下一步复现，不能宣布整个输出异常已解决。

针对容量路径，建议只传递确实仍驻留、已获同代 GPU 权限的块的已有窗口许可；不通过全局放宽 `is_live()` 来绕过新选择、CPU 读取、H2D 或下一窗口的退休检查。但导师文档 §4.3 同时规定新 reselect 建立新 ActiveId，因此“因容量不足自动整理是否属于新 active”需要明确；此建议尚未作为最终行为实现。边界定型后先补容量压力用例，再重放第二轮及六轮对照。期间继续独立的并发排障和资源预约审计。长期全部资源有界、接纳后无新分配及原偶发 CUDA 错误仍需后续处理；这些门槛未完成前不恢复扩展配置开发。DeepSWE 按用户要求继续暂缓。

### 并发检测运行库与存储关闭通知（2026-10-20）

先把两个异常分开调查，保留上一轮失败日志。lane pool 的 `double lock of a mutex` 来自当前 GCC 11/libtsan.so.0 与标准库计时等待的兼容性缺口：实际二进制引用 `pthread_cond_clockwait@GLIBC_2.34`，运行库却没有对应 interceptor。不含项目头文件、全部共享数据均由同一 mutex 保护的最小条件变量交接程序，同样报 double lock、退出 66；仅改用 system_clock 的等待重载后退出 0。该行为与 [GCC 的修复讨论](https://gcc.gnu.org/pipermail/gcc-patches/2021-May/570244.html) 和 [GCC 11 问题记录](https://gcc.gnu.org/pipermail/gcc-bugs/2022-July/793323.html) 一致。

从主机配置的 Ubuntu 软件源下载并解包 GCC 12.3.0 与 libtsan.so.2 到本轮隔离目录，没有安装或替换系统编译器。新运行库提供 `pthread_cond_clockwait` 拦截；两种最小等待程序均通过，未改 lane pool 产品源码的完整宿主 TSan 回归也为 30/30。`setarch x86_64 -R` 仅作用于测试子进程以处理此前的地址布局启动冲突，没有修改系统 ASLR 设置，也没有使用告警抑制。

新增 `scripts/check_tsan_runtime.sh` 与独立 `tests/tsan-runtime-probe.cpp`，用于在解释业务并发测试前确认检测环境。脚本先运行正确同步的 timed-wait，再运行故意包含数据竞争的阳性对照；后者必须退出 66 并报告 `ThreadSanitizer: data race`。脚本覆盖继承的 TSAN_OPTIONS，避免外部抑制选项造成假通过。旧工具链被自检拒绝，新工具链自检通过，阳性对照确实退出 66。本轮运行方式如下，工具链路径为已解包目录，不是新增产品依赖：

```bash
out=/tmp/kvmem-gc-99bef27.3CD9MU/historical-tsan-runtime-18eaddd
LD_LIBRARY_PATH="$out/toolchain/usr/lib/x86_64-linux-gnu" \
KVMEM_TSAN_FIXED_LAYOUT=1 \
bash scripts/check_tsan_runtime.sh "$out/toolchain/usr/bin/g++-12" "$out/preflight-gcc12"
```

另发现 RawKvStore 析构中真实的条件变量协议缺陷。旧代码在 `mu_` 之外设置原子 `stop_io_` 并通知，而 I/O worker 在 `mu_` 内检查条件再等待：若通知恰好落在“检查为 false”与“进入等待”之间，worker 会错过通知，析构的 join 可能一直等待。原子变量只能消除变量自身的数据竞争，不能修复丢失唤醒。

修复将停止标志的发布放到同一 `mu_` 临界区内，解锁后通知，最后等待 worker 退出；已有排队 I/O 仍按原规则结清。回归通过仅测试可见的 friend 访问，在真实 store 的线程句柄上装入可控 waiter，使用真实 mutex、condition variable、stop flag 和析构函数，暂停在检查与等待之间，不依赖磁盘速度。旧实现复现 `early_stop=1 notified=0`、退出 134；新实现通过。测试中的等待有超时，失败不会无限卡住。这个复现证明了关闭协议缺陷，**但此前原始卡住现场没有调用栈，不能宣称已证明那一次挂起必然由此造成**。

修复最终源码通过 GCC 12 TSan **30/30**、ASan/UBSan **30/30**，`raw_kv_store_test` 在 TSan 下重复 **20 次全部通过**。本轮没有改变选块/退休语义，也没有重跑真实模型质量或一小时负载，不能将宿主通过扩大为生产验收完成。普通 C++ 分配统计还只是前台 decode/同步/harvest 区间的累计请求字节，既不是存活内存，也不覆盖 C malloc、其他线程或 CUDA 分配；完整资源预约缺口仍然保留。

原始日志、旧/新最小复现、阳性对照、软件包及最终源码指纹保存在 `artifacts/historical-tsan-runtime/evidence.tar.gz`，SHA-256 为 `cfecde2370da56f2ad309ac9eb68644b1ba5f26d45f52cd542e60b213c03b16d`；远程证据目录为 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-tsan-runtime-18eaddd`。源码修复与检测工具自检单独提交，后续继续按首批范围推进资源预约和真实负载问题。

### 接纳后分配来源审计与层目录复用（2026-10-20）

新增仅在 Linux 模型测试中启用的前台调用栈采样器，使用 `KVMEM_TEST_ALLOCATION_TRACE=/path/to/trace.tsv` 打开。采集区间与既有 reset soak 的普通 C++ 分配计数一致：prefill 包含 decode、同步、harvest/commit，生成包含 decode 和同步。采样表固定 2048 项、每项最多 16 层栈；先预热 unwinder，采集期间不扩容，符号查询和文件输出放在采集结束后，溢出次数单独记录。它不覆盖 C malloc、aligned new、其他线程和 CUDA 内存，也不表示存活堆大小。本轮两份 trace 均无表溢出，调用次数和字节数逐项与独立计数器相等。

在当前 Qwen3 模型上固定执行 96 次请求：正常完成、prefill 取消、生成取消各 32 次，共 5,248 tokens（含 128 个生成 token），95 个旧域均后台退出。原适配器在此区间发生 249,921 次普通 C++ 分配、累计申请 24,711,176 bytes。其中计算图调度的临时依赖表为 166,944 次/7,178,240 bytes，CUDA MoE 匹配器为 37,116 次/2,375,424 bytes，其余 CUDA 图优化为 18,438 次/958,824 bytes，物理层目录复制为 21,504 次/4,128,768 bytes，其他来源为 5,919 次/10,069,920 bytes。这些是一次固定轨迹的计量，不是全模型/全配置上界。

本轮先去掉历史采集等路径中可独立消除的层目录复制。此前每次判断某层是否存在，都调用按值返回的 `get_layer_ids()`，临时分配并填充整个向量；在逐层捕获中反复发生。现在适配器构造时只保存一次真实缓存的完整物理层 ID 和原顺序，后续查询、容量计数和诊断都复用它；其容量计入 `host_bytes()`。不能直接用仅含主模型 attention 层的 historical 配置列表替代，因为实际 KV 缓存可能包含额外草稿层。新增测试对照真实缓存目录，覆盖负值、缺失层和有效层，在禁止普通 C++ 分配的区间重复查询。

相同 96 次请求复验后，计数为 228,417 次/20,582,408 bytes；恰好移除全部 21,504 次层目录分配，减少 8.60% 的调用及 4,128,768 bytes（约 3.94 MiB）的累计申请，其他来源的次数和字节数完全相同。没有用这个数字宣称 RSS 降低或吞吐提升。剩余 228,417 次分配仍存在，**完整预约尚未完成**；下一项主要来源是调度器的临时依赖表，随后是有明确长度上限的 CUDA MoE 匹配容器。处理它们须保留图依赖和融合判断，不能通过关闭正确性检查或改动 GC 删除策略让计数归零。

首次完整模型回归在前六个场景通过后，于零基场景 6（AfterSubmission）中的后续 decode 报 `cudaStreamSynchronize: an illegal instruction was encountered`，进程退出 134。此前已经输出 Host 提前释放、GPU consumer 字节不变的检查通过，但这不能抵消后续 CUDA 失败；它也不能未经定位与旧的 illegal-memory-access 记录归为同一根因。保留首败后，分别重建 `d9934a5` 产品代码基线（含未启用的同一诊断 hook）与候选版，完整模型对照两者均退出 0，覆盖 43 个场景、reset/payload quota、两组重复请求、Closed capture 和 backend guard。准确结论是“候选首次失败、后续完整复验通过”，不是“异常已解决”或“已证明与本轮改动无关”；CUDA 根因仍是生产验收未关闭项。

本轮证据目录为 `artifacts/historical-allocation-audit/`，远程为 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-allocation-audit-d9934a5`。采样基线源码包 SHA-256 为 `39d7e50e9a49c7b1428687bbd83b6559179685a4c152083e15962a8abce25f09`，候选源码包为 `7a1bcbf382fc3be9985f5bb805e0ddc6be993d8661b7ebd25fb01eadd35f3827`，完整日志/trace/对照归档 `evidence.tar.gz` 为 `6cafb89cbe1f6d487d9d5516b6bcdd64e6efdfc917c4fad70a8881f6cb00c6a1`。本轮没有修改 GPU 容量重选的 active 边界，也没有解决此前长回答压力失败。

### CUDA 场景重复诊断与重置长测（2026-10-20）

为定位首败所处阶段，模型测试新增 `--scenario-range FIRST LAST ROUNDS`：零基、首尾均包含，轮数限制为 1..1000。每次执行重新设置该场景的配置，避免上一轮 f16/recency 参数污染下一轮起点。设置 `KVMEM_TEST_MODEL_TRACE=1` 后，在 decode 进入/返回、同步返回、harvest 返回和 commit 返回处输出阶段记录；场景还有开始和 context 关闭记录。没有为诊断增加产品同步，也没有改变 GC/GPU 生命周期或删除条件。

在同一产品版本 `4cb3f16` 上执行两组定向复现：

- `--scenario-range 5 7 20`：60 次场景全部关闭，400 次已记录的 decode 均有后续同步、harvest 和 commit 返回，backend guard 通过。
- `--scenario-range 0 7 8`：64 次场景全部关闭，272 次已记录的 decode 均有完整返回记录，backend guard 通过。该组包含原失败前缀，特别保留场景 4 的 q8_0 与场景 5/6 的 f16 context 切换。

两组驱动退出码均为 0；按完整日志校验轮次/场景顺序、关闭记录、阶段计数和错误标记，未发现本次重复中的 CUDA 失败。五种非法范围/轮数在模型加载前拒绝。准确结论是“124 次定向执行未复现”，不是故障已修复，也不是完整生产验收通过。证据为 `artifacts/historical-cuda-repro/evidence.tar.gz`，SHA-256 为 `4a8b6c9b74b6a202aae3f40e7ec1626ede2980cb8956e6cb30b6af52d774546e`；本地 `summary.json` 保存校验结果。

此前 Compute Sanitizer 日志明确报告 WDDM debugger interface 初始化失败，不能作为业务 GPU 正确性检查。只读核对发现 Windows 主机 `DESKTOP-OH21P4S` 的 `HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface` 不存在，当前 SSH 对应的 Windows 用户令牌不是管理员。[NVIDIA 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior) 要求该值为 DWORD 1。已准备 `scripts/windows/enable-cuda-debug-interface.ps1`：强制校验机器名，先保存原值，再设置并核对该值；脚本仅通过语法检查，尚未执行。**用户已明确暂时不改主机设置**，因此不执行该脚本，也不修改注册表、TDR、驱动或重启主机；专用检查器路径暂缓，继续现有环境下的定向复现和日志排查，原始 CUDA 故障保持未关闭。以后只有用户重新授权该设置时，才恢复接口配置及小型有效/故意越界 CUDA 对照验证。

在等待设计边界确认、调试接口设置暂缓期间，继续既定的一小时 reset/cancel 轨迹：同一个模型/context 中交替正常结束、prefill 取消和生成取消；保留旧 CPU reader 跨 reset 和新域首轮 decode，随后验证旧域后台退出。使用已编译的同一测试二进制，固定 8 槽、64-token 输入预约、3-token 输出上限及 48 MiB 跨域 payload 预算。驱动保存二进制/关键源文件指纹、实际 PID、模型日志、进程 RSS/HWM 和设备级显存采样，4200 秒上限避免失去终态。

该长测验证生命周期与资源复用，不涉及尚未定型的容量重选语义，也不替代编码质量或同域持续追加验收。只有测试进程成功退出、至少 3600 秒、旧域全部结清及 backend guard 通过，驱动才记录该项通过；设备级显存采样不能直接归因于本进程，累计分配字节也不能当作保留量。远程证据目录为 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-reset-hour-4cb3f16`，状态和结果保存在本地 `artifacts/historical-reset-hour/`。

**本次终态：** 产品基线 `4cb3f16`、诊断测试 `f59a47c`，模型加载后的轨迹持续 **3600.244 秒**，含加载和退出的驱动总时长 **3668.261 秒**，进程退出 0。13,303 次请求由 4,435 次正常结束、4,434 次 prefill 取消和 4,434 次生成取消组成，共处理 727,243 token、生成 17,739 token。13,302 个旧域全部结清，pending=0；测试结束采样时当前域仍由 context 持有，不能把该时点解释成所有 payload 已归零。backend guard 也通过。

完整日志的 355 个生命周期样本和 365 个进程/设备样本已用检查脚本复核。记录到 6 次接纳前拒绝，与共享 payload 预算的 refusal 计数相符；测试调用方重试后继续完成，不是已经接纳的请求等待 GC。检查点记录的 payload 峰值为 **36 MiB**，共享额度为 **48 MiB**；这是采样峰值，不是连续追踪得到的全程最高值。清理 ticket 占用采样为 2–5/64，块表、位置容量、arena、输出 buffer 和图 metadata 的记录容量保持不变。

模型加载后的进程 RSS 采样范围为 **806,204–820,472 KiB**（约 787–801 MiB）；进程生命周期高水位 **18,532,308 KiB** 包含模型加载。前后十分钟中位数相差约 2.34 MiB，描述性回归斜率约 1.22 MiB/小时；这些数据没有预先约定的泄漏判定阈值，不能据此证明完整内存上界或“没有泄漏”。2026-10-20 08:49:01 UTC 起另有最多 2 个并行编译任务准备独立的 CPU 侧 ASan/UBSan 诊断构建，因此本轮不能作为无竞争的吞吐/延迟基准。没有并行启动第二个模型负载。

原始证据归档为 `artifacts/historical-reset-hour/evidence.tar.gz`，SHA-256：`1193baa9b4d1d370d671f2d7109a392b0b5f723214af7a4a26b137dd98636aef`；解包目录的 `analysis.json` 保存复核结果。结论仅是该 reset/cancel 生命周期轨迹通过，CUDA 首败根因、实际编码质量和完整资源预约仍未关闭。

完整证据目录可用 `python scripts/analyze_historical_reset_soak.py EVIDENCE_DIRECTORY` 复查；须包含 `model.log`、`resources.jsonl`、`summary.json` 和 `driver.exit`。脚本核对每个样本的请求分类、token 数、旧域结清关系、额度和固定容量，以及终态退出、至少一小时和 backend guard；输出 `analysis.json`。`--partial` 仅作中途观测，不判定通过；缺少终态证据或使用会关闭断言的 Python `-O` 均拒绝验收。该脚本不代替完整生产质量与内存上界审计。

### 首批配置的资源上界核算（2026-10-20，尚未完成总量验收）

本节核对 `f59a47c` 的受限 RAM 路径和上述 reset 长测初始化日志，区分已有容量约束与仍缺失的总量证明。长测终态见上一节；较大上下文的编码请求须使用自身冻结参数重新计算容量。

- **历史 K/V payload：** `HistoricalKvPayloadBudget` 在实际 vector 分配前扣账，底层 `deallocate` 完成后才还账；准备失败也经 allocator 退账。令 `N` 为历史槽位数、`P` 为一个 full-slot 的各层 K/V 容量，当前 recency 无 Mean-K payload，额度为 `2 × N × P`。reset 复用同一预算对象。本次 `N=8`、`P=3,145,728 bytes`，新旧域及私有预约共用 **48 MiB**，不是每次 reset 重新获得 48 MiB。每请求最多 67 token、5 次追加，按碎片上界准备 6 槽，初始实际取得 **18 MiB**；未释放的旧 payload 仍占额度。该账本不含暂存、metadata 或模型内存。
- **搬运与重排空间：** `kvmem_capture_workspace_reserve` 在首个请求提交前实际分配固定 slab、描述符、事件和 layout scratch，失败撤回，接纳后禁止扩容。初始化日志记录 GPU workspace **34,275,328 bytes**，其中已包含 layout 的 **458,752 bytes**，不能重复相加。D2H 的两个 page-locked slab 合计 **64 MiB**：即使 `historical_pinned_bytes=0`，这份执行暂存仍存在。另有 Host 队列 **2,424,832 bytes**、同步 fallback **32,768 bytes**、stage-in K/V buffer **65,536 bytes**。这些执行空间跨 reset 复用，与被 GC 删除的历史数据属于不同分配。
- **有限请求记录与块目录：** 当前域的 repository/payload-store 槽数组、dense block 表及位置索引按 `N` 准备；目录 map 节点在私有预约与已安装目录之间转移，墓碑原位复用。manager 的 round、observation、retry、代际数组和 LRU candidate 数量受仓库容量限制，decision buffer 受 batch 上限限制。仍须核算实际 capacity 与控制块开销，不能只统计有效目录项。日志中的 block 表 **1,312 bytes**、256 行位置容量、capture arena **47,104 bytes**，只覆盖其中列出的项目。
- **旧域与后台清理：** `HistoricalKvCleanup::process()` 的 64 个 ticket 槽位在接纳前取得，物理析构后返还；新域、请求预约、交接和最终清理共用它。64 是任务容量，不能写成“允许 64 个完整域”。旧 reader 持续占用时，payload 预算或 ticket 不足可以令后续请求在接纳前失败；退休和提交清理都不等于释放。下文零 payload 的真实模型故障注入进一步验证：有限请求完成、unused 清理结清后，每个保留域仍占 metadata 与最终 CPU disposal 两个 ticket；新 reset 需要同时取得三个 ticket。此轨迹最多保留 30 个旧域加当前域，62/64 时下一次 reset 拒绝。构造未接纳的新域时还会短暂存在私有对象，须计入峰值；旧域的 runtime、repository、目录、manager 及线程/栈的完整字节总账仍未完成。
- **后端与进程总量：** 本轮初始化的输出 buffer 为 **38,895,616 bytes**，10 个图结果的 metadata 合计 **19,778,080 bytes**；测试检查其身份和容量不变。这不覆盖整个 scheduler、CUDA 图/驱动、模型映射、allocator 保留量及所有构图临时容器。进一步只读核对当前 ggml CUDA 后端，`cuda_graphs` 以首个 tensor 地址为键，访问时每 5 秒扫描、删除至少 10 秒未使用的条目；没有直接设置条目数上限。10 个 `llm_graph_result` 不能直接换算成 10 个 CUDA graph 实例，仍需核对实际键集合与图内资源。后端 buffer 未经归属核对不能直接相加为 Host RSS，普通 C++ 累计分配字节也不是当前保留内存。

当前可分别说明 payload 额度、若干 workspace 的实际预约和固定容器的结构上限，**尚不能给出整进程、全部新旧域及后端资源的完整数值上界**。继续补齐账目和失败路径，通用推理框架全面零分配改造按上文约定暂不扩展。`host_bytes()`/对外 store 字节接口只是约定范围的占用统计，不能替代这项完整验收。

### 修正 Historical payload 的对外占用统计（2026-10-20）

资源核算发现，`llama_memory_kvmem::host_bytes()` 原先累加 legacy Raw、runtime 和部分适配器容器，却没有加入独立 `HistoricalKvRawStore` 的 K/V payload。因此 `llama_kvmem_store_active_bytes()` 和 `llama_kvmem_store_bytes()` 会漏报当前预约及 reset 后仍由旧读者保留的数据。共享 payload 预算本身此前已扣账；此问题影响对外占用统计，不表示旧代码没有执行该预算约束。

修复直接加入既有共享预算的 `used_bytes()`，只加一次，覆盖当前私有预约、已安装 payload 和所有尚未释放的旧域。额度随实际底层释放归还，退休、请求结束或 reset 不提前扣除它。没有预算的组件级存储使用当前 store 的 payload 计数作为后备；公开受限请求路径始终配置共享预算。本项没有调整 GC 调度、退休权限、GPU 工作集或实际分配大小，也没有把该接口改称完整进程 RSS。

验收先只更新测试、保持原 `libllama.so`/`libggml-cuda.so` 指纹不变：真实模型 `--reset` 退出 1，准确触发 `host accounting omitted current historical payload`。随后加入产品修复，同一模型测试退出 0，覆盖：

- 当前历史 payload 被计入；11 个旧域 reader 加当前域占满 **36 MiB** 的该 fixture 额度时，共享 payload 未被漏记，两个公开字节入口结果一致。
- 归还旧 reader 不在前台销毁 payload；后台物理释放旧数据后，在当前 runtime 不变的情况下，报告值精确减少 **33 MiB**，没有重复扣除。
- 原有取消、reset、容量不足回退、暂停旧 GC、旧数据读取隔离和后端扩容拒绝检查仍通过。

这组 fixture 使用 6 个物理槽、36 MiB 额度；上一节一小时轨迹使用 8 槽、48 MiB，两组容量不能混报。长测运行在修复前的产品版本；本项是针对统计变化的新旧对照。原始日志、源码包和二进制指纹保存在 `artifacts/historical-host-accounting/evidence.tar.gz`，SHA-256 为 `0d3bf2e30faddeeea1b238f07beba53ecba6ccdd0a6406a040d52f6d88d79828`；三个本地修改源码的 SHA-256 与受测修复版逐一一致。

### 不修改主机设置的 CUDA 宿主检测（2026-10-20）

用户暂缓 Windows 调试接口设置后，在隔离目录复制 `f59a47c` 源码，独立构建真实 CUDA 适配器及模型测试。使用 GCC 11 的 ASan/UBSan、RelWithDebInfo 和 `sm_86`；C/C++ 及 `.cu` 的宿主部分均启用检测。源码全量 manifest 在构建前后核对，实际链接记录与库路径已检查：本次 `libkvmem.a` 以静态方式进入独立 `libllama.so`，其它 ggml 库也来自该独立目录。没有替换原来的普通构建。此快照刻意保持原 CUDA 定向复现的版本，早于 `30e1e8b` 的占用统计修复；不把结果称为最新产品的完整回归。

最初配置因 SSH PATH 没有找到 nvcc 而失败；显式指定 `/usr/local/cuda-13.2/bin/nvcc` 后构建完成。首次模型诊断在 CUDA 初始化时即报 `out of memory`，随后模型退到 CPU，在场景 0 的 pinned fixture 失败，未完成任何场景；退出还报告 5 项共 50,212 bytes 的泄漏。失败后可用显存仍约 15 GiB、系统可用内存约 61 GiB，不能把这个错误直接解释成历史 KV 预算耗尽。

为区分检测环境与产品问题，使用不包含模型或 KVMem 的小程序，依次执行 CUDA 初始化、4 KiB GPU/页锁定 Host 分配、H2D/D2H、CPU 字节校验和释放。普通版本在检测前后均退出 0；默认 ASan 版本复现相同的初始化错误及 50,212-byte 泄漏报告。[NVIDIA 的 CUDA/ASan 兼容说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting-analysis-tools.html#issues-with-cuda-and-address-sanitizer) 建议 `protect_shadow_gap=0`。仅对测试子进程采用该参数后，小程序的全部 CUDA 操作成功；退出仍报告 `libcuda.so.1` 中 260 bytes 与 40 bytes 两项泄漏、退出 1。故意在 CUDA 操作后进行 Host 释放后访问的阳性对照仍准确报 `heap-use-after-free`，UBSan 非法移位阳性对照也有效。没有关闭 heap poisoning、访问检测或泄漏检测，也没有使用 suppression。

随后复用**同一二进制**，保留首次失败日志，另目录运行以下配置：

```text
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:protect_shadow_gap=0
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
LSAN_OPTIONS=
KVMEM_TEST_MODEL_TRACE=1
kvmem-historical-access-test MODEL --scenario-range 0 7 1
```

这次 CUDA 正常识别设备，原失败前缀的 8 个场景全部开始、执行并关闭，8 个场景检查及 backend guard 均完成；34 次 decode 均按顺序记录 decode 返回、同步、harvest 和 commit 返回。包含原先 q8_0→f16 context 转换只是复现前缀，不表示扩展首批支持配置。日志未报告 Host 访问错误、UBSan 错误或新的 CUDA 错误，但进程最终仍因两个共 **300 bytes** 的泄漏报告退出 **1**；大小、数量及库内位置 `libcuda.so.1+0x1025a` / `+0x1c4de` 与无模型对照完全相同。这个对照说明这些报告在没有 GC 时也会出现；不把它们抹去，也不据此宣称整个项目没有泄漏。

准确结论是：**选定模型路径完成了宿主检测执行，没有新的 Host 访问/未定义行为报告；存在与独立 CUDA 对照一致的退出泄漏，不能标记整次 sanitizer 验收通过。** ASan 不检查 GPU kernel 的设备访问或闭源驱动内部；原始偶发 CUDA 非法指令/访问仍未定位，性能、真实任务质量和完整资源上界也不由本次诊断证明。本轮没有修改 GC 产品语义、注册表、驱动、TDR 或主机启动设置。

本地证据位于 `artifacts/historical-adapter-host-asan/`：

- `evidence.tar.gz`：独立构建、原失败、编译参数、源文件与动态库指纹；SHA-256 为 `a7c855187372d3b677866ae9886ff10f44c2ae534dfd2511a12b1512340b0ad4`。
- `compatible-evidence.tar.gz`：无模型对照、阳性对照和兼容参数下的模型完整日志；SHA-256 为 `8844da57efa36482829a3fd2a0d417bb29f5003c423841fbc3ed23d9f4db3330`。
- `summary.json` 与 `compatible-summary.json` 保存原始退出码、场景/阶段顺序和每项泄漏栈比较。远程原目录为 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-adapter-host-asan-f59a47c`，兼容运行与小对照分别位于同级 `historical-adapter-host-asan-compatible-f59a47c` / `historical-cuda-asan-runtime-control`。

### 零 payload 旧域的清理容量验收（2026-10-20）

此前 reset 的共享预算用例保留真实 Host reader，主要验证旧 K/V payload 未释放时的 36 MiB 额度；它不能单独证明“payload 已经归还，但旧 runtime/目录/manager 尚未退出”这一分支的上界。本轮在 `tests/historical-access-test.cpp` 的 `--reset` 路径增加专门用例，使用真实模型生成每个域的 32-token K/V、完成请求，然后显式退休该块。真实 GC worker 先完成物理回收，再停在受控的策略调用屏障；这一暂停仅用于故障注入，模拟 worker 迟迟未退出，不是新增生产策略。

测试使用既有 6 个历史槽、32-token 块、每请求 32-token 输入与 0-token 输出，保留初始 36 MiB 共享 payload 额度。每轮确认 payload 已为零，再通过公开 reset API 接纳新域并完成一次真实 decode。旧 worker 全部保持暂停，旧 runtime/目录/存储 owner 的弱引用必须仍存活；测试最后逐一检查旧 runtime 的块表容量没有变化、每个目录只有本轮的一个记录。容量拒绝还有 5 秒 watchdog：若 reset 错误地等待旧 worker，测试先打开屏障以避免无限挂起，再判定失败。

真实模型运行退出 0，得到以下结果：

- **独立容量限制：** 30 次 reset 成功后，30 个旧域加 1 个当前域、31 个暂停的 GC worker 共占 **62/64** 个 cleanup ticket；所有历史 payload 此时为 **0**。下一次 reset 无法取得完整的三个新 ticket，在接纳前拒绝，私有准备取得的名额被撤回。共享 payload 的 refusal 计数保持原值，证明这次拒绝由清理容量触发。
- **拒绝不替换当前域：** repository 身份、未关闭状态和 32-token 位置保持不变，已回收的代数仍保持回收状态。已接纳的新请求均在旧 worker 继续暂停时完成，测试没有用前台等待 GC 来获得容量。
- **后台结清与恢复：** 打开屏障后，30 个旧域的 repository/storage 弱引用全部过期，清理占用恢复为当前域的两个 ticket。随后 reset 再次接纳、旧的当前域也销毁，新请求 decode/finish 成功；共验证 31 个被保留域最终结清。新当前域的 payload 为一个 full-slot，3,145,728 bytes。
- **固定结构：** 所有被保留域的 store 块表统计均为 **1,096 bytes**，适配器的 row buffer、行容量与 capture arena 复用原分配。1,096 bytes 只代表该 store 统计范围，不能当作整个域的全部内存；本用例不证明 CUDA/驱动、线程栈、allocator 或整进程 RSS 的完整字节上界。

数量限制也与源码一致：在这一请求已完成且 unused 清理结清的轨迹中，一个域持有 metadata 与最终 CPU disposal 两个 ticket；创建下一个域需要 CPU disposal、domain metadata、request reservation 共三个。当前容量为 64，62 已占用时剩余两个不足以完整接纳新域。准备期间的私有 runtime/目录会短暂共存，失败后同步撤回；这个瞬时量不等于新增一个已接纳域。该结论针对当前串行 reset 协议，不能将 31 当作全部入口/任意任务组合的通用会话配置上限。

受测产品为 `30e1e8b`：编译前后核对 `libllama`、`libkvmem` 和四个 ggml 库的 SHA-256，六个产品库均未改变，仅更新测试程序。原有 reset/cancel、Host reader 隔离、共享 payload 额度、占用统计与 backend guard 检查仍通过。新增测试源码 SHA-256 为 `748b0643fd422ca0e8665ee0fa32f41a6270d8ed6a9e477dcb955aed847774e9`；二进制为 `4481cf6b8a67cfffbc91bbb0dd400c02a98d4a4b439c1b874ae79d06b93dbe3a`。

原始日志、旧/新测试源文件、产品库核对和实际命令保存在 `artifacts/historical-reset-metadata-bound/evidence.tar.gz`，SHA-256 为 `bbcce05543a0cf4378bee488f2088acfec5308233b864c3aa19940cbe28517fc`；`summary.json` 保存本地复核结果。远程目录为 `/tmp/kvmem-gc-99bef27.3CD9MU/historical-reset-metadata-bound-30e1e8b`。本轮补齐了一条资源边界证据，没有改变生产 GC 行为，也未关闭实际任务质量或原始 CUDA 故障。

## 剩余生产接入与完整验收

以下是导师方案仍未满足的完整待办，不是当前已支持功能，也不是本轮全部开工的清单。影响首批冻结配置的部分随验收修复；可选策略和配置扩展在首批通过后逐项安排：

1. **生产身份创建与请求 owner。** packed K/V、Mean-K D2H capture、同步 fallback 和 decode accumulator 已使用执行前捕获的同代 writer/范围，最后一个必需分量写完后自动发布；短尾块、非对齐范围和批量身份登记已有接线。CLI 的单请求 recency 路径现能自动预约 RAM、创建身份、向任务交接 producer，并显式运行 shadow/delete；适配器也已支持同一域的有限连续请求。server 已接入单 lane 的有限预约、精确前缀延续、新会话与健康取消；仍需扩展缓存分支、设备故障恢复和其它生产路径。这个受限入口没有承诺整请求的全部资源，更未覆盖所有调用路径。
2. **所有 active 与资源依赖。** 给 reselect、prefill、same-query、all-fit、commit-resident、keep-selected、attach 等入口建立新的 ActiveId/ExecLease；保护已经提交的 GPU 输入到实际完成。真实执行才更新 LRU；评分、预取与单纯选中不计实际使用。受限 recency/RAM 路径已拆分 Host 与 GPU/metadata；packed K/V 和 Mean-K 的读取、已完成 writer 分量及回收已独立结算。独立 NVMe 与 pinned 来源均已接到受管理 stage-in；historical pinned 副本已有精确代际、producer/reader 许可、真实内存 owner 及独立回收确认。legacy CPU cache 与新历史域的通用整合、批量异步搬运和其余 active 入口仍未完成；仍须验证内容同步不会引入前台 GC 等待。
3. **请求预约与长期有界。** RAM 槽位、payload、scratch、slot_info、微批次数据、受限输入整理、单个解码上下文对象、后端输出 buffer 和图结果元数据的预约及碎片上界已有资源组件；还需在请求进入 active 前统一取得 GPU、其余后端与构图对象、元数据和 rollback 等全部资源，并落实额度约束及失败处理。不能等 GC 补给；RAM API 仍拒绝分段 streaming 预约；server 的 SSE 按有限输出整请求预约。将生产 dense block 表及版本、评分事件、墓碑、pending decisions 一并改为有界；累计追加至少 10×仓库容量验证完整链路复用与计数守恒。
4. **会话、检查点与支持组合。** 显式 reset/new-begin 已能切换独立 incarnation 并后台结清旧域；已接到受限 server 的取消/新会话入口；仍需将普通 clear/truncate/关闭、前缀回退统一到相应所有权协议。闲置 checkpoint 失效不能复活历史，已承诺 rollback 保留实际依赖。NVMe tier 的 allocation generation、提交前 IO lease、独立 historical 副本绑定、真实 extent 预约及受管理 stage-in 来源已接通；pinned 独立副本也已接到受限公开入口。仍需完成 NVMe 公开配置、批量搬运依赖、新旧域共同的总预算及相应验收。MTP、hybrid、query replay、refresh、多 lane 等按实际使用组合逐项接入与验收，未验证组合明确列出。
5. **策略开关、观测与效果。** 已有 predicate 策略模板和版本条件原子提交；仍需实现独立冗余策略，并将受限 CLI/server 已有的 off/shadow/delete、LRU 压力与 pending credit 扩展到其余生产入口。策略决定不受 manager 隐含年龄/驻留/引用门槛限制。保存 retirement/release/ActiveId/延期原因 trace、allocator 与 metadata 长轨迹、入口拒绝和前台 GC 等待计数；补完整生产组合的 TSan 结果。固定模型、seed、任务和容量，对 off/shadow/delete 及 LRU/新增策略做延迟、吞吐和真实 agent 质量对照，保留原始日志。质量阈值未约定时只提交比较结果，不宣称通过。

单 GPU、单 lane、文本、RAM、MTP off 是首个生产闭环的阶段验收范围，不是完整交付的替代定义。目前 repository、按句柄 payload store、预约追加块表、宿主持久域以及受限 llama 适配器入口具备固定物理槽位跨请求复用；server 的多 lane/多会话、跨请求检查点引用及其余队列仍需改造。当前结果不证明生产长期总内存有界，也不证明所有已接纳生产请求都完全不依赖 GC。

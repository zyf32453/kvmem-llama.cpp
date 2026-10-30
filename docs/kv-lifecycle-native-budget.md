# B4 原生预算、生产者暂存与健康接线

日期：2026-10-31。父提交 `d311d1b8c7931c1354e323fb0fcd6d5813f1e108`。本批原生接线、构建、主机与真实模型功能验证已完成；阶段 B 的剩余门槛及后续 C–F 仍需继续，完整生产 GC 未完成。

后续更新：本文保留第 25 补丁的原始验收归属。第 26 补丁修复 decode 额度不足的请求清理，并完成本文列出的活动停止及持续轮换服务门槛，见 [服务故障、停止与旧域轮换](kv-lifecycle-service-failures.md)；不能将后续结果归属给本文的旧服务二进制。

## 原生接口与资源归属

第 25 个 NInfer 补丁 `historical-gc-lifecycle` 在原有 24 补丁后增加：

- `--kvmem-gc-aux-mib N`：独立、正数的 GC 辅助额度。启用 LRU/redundancy 时默认 1024 MiB；没有策略、值为零或字节换算溢出时拒绝。该默认值是本候选的工程配置，完整生产默认值仍须根据验收决定。
- 每个 Program/model 一个共享账户，供该 Program 内的新旧 window、GC、source、range 和临时数组使用，不按每次重建重新给一份额度。KV payload H 继续独立计费；多模型进程总额及线程/其他引擎工作区不由这个账户代管。
- bridge、策略及 source/range 对象使用计费工厂或 allocate_shared；retired/protected 位图、页面描述符暂存、GC 专用 Mean-Q 分组计数、均值/范数/key/score 暂存也计费。普通召回已有的引擎评分结果只借用读取，不为交给 GC 再建立无账户的中间副本。
- 新 GC owner 图与线程启动先在局部完成，成功后才发布到 window；失败时不留下半个新 source/manager。位图迁移保留原内容，在两个新容器均完成后替换。
- `/props` 的 `kvmem` 配置增加 `gc_policy` 和 `gc_aux_budget_bytes`；usage 增加 `gc_aux_budget_bytes/gc_aux_bytes/gc_aux_peak_bytes/gc_aux_denied`。计数是 allocator 请求及保留容量，不是 RSS。

首批 GC 仍为单 GPU、单运行请求、RAM 历史。原生解析器与 Engine normalize 都拒绝 GC+SSD/多 GPU；不把未验收的组合默认放行。本批没有实现 E 阶段的分层或多会话能力。

账户自身及其 shared control block 在 Program 建立时分配一次，不从自身额度递归计费。它们是每个 model 的固定基础开销；GC 线程运行时/栈、libc 及其他引擎工作区也不属于 M。用量报告不能解释为这些资源也已包含在 1024 MiB 中，后续长运行验收须另行记录进程与线程的实际峰值。

## 健康与错误

公共 manager 提供 Healthy、PolicyFailed、WorkerFailed、Stopped 四类状态。失败原因通过已有原子计数发布，stop 不清掉原故障诊断。线程启动失败会保留失败状态，失败或停止的实例不能静默 restart；需要新的 owner。

失败/停止后，新容量需求与新边界批次关闭。policy 失败后仍可回收已退休资源。已经在故障发布前取得提交权的批次可完成原许可下的提交；原生 bridge 先完成该批 tombstone/Host detach，再向请求暴露故障，不发布半个新工作集。并发销毁 manager 不在此契约内，销毁仍须与执行 owner 串行。

原生入口检查 GC 健康，prefill 的既有两秒容量等待会直接收到策略/worker 错误，而非将其伪装成普通空间不足。新的辅助预算不足及 GC 不可用经 Engine 的请求清理路径转为 `RequestErrorKind::Unavailable`；非流式响应为 503，已开始的 stream 使用终态错误。真实模型已验证两次非流式辅助额度不足均返回 `503 service_unavailable` 和明确的 auxiliary budget 原因，随后短请求成功。流式错误协议仍沿用既有入口，本批没有把非流式负例当作流式故障注入证明。

这里没有新建一套 worker 重启或完整旧域调度器。旧 owner 的真实资源继续按账户保留；请求取消、持续轮换、停止期间批次和线程运行时资源的完整服务验收仍是 B 的剩余门槛。

## 验证范围与当前状态

Windows Debug 20 项契约通过，新增/扩展的检查包括：部分策略输出后抛错不得提交、unsupported 条件可见、关闭后 prepared 批次不得改变可见性、旧退休资源继续释放、真实线程启动分配失败的粘性诊断，以及 Mean/Max 评分在辅助账户下与既有结果/独立 FP64 oracle 相符。验收检查器 10 项测试通过，新增辅助额度缺失/零/负数/bool/浮点/溢出拒绝。

同源 Linux Debug 10 项 ASan/UBSan 通过。TSan 本批没有新结论，之前环境限制仍在。补丁树与源文件清单已校验；原生 CUDA/服务构建在新的隔离目录中进行，旧验收源码及模型未修改。

另用原生 `compile_commands.json` 的实际 `-O3 -DNDEBUG -std=gnu++20` 参数编译两个主机探针，并链接本次构建已经生成的生产 `libkvmem.a`。Mean/Max 评分和 GC 健康探针均退出 0；测试前后核心 archive 哈希未变。这补充了优化构建的主机证据，未运行 GPU，也未启动第二个 Ninja 或修改原生构建目录。

原生 Release/CUDA 13.2、sm_86 构建完成，5 项选项/容量/默认配置测试通过。新服务 SHA-256 为 `a2a5f55b2a706c0d5c2142ee864deee748b48b2e4adcc42a54fd680f8631a5d0`；只对这个二进制使用下述 GPU 证据，没有沿用旧服务记录。

在固定 Qwen3.8-27B 模型、RTX 3080 Ti Laptop 上，LRU/redundancy × 普通解码/MTP4 四组均成功处理 1024 条记录并给出正确答案 599。每组 H=96 MiB、M=1024 MiB，B=2048、R=512；普通两组分别实际回收 397 页，MTP4 两组分别 400 页。冗余组既发生带精确 witness/score epoch 的真实冗余提交，也发生显式容量兜底；逐条检查 committed redundancy ACK 的证据、代表身份、版本和相似度。四组均无辅助额度拒绝，终态 policy/worker failure 为零。该压力输入的 context=262144，诊断与传输校验开启，属于功能组合，不能冒充冻结微型性能档位或 Agent 质量结果。

运行中拒绝使用相同 H=96 MiB、M=8 MiB：服务完成预热后，两次长输入因辅助额度不足结束；每次清理后 active request/reservation 为零，denied=2，最高请求计费 8,355,596 字节，没有超过 8,388,608 字节。随后短任务仍回答 599。这里没有把 KV payload 的 H 上限当作辅助 M 的上限。

LRU/普通和 redundancy/MTP4 分别验证 prefill 与 decode 期间 TCP 断连，共 4 项。prefill 由已执行至少 512 输入 token、尚无 decode 的计数证明；decode 由至少 16 输出 token 证明。每次断连后执行 owner/预约清零、辅助与 Host 用量在各自额度内，后续短任务成功。单次观察到的清理时间约 0.05–0.63 秒，只描述本次样本，不构成延迟 SLA。

保留三个未通过的早期总检查：M=1 MiB/H=320 MiB 和 M=8 MiB/H=320 MiB 在默认预热时已触及额度，未进入预期 HTTP 场景；之后用已测得预热成本的 H=96 MiB 运行中场景。另一次 LRU 中断总检查误读最后一个短请求 owner 的零 GC 计数；其两个断连子检查实际通过，日志也证明先前 owner 已回收。修正脚本明确单个 owner 计数口径后执行尚未运行的 MTP4 组合，没有把跨请求最大值写成进程累计数，也没有覆盖原失败。

构建环境最初下载 Python 3.11 镜像被网络阻断，已改用主机缓存的 Python 3.11 镜像，生成独立工具镜像；未修改主机代理、系统 Python 或调试设置。完整构建约 54 分钟。晚期容器 cgroup 采样记录截至采样时的 peak=61,914,112,000 字节，oom/oom_kill=0；这包含容器编译内存和文件缓存，不是推理 RSS 或运行时 M。原始构建日志及独立资源采样一并保存。

汇总见 [执行摘要](experiments/2026-10-31-lifecycle-stage-b-native/summary.json)。完整源码、运行脚本、日志、JUnit、全部失败尝试及固定版本元数据保存在 `artifacts/lifecycle-stage-b-native/evidence.zip`，SHA-256 为 `137810c95157d654ea54b333f34e0e3c432f2a0dd1542c2fe13eb7d31df79b7c`。

下一步完成阶段 B 的剩余审计与服务门槛：活动请求期间停止、持续旧域轮换/准入及更长资源边界运行；之后进入 C 的恢复与版本闭环。完整生产、性能和 Agent 质量门槛仍归 F；本批没有关闭先前偶发 CUDA 问题，也没有取得新的 TSan 结论。

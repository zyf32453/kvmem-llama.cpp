# NInfer checkpoint owner 总预算

日期：2026-10-31。C5 的首批 checkpoint owner 预算已通过原生与模型功能验收；阶段 C 和完整生产目标尚未完成。当前状态见 [六阶段进度](kv-lifecycle-progress.md)。

## 要解决的实际缺口

[C4 的真实模型检查](kv-lifecycle-native-forget.md) 观察到：H=96 MiB 时，5 个域的 checkpoint 仍可占约 1.54 GB；显式遗忘后还有约 0.92 GB。普通配置的 checkpoint pool 未配置有限额度，而目录扫描不覆盖所有私有/已脱离目录的 owner。不能据此宣称 CPU 内存整体有界。

本次复用既有 KvmemCheckpointBufferPool 和 checkpoint_memory_room，不增加第二个 allocator、快照目录或推理线程。`--kvmem-checkpoint-mib N` / `KvmemOptions::checkpoint_bytes` 提供独立于 H/S/R/M 的 C。启用 KVMem 时默认 1,024 MiB，是整个 Program 的总额，不按每条会话放大。可选 spread 策略原有 host cap 作为更严格的上界保留。

## 计费跟随真实 owner

- pinned image 按真实 pool allocation 计费；活动、恢复、临时 MTP rollback 和空闲 buffer 全部在内。复用同一个 buffer 不重复计费，真正 free 完成后才减 image 账。
- checkpoint 带有 move-only metadata charge；构造前先预约容量，构造后收紧到实际记录的 capacity。目录删除、shared alias 移动及未发布状态都不能提前退款；最后 owner 销毁后才结清。LRFU 替换需要更大的 attachment 时先增加额度，额度不足撤销可选旧 alias，避免留下不匹配的历史状态。
- MTP rewrite 的临时 prefix identity、digest、mean/graph/feedback 副本也有独立生命周期 charge，与临时 image 使用同一个 C。
- disk read 的既有单 buffer 在 C 内预留容量，实际 staging 用量另报。worker 取消后仍持有的 buffer 继续由原 HostKvStorage owner 计费；本批不因此宣称 GC+SSD 组合通过。

metadata 是容量计费，包含保守的 shared attachment 重复预算；固定 account/control、shared control block、allocator 和其他运行时开销在 C 之外。捕获前构造的固定模型布局描述符属于既有规划工作区，其规模由层数/plane 配置约束，不按源长度增长；它在预留 C 前存在，不属于本预算的分配覆盖。prefix、digest、mean、graph 和 feedback 的长度相关副本在相应 charge 生命周期内。这里不是进程 RSS 上界，也不等于所有后端工作区、源、KV 和权重都由 C 管理。

## 准入、回收与失败

必要捕获先检查 pool 自身账户，而非分配并扫描一份目录用量估计。空间不足时先 trim 空闲 buffer，再复用已有规则清理允许丢弃的 coverage/旧历史域，保留正在准入的恢复来源。已经脱离目录但仍被引用的对象继续占 C，不能拿它们的预期释放量当可用空间。

必要状态仍放不下时抛出有类型的 BudgetExceeded，由现有 Engine 有序清理本次请求；可选 capture 允许跳过。源 S 不随这类缓存清理自动遗忘，旧源仍可按独立恢复条件重建。不能把“预算导致 cache miss”写成保留承诺不受影响；较小 C 会减少复用机会，性能影响须另测。

`/stats` 的 `checkpoint_bytes` 表示 image + metadata +固定 read-staging 预约总 charge。另报 `checkpoint_budget_bytes`、`checkpoint_image_bytes`、`checkpoint_metadata_bytes`、`checkpoint_idle_bytes`、`checkpoint_peak_bytes`、`checkpoint_denied`、`checkpoint_staging_reserved_bytes` 和实际 `checkpoint_staging_bytes`。峰值包括构造及私有 owner 的保守预约，不是 RSS 峰值。

## 本批验收与剩余

候选输入为 [checkpoint-budget-candidate-v1](contracts/kv-lifecycle-checkpoint-budget-candidate-v1.json)，模型入口为 [ninfer_checkpoint_budget_acceptance.py](../scripts/ninfer_checkpoint_budget_acceptance.py)。新增 CPU contract 及其定向 ASan/UBSan 目标检查共享额度、移交、真实分配失败和脱离目录的 alias；既有 pool 模型测试还检查真实 pinned image 与 metadata 共用额度。

模型检查要求：1 MiB 必要捕获明确拒绝、没有半成品且同 Engine 可继续无缓存推理；384 MiB 下多次换出旧 checkpoint owner，源保留、旧 native 域确实被换出后仍可重建。原有 C3/C4 回归各显式使用 2,048 MiB C，满足它们保留多个目标域的前置条件；不降低物理页释放、源一致性或三代撤销的断言。

当前固定原生 tree 为 `ca0ce1b40686185f1020f8be9910ae523d2871d9`，第 30 补丁，108 项输入源码检查通过。共享核心/主机契约源码未改变，可复用其已归属版本的 22 项覆盖；新预算还必须有自己的原生和模型证据。本地额外用同一 pool 头配 CPU buffer stand-in 编译检查额度、alias、空闲池及跨线程 owner 结清，通过；这不是实际 pinned 分配、完整原生构建或 CUDA 结论。

SSH 恢复后确认原构建容器 exit 1、OOMKilled=false；现存日志显示写入 `/work/native-logs/build.exit` 发生文件系统 I/O 错误，原编译结果及临时工作目录未保留下来。不能把它记录为编译通过，也不能据此推断 OOM 或具体代码错误。恢复 Docker 运行连接后，在新的持久独立目录验证同一份 108 项输入并完整重建；首个恢复输入包遗漏三个根目录 probe 文件，在进入编译前即被校验拒绝，已补齐并保留失败。产品代码与验收断言未变。

新的持久工作目录完成全部 873 个构建步骤，容器 exit 0、未 OOM。使用固定镜像的 GCC 13.3、CUDA 13.2/sm86、Python 3.11.18；该主机此前同类原生构建观察到约 61.9 GB 的 cgroup 内存峰值，因此本轮用四个编译任务，完整冷构建约 90 分钟。中途一次观察到约 5.33 GB 的峰值、无 OOM 事件，这不是最终峰值或推理资源结论。等待控制任务的一小时时限先于编译结束到期；已保留超时记录，在确认没有任何 GPU 用例启动后延长编译等待，原构建未重启，模型每组 1,800 秒的时限、输入和断言未变。原有 CUDA 未使用模板警告保留，未做无关算子修改。

同一候选的 12 项原生检查全部通过，包含实际 checkpoint metadata 对象的共享额度、移交、脱离目录 alias、最后 owner 退出及三个真实分配失败点；定向 ASan/UBSan/泄漏检查通过，并核对实际链接 libasan.so.8、libubsan.so.1。这不是整个原生库或 CUDA 路径的全量 sanitizer 结论。22 项既有主机契约在逐项比对 89 个冻结公共 core/测试输入相同后复用其 C4 结果，没有记成这次重新运行。

同一 Qwen3.8-27B v3 模型、RTX 3080 Ti Laptop 16 GiB / 驱动 596.08 的七组 GPU 检查全部 exit 0、未 OOM：真实 pinned pool 一组；LRU／redundancy × 普通解码／MTP4 四组预算压力；规范源重建及家族遗忘各一组 MTP4 回归。模型检查前后均核对 3,440 个源输入未变。

四组压力检查均验证 1 MiB 必要捕获明确拒绝、没有半成品，Engine 仍可在同一实例完成无缓存后续请求。随后 384 MiB C 下执行八份不同前缀，每组实际回收八个旧域，原 source 仍在、旧 native 域不在；仅从源重建的八个输出 IDs 与原结果一致。普通解码峰值为 308,084,854 bytes，MTP4 为 308,089,986 bytes，均低于 402,653,184-byte C；其中 image 为 307,908,608 bytes，metadata 分别为 176,246／181,378 bytes。两组 MTP 重建实际各运行两轮投机。

2,048 MiB C 的重建回归仍保留原先更强的物理条件：32,768-token 输入实际退休 480 页，重建前缀 1,024 tokens 内有 15 个 64-token 页真实释放；重建与独立 fresh IDs、旧 native window 换出后的 source-only 路径输出相同，部分取消在 512-token prefill、reserve 拒绝和首 token 停止均通过。遗忘回归仍严格验证三代 source/native 域 3／3 撤销、忙碌拒绝、旧准备对象失效、R 延迟归还及无关源保留。

遗忘回归中 H 从 88,954,880 降至 26,808,320 bytes，C 从 1,540,423,578 降至 924,062,228 bytes；后者仍含无关有效 owner 和空闲池，继续计费，不代表所有 CPU 内存清零。较小 C 会降低可保留的 cache 数量；本批不提供吞吐、p95、长期或 Agent 质量结论。默认 1,024 MiB 的归一化/CLI 路径已检查，没有额外运行默认值的 GPU 专项；真实 GPU 档位按上述 1／384／2,048 MiB 归属，不能外推到其他配置。

详细结果见 [执行摘要](experiments/2026-10-31-lifecycle-stage-c-checkpoint-budget/summary.json)。原始输入、失败、构建、sanitizer、模型和复用主机覆盖证据为 `artifacts/lifecycle-stage-c-checkpoint-budget/evidence.zip`，SHA-256 `a28f8dd6910ac281aa2ee5c7a53f278a46122258dd8c630028c0cbf8d86bf592`。持久化 checkpoint/源归档、跨进程退休与遗忘 fence、长源重放、完整发布/回滚故障以及 D–F 门槛继续保留。

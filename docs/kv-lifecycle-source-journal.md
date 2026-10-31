# C1：独立 RAM 源记录

日期：2026-10-31。父提交 `e595b746ea2db1e9902c1d72f1fad606a470c501`。本批实现公共源记录组件，尚未接入 NInfer 的请求提交、恢复执行或持久化。阶段 C 及完整生产系统均未完成。

后续更新：本文保留 C1 的实现与证据归属。C2 已完成 NInfer 记录、配置/事件来源、原生配额及服务检查，见 [C2 原生接线](kv-lifecycle-native-source.md)；它仍未提供恢复执行或持久化，不能把 C1 的主机证据重新归属给原生 GPU。

后续 [C7](kv-lifecycle-source-import.md) 已增加 `import_image`：将受 R 约束的不可变镜像原子复制到 S，创建 computed=0 的当前新源；归档许可由 SourceArchive 当前读回执单独验证。C1 以下结果保留历史归属，不代表已接通 Engine 持久恢复。

## 源记录与 KV 的关系

新增 `SourceJournalStore`，复用既有 `HostKvStorage` 保存 token/event 数组，并复用 `SessionIdentity` 的身份格式。目录与派生 KV 的 repository、Host payload H、GC 辅助 M 独立：CPU KV 页真实释放后，源 token 仍可读取；读取源记录不会重新授权已经退休的 KV 身份。

每份记录保存接受的 token ID、配置身份、消息/工具结果/模型输出/控制事件的区间，以及两个不同的 frontier。例如接受 193 个 token、计算到 192 时，最后一个已接受的采样 token 也被保存，但不能声称其 KV 已存在。更新计算 frontier 不会截断原始源内容；它也不构成 KV/checkpoint 兼容证明。

配置字段是由 backend 产生的非零 256-bit digest，合同要求覆盖模型、tokenizer/template、位置/输入编码、KV/state schema 和执行配置。当前组件只保存及返回这个字段，实际 digest 的生成与验证在原生适配器接线；随机 fixture digest 不是模型兼容验证。当前 payload 是文本 token ID，未实现多模态资产、多轴位置行或任意稀疏执行轨迹的精确回放。

工具结果保存为已接受 token 及 `ToolResult` 事件。组件没有执行工具的入口；后续恢复应读取保存结果。尚未运行真实恢复，因此本批不宣称已经通过了“恢复过程不重复执行外部工具”的端到端验收。

## 版本、分支和遗忘

- `capture` 发布独立源身份；`append` 只接受当前 revision，成功后返回新 revision。旧 revision、其他 store 的引用以及已忘记的引用均拒绝。
- 追加只能延伸当前接受序列；不能修改先前 token。事件只能按顺序新增，或以同一 id/kind/begin 延伸最后一个尚未完成的事件，已完成事件不能被改写。
- `branch` 为明确的前缀创建新源身份，复制 token 和事件前缀，不修改原记录。截断到事件中间时，该分支事件标为 incomplete；新分支的 computed frontier 为零，没有凭空复制原计算状态。
- `forget` 显式删除该 store 中的目标及已记录的派生分支，先完成有界的后代标记再清理。复用目录空位会取得新身份，旧引用不能命中新记录。

这里的 `forget` 只覆盖本 store 和它记录的分支。它尚未撤销 backend KV、外部别名或落盘归档，不能称为系统级 `ForgetSource`。引用只有进程内意义，不是可以跨重启直接恢复的 durable ID。持久 fence、别名撤销和恢复事务仍属于 C 的后续工作。

## 配额与失败原子性

构造时显式指定 S、最大源记录数、每份最大 token 数和事件数。S 包含固定 Entry 数组及真实 token/event buffer 容量，计入几何扩容的空余容量与 old+new 过渡占用。原始数据按需要增长，不提前为每份记录分配最大上下文。

复用 `HostKvStorage` 的真实预约与最后释放退款。需要扩容两个 buffer 时，先一次预约两者，再完成所有可能失败的分配、验证和可写借用，最后发布。即便最终新 buffer 放得下，只要 old+new 过渡放不下，也会拒绝；不能先销毁唯一旧源记录来腾空间。

容量不足返回 `BudgetExceeded`，不会自动选择一份源记录永久遗忘。token/event/目录数量上限也明确拒绝。一般分配失败同样保留原内容、revision 和当前计费；实际拒绝的配额计数与普通分配失败分开。

S 的峰值计入已预约但尚未完成分配的空间。S 不包含 store/account 自身、`HostKvStorage` 的 allocation/control 对象、编译器/allocator 的固定记账开销、libc 开销或调用者的读出 buffer。这些对象数量有界：每份记录最多两个 buffer，串行操作另有最多两个暂存 buffer。它不是整个进程的 RSS 上界，也没有解决此前 checkpoint 的额外占用。访问方式目前是单 owner 串行调用、同步复制读出，不暴露异步 lease、裸 buffer 指针或未计费快照副本。

## 验证及下一步

[source_journal_test.cpp](../kvmem/tests/source_journal_test.cpp) 验证接受/计算边界、工具与生成事件、旧 revision、分支/部分事件、跨 store 身份、后代遗忘、目录复用、真实容量与过渡峰值拒绝，以及 capture/append/branch 的 12 个实际分配失败点。所有失败都检查旧数据与计费未被部分改写。

还将源记录与真实公共 `HistoricalKvPageGc` / `HostKvStorage` 结合：退休中间页，先由 reader 保留物理资源，reader 退出后真实释放为零，再核对完整源 token 仍在、退休页仍不可选。这个测试证明源记录和 KV 生命周期独立，不是从源记录重建 KV 的测试。

Windows Debug 检查与 Linux GCC 13 ASan/UBSan/LeakSanitizer 检查结果见 [C1 执行摘要](experiments/2026-10-31-lifecycle-stage-c-source/summary.json)。Linux 使用固定构建镜像和 Python 3.11，隔离目录验证全部源文件后构建，只运行新增组件的相关检查；没有重复使用旧服务二进制冒充新恢复验收。

Windows 共 21 项当前契约通过（包含本批新增的一个组件测试目标）；Linux 的新增组件测试目标覆盖上述六组场景。原始源文件、命令、sanitizer 日志、JUnit 与本地主机结果保存在 `artifacts/lifecycle-stage-c-source/evidence.zip`，SHA-256 为 `0644fd910fe6087b4b5b35bd8abab36116be565d0ac3d60eb7c72f46016e61b0`。此次没有运行新的 GPU、Agent、性能或 TSan 检查；先前 CUDA/TSan 限制继续保留。

下一项在 NInfer 的已接受前缀提交点接线：由真实 frontend/Program 数据生成配置和事件来源，发布源记录与正确 frontier，增加独立配额和用量观察，并覆盖取消、provisional/MTP 丢弃以及请求失败。之后恢复 planner 先生成明确计划、校验退休/版本 fence、预约资源，再执行及原子发布。保留 token 不等于有空间恢复；完整 attention 放不下时必须拒绝或显式选择允许改变执行语义的新域，不能隐式声称精确补回旧 KV。

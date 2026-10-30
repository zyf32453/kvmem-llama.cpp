# 源记录恢复规划与准备契约

日期：2026-10-31。状态：C3 的规划和源数据准备组件已实现并验证；NInfer 实际模型重建、物理资源预约与事务发布未接线，阶段 C 仍未完成。当前总状态见 [六阶段进度](kv-lifecycle-progress.md)。

## 本次完成的边界

复用 C1 的 `SourceJournalStore` 作为逻辑源目录、`SourceRef` 作为源身份、`MemoryStamp` 作为派生域版本，以及现有 `HostKvStorage` 作为实际准备空间账户。不增加另一套历史目录、checkpoint registry、身份分配器、异步线程或模型执行框架。实现位于 [source_recovery.hpp](../kvmem/include/kvmem/source_recovery.hpp) 和 [source_recovery.cpp](../kvmem/src/host/source_recovery.cpp)。

此前可以独立记录、读取源 token，但尚无统一恢复计划。现在必须先形成 `SourceRecoveryPlan`，声明源版本、接受前缀、当前执行域及配置、由现有 owner 分配的新身份、恢复模式与预算。计划通过后才能准备源副本；它本身不预约或发布模型资源。

`ExactState` 明确拒绝从源 token 自动回退。唯一实现的模式是显式 `RebuildAsNew`：在新身份下从头计算完整、连续的规范前缀。原来的 sparse attention、recurrent、MTP 或数值路径不因文本相同而被认定可逐位恢复。新身份必须与待替换域不同，且由 backend 的现有身份分配器保证从未使用；组件不另造一个分配器，也不能仅靠“不同”证明全局新鲜。

## 先检查完整容量

计划按现有原生页大小分别向上取整完整前缀和生成预留，再检查 selected/device 预算及逻辑上下文上限。例如 page=64、源前缀 193、生成预留 65，需要 256+128=384 个 device token slots。只有 256 个 slots 时，即使 prefill 每次只算 64 tokens，也明确拒绝。

这个首个恢复组件采用保守的完整规范前缀路径，长历史即使有源也可能无法恢复。它尚未实现按当前稀疏执行器重新选择历史的新域模式；若后续加入，必须声明语义、资源和执行证据，不能偷偷绕过本计划的容量拒绝。

预算数值来自调用方，而非物理预约证明。主/草稿 KV、recurrent/state image、scratch、尾页/COW 和 rollback 的实际可行性仍由 Program 原有资源规划和预约承担。`MemoryBudget.host_payload_bytes` 不会由本组件假装兑换成实际 Host/checkpoint 容量。必须在真实资源预约成功、模型执行完成并再次校验后，才能发布新域。

## 源副本的真实准备空间

`prepare_source_recovery` 先在调用方的 R `HostKvStorage` 中预约完整前缀副本的真实 bytes，再分配并复制。复制使用固定 256-token 栈缓冲，避免再创建一份没有计费的前缀向量。准备对象封装副本，不导出裸指针或 lease，只提供有版本校验的复制读取。

R 与源 S、历史 Host H、GC 辅助 M 独立。同一源的两份准备副本分别计费；对象移动不退款，最后实际 owner 销毁才归还。已被遗忘、禁止再读的准备副本在销毁前同样保留计费。R 只限制实际 payload；固定的 plan/对象、HostKvStorage account/control、allocator 及调用方读缓冲仍在其口径之外，不是进程 RSS 上界或 GPU 预约。

分配/复制失败没有源或模型状态变更，预约和已分配副本按真实生命周期回收。测试逐点注入本路径实际发生的两个 allocation failure，检查 R 归零、S 和原始 token 不变。

## 版本与遗忘边界

规划、准备及每次读取都会检查当前 `SourceRef` 的 store incarnation/revision，以及完整预期 `MemoryStamp`。原生 owner 在退休或内容变化时必须推进对应版本，不能拿旧 stamp 充当当前事实。即使只更新了 source computed frontier，旧计划也保守地失效。

每次校验还必须传入当前实际执行配置摘要，而非只读取计划中保存的摘要。当前配置改变会拒绝；源配置与当前配置不同也拒绝。本组件不提供跨 tokenizer/template/model 的源迁移。来源配置摘要只允许解释源行，仍不是派生 KV 的复用证明。

准备完成后局部遗忘源或其父分支，旧计划和已有准备对象都失去读取资格；槽位复用、另一个 source store 也不能让它们重新有效。这里仍是 C1 的**进程内局部遗忘**，不涵盖外部副本、已交给 backend 的缓冲、checkpoint/归档及跨进程别名，不构成系统级 `ForgetSource` 或安全擦除。

全部调用由同一 engine owner 串行完成。当前组件没有跨线程 publish 操作；backend 必须在执行和发布的安全边界重新校验，在最终校验与发布之间不得插入 owner 变更。已经复制源数据不意味着可以无条件发布模型状态。

## 验证结果及实际限制

机器可读候选为 [recovery-candidate-v1](contracts/kv-lifecycle-recovery-candidate-v1.json)，冻结 102 项输入和 27 个原生补丁声明。Windows/GCC 16.2 Debug 的 22 项主机契约通过；Linux/GCC 13.3 Debug 的 `source_recovery_test`、`source_journal_test`、`memory_contract_test` 三项在 ASan/UBSan/LeakSanitizer 下通过，实际链接了对应 sanitizer。完整结果见 [执行摘要](experiments/2026-10-31-lifecycle-stage-c-recovery/summary.json)。

新增测试覆盖六组行为：显式新域与完整容量；partial-prefix/空源/跨 256-token 块复制；七项执行域变化和配置/源 revision；遗忘后拒读及实际延迟退款；真实准备配额和两处分配失败；实际 CPU historical page 退休并最后释放后，源仍能形成独立恢复计划并准备精确 token，原退休页保持不可选。最后一组**没有执行模型恢复**，不能写成“删除后模型恢复已通过”。

Linux 的三项测试和 sanitizer 实际通过后，首次归档驱动误找 `build/bin` 中的二进制；最小核心 CMake 实际输出在 `build/kvmem`，原容器因此 exit 1。原始脚本、终态和失败保留；只修正复现脚本路径，并核对已经完成的 JUnit、真实二进制及链接库后补齐归档，没有将首次驱动写成整体 exit 0，也没有为改善结论重复测试。

本次没有重新构建或运行 GPU 服务，不把 C2 的 `919b18eb…` 当成本组件的原生重建证据；没有配对性能、长时间或新 Agent 结论。旧 TSan 环境限制、偶发 CUDA 问题继续保留。

下一步接 NInfer 既有 Program 资源规划：预约实际模型资源，执行隔离的新域重建，失败不发布半成品，成功后在安全边界校验和发布；随后补 checkpoint/归档的退休与遗忘 fence、持久恢复。只有实际模型删除后恢复及旧快照拒绝等完整条件通过，才能关闭阶段 C。

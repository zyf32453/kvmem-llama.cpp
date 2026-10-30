# KV 生命周期六阶段进度

更新时间：2026-10-31。完整目标仍为 [总体设计](kv-lifecycle-system-design.md) 中的六个阶段；单项通过不代替完整生产验收。

- **A 基线与契约：已完成。** NInfer 主后端、模型/源码固定身份、首批范围、动作及资源 owner、可执行契约已建立。证据：[阶段 A](kv-lifecycle-baseline.md)，提交 `ca2bbfe`。
- **B 可靠机制与预算：首批支持档位已完成。** B1 原子交接、B2 固定提交条件、B3 精确直接代表/评分证据和 B4 GC 自有辅助预算、旧 owner 计费、worker 健康均已实现。真实线程与交叉退休、禁止新分配的前台提交、分配故障注入、版本/保护撤销及 Linux 核心 ASan/UBSan 有独立证据。TSan 因工具在进入应用前失败仍无结论；不因此宣称竞态检测通过。证据：[B1](kv-lifecycle-boundary-mailbox.md)、[B2](kv-lifecycle-commit-conditions.md)、[B3](kv-lifecycle-score-evidence.md)、[特征预算](kv-lifecycle-auxiliary-budget.md)、[owner 预算](kv-lifecycle-owner-budget.md)、[B4 原生接线](kv-lifecycle-native-budget.md)。

  服务验收发现并修复了生成中辅助额度不足会使整个 Engine 故障的真实缺陷。第 26 补丁区分有效 sequence 的有序 abort 和已消费 membership 的结清，保留等待 FIFO。当前版本 20 项主机契约、5 项原生测试、两环境各 6 项协议测试通过；真实模型 8 项活动/排队停止、2 项生成中额度不足、12 次 M=10 MiB 独立会话轮换全部通过。之前的四组强压力回收、非流式拒绝/后续恢复、断连与中间二进制 48 次轮换按各自版本保留，不混记。见 [服务故障及完整结果](kv-lifecycle-service-failures.md)。

  这里完成的是冻结的单 GPU、单运行请求、文本/RAM 档位的机制和有界 GC 自有资源；M/H 不是整个进程或 checkpoint 的上界。新测试实际观察到冷域/checkpoint 带来的 RSS 增长，完整 checkpoint/源记录预算继续由 C/E 处理。长时间、固定工作量性能、先前偶发 CUDA 和 Agent 质量属于后续门槛，完整生产系统尚未完成。
- **C 恢复与版本闭环：实现中，未完成。** C1 独立 RAM SourceJournalStore、真实配额、分配失败原子性、revision/分支/局部遗忘 fence 已完成组件验证，见 [C1](kv-lifecycle-source-journal.md)。C2 已接入 NInfer：实际 frontend 模板/配置及消息来源、接纳输入与最终提交输出、独立 `--kvmem-source-mib` 和用量，以及取消/失败清理。最终二进制通过 21 项主机契约、7 项原生检查、4 组真实内容记录、2 组源额度耗尽/FIFO 和 4 组取消边界检查，见 [C2 完整说明](kv-lifecycle-native-source.md)。源记录默认关闭、当前按请求保留（含预热/失败输入），没有源过期/公开遗忘或去重。

  C3 的规划/源准备组件已完成：显式规范前缀新域计划、完整容量检查、独立 R 实际准备配额、source/domain/config 版本及局部遗忘校验。22 项主机契约及 Linux 三项 ASan/UBSan/泄漏检查通过，两个真实分配失败点不留下部分准备。保留首次归档驱动路径错误及修正；详见 [C3 组件说明](kv-lifecycle-source-recovery.md)。它尚未接入原生重建，不支持超出完整规范窗口的稀疏源重建，也不预约模型资源或发布状态。实际重建、checkpoint/归档 fence、事务发布、持久化与系统级遗忘仍待完成，不能把“能记录/准备源数据”写成“能恢复”。
- **D 任务感知保留：未完成。** 仍需结构化任务事件、有限锚点、衰减价值策略及质量比较；已有角色/thinking/冗余能力保留为基础。
- **E 分层与多会话：未完成。** 已有 HostKvStorage/SpillFile 等组件不等于 GC+SSD/并发组合已验收。
- **F 完整生产及研究验收：未完成。** 仍需新版本 GPU/Agent、长时间多次回收/恢复、资源边界及性能验收。

下一步继续 C3 原生接线：复用既有 Program 物理资源规划，先预约实际模型空间，再执行隔离的新域重建并安全发布；接通 checkpoint/归档/版本/退休 fence，完成缺页/前缀分歧及恢复失败。只有真实删除后成功恢复并验证旧快照不能复活退休身份，才关闭 C。源准备组件和 C2 记录通过不代替实际恢复或 F 的配对性能、长时间与 Agent 质量验收。

用户“暂不修改主机调试设置”和“DeepSWE 暂缓”的约束继续有效。当前无需用户补充信息即可推进下一项。

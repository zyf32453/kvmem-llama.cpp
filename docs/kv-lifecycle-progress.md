# KV 生命周期六阶段进度

更新时间：2026-10-31。完整目标仍为 [总体设计](kv-lifecycle-system-design.md) 中的六个阶段；单项通过不代替完整生产验收。

- **A 基线与契约：已完成。** NInfer 主后端、模型/源码固定身份、首批范围、动作及资源 owner、可执行契约已建立。证据：[阶段 A](kv-lifecycle-baseline.md)，提交 `ca2bbfe`。
- **B 可靠机制与预算：进行中。** B1 原子交接与 B2 固定提交条件已完成主机候选验证。前台不再获取 worker/内容范围 mutex 或调用策略 permits；原生保护、Query/分类换代及 Retry 原条件仍复核。14 项契约、两个旧/新等待对照、500 代际交接及 Linux 四项 ASan/UBSan 完成。TSan 尚无结论；结构化 witness/评分证据、完整辅助预算、worker 故障路径和新原生服务验收仍未完成。证据：[B1](kv-lifecycle-boundary-mailbox.md)、[B2](kv-lifecycle-commit-conditions.md)。
- **C 恢复与版本闭环：未完成。** 在现有 MemoryStamp/内容版本/checkpoint 能力上继续实现恢复规划、源记录边界和失败事务，不重复建设底层类型。
- **D 任务感知保留：未完成。** 仍需结构化任务事件、有限锚点、衰减价值策略及质量比较；已有角色/thinking/冗余能力保留为基础。
- **E 分层与多会话：未完成。** 已有 HostKvStorage/SpillFile 等组件不等于 GC+SSD/并发组合已验收。
- **F 完整生产及研究验收：未完成。** 仍需新版本 GPU/Agent、长时间多次回收/恢复、资源边界及性能验收。

下一步实现 B3：将 redundancy 的直接代表、Query/评分版本纳入精确的提交证据；随后补预算和故障路径、新原生服务验收，再进入 C。

用户“暂不修改主机调试设置”和“DeepSWE 暂缓”的约束继续有效。当前无需用户补充信息即可推进下一项。

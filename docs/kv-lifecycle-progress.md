# KV 生命周期六阶段进度

更新时间：2026-10-31。完整目标仍为 [总体设计](kv-lifecycle-system-design.md) 中的六个阶段；单项通过不代替完整生产验收。

- **A 基线与契约：已完成。** NInfer 主后端、模型/源码固定身份、首批范围、动作及资源 owner、可执行契约已建立。证据：[阶段 A](kv-lifecycle-baseline.md)，提交 `ca2bbfe`。
- **B 可靠机制与预算：进行中。** B1 原子交接、B2 固定提交条件和 B3 直接代表/评分证据已完成主机候选验证。前台不再获取 worker/内容范围 mutex 或调用策略 permits；原生保护、Query/分类换代及 Retry 原条件仍复核。直接代表的精确身份/版本与 victim 退休在单次尝试的原子事务中复核，提交 ACK 记录实际证据。16 项契约、旧/新失效代表对照、100 轮交叉退休和 Linux 六项 ASan/UBSan 完成。TSan 尚无结论；完整辅助预算、worker 故障路径和新原生服务验收仍未完成。证据：[B1](kv-lifecycle-boundary-mailbox.md)、[B2](kv-lifecycle-commit-conditions.md)、[B3](kv-lifecycle-score-evidence.md)。

  B4 已接入特征/范围/redundancy、manager 队列与快照、repository 槽位/控制块/预约表、原生 bridge 目录/固定数组、LRU、内部对象与显式回调 owner。旧 reader 保留真实目录计费，失败构造/预约不产生半份状态；原生公共 bridge 热路径和 5 轮真实 Retry 在禁止新 C++ 分配时通过。当前候选 18 项 Windows 契约、8 项 Linux ASan/UBSan、原源码共享探针通过；两档共 65 个构造分配点分别注入底层失败与额度不足，130 个回退检查完成。原生 CLI/producer、请求与旧域准入、worker 健康仍未接线，尚不能宣称运行服务的全链路预算已生效。见 [特征侧](kv-lifecycle-auxiliary-budget.md)、[管理与 bridge 侧](kv-lifecycle-owner-budget.md)。
- **C 恢复与版本闭环：未完成。** 在现有 MemoryStamp/内容版本/checkpoint 能力上继续实现恢复规划、源记录边界和失败事务，不重复建设底层类型。
- **D 任务感知保留：未完成。** 仍需结构化任务事件、有限锚点、衰减价值策略及质量比较；已有角色/thinking/冗余能力保留为基础。
- **E 分层与多会话：未完成。** 已有 HostKvStorage/SpillFile 等组件不等于 GC+SSD/并发组合已验收。
- **F 完整生产及研究验收：未完成。** 仍需新版本 GPU/Agent、长时间多次回收/恢复、资源边界及性能验收。

下一步继续 B4：接入 NInfer 原生预算配置、计费 factory/source 与 producer 暂存，处理旧域/请求准入和 worker 健康/故障路径；随后做新原生服务验收，再进入 C。

用户“暂不修改主机调试设置”和“DeepSWE 暂缓”的约束继续有效。当前无需用户补充信息即可推进下一项。

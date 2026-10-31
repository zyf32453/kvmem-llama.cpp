# NInfer 源家族遗忘与原生别名撤销

日期：2026-10-31。C4 的进程内源家族及原生别名撤销已通过四组模型功能检查，完整阶段 C 尚未完成。当前总状态见 [六阶段进度](kv-lifecycle-progress.md)。

## 边界与语义

`Engine::forget_historical_source(receipt)` 遗忘当前 Engine 中的一条源记录及其显式子孙分支，同时撤销这些源所绑定的原生历史 KV、checkpoint 和 coverage 别名。返回实际删除的源记录数、撤销的原生历史域数。它使用已有 SourceJournalStore 的父来源和不可复用 source identity，不增加第二套目录或墓碑表。

普通内存压力 GC 仍只退休历史 KV，不调用这个接口。源额度不足继续明确拒绝；只有显式调用才放弃这一源家族的后续恢复依据，避免把“清理缓存”和“永久放弃源”混为一谈。

本接口复用单 GPU、单运行请求、文本/RAM、普通解码或 MTP 的已支持范围。没有 HTTP/CLI 遗忘入口，也不跨进程、跨 Engine 或匹配所有包含相同文本的独立请求。调用方已有 token 向量、返回结果及复制的 transcript 不在其删除范围；接口不承诺安全擦除。

## 原子边界与资源

Engine 同时排除新的队列提交和模型单元，要求没有尚未结清的请求，包括已经结束但结果尚未取走的 handle。忙碌时返回 Pending，不自动取消请求。Program 先验证祖先回执的当前 revision 和所有相关 owner，准备所需的空窗口；这些准备失败不改变源和原生别名。

提交阶段不做新分配：先撤销相关原生历史域及其所有 checkpoint/coverage 别名，再删除源家族。尚由 ResourceManager 持有的 device continuation 不会被这个接口擅自释放；存在时明确拒绝。source identity 单调分配、不复用，所以已删除的回执无需新增无限墓碑表即可失效。

已经准备好的恢复对象可能仍保存源副本；后续正常提交会在准入时重新校验源，拒绝进入模型执行。它们继续占用 R，直到实际 owner 销毁。S 和历史 payload 随真实 owner 退出结算；返回的域数不是物理 bytes 释放证明。checkpoint pool 的空闲 buffer 也可能继续分配，不能把遗忘宣称为全进程内存清零。

## 验收要求

主机检查覆盖有界、无分配的多代家族查询、旧 member revision、祖先 stale、其他 store/独立源及目录复用。真实模型检查要求三代源和原生域一起撤销、无关源保持有效、实际 partial prefill 期间及未消费 handle 时拒绝遗忘、旧 revision 不部分修改、旧准备对象拒绝且 R 延迟归还、Engine 后续可用、last-source 观测清理，以及调用方重新提供原始 IDs 时建立新身份而不命中旧 checkpoint。

机器可读输入为 [native-forget-candidate-v1](contracts/kv-lifecycle-native-forget-candidate-v1.json)，真实模型入口为 [ninfer_source_forget_acceptance.py](../scripts/ninfer_source_forget_acceptance.py)。

最终原生 tree 为 `5570676d1e0145b413116877b96b0034f3762644`，29 个有序补丁，冻结 106 项输入。最新候选的 9 项原生检查全部通过。22 项 Windows/GCC 16.2 Debug 主机契约、3 项 Linux/GCC 13.3 的 ASan/UBSan/LeakSanitizer 核心检查在本次首个候选通过；后续仅修改原生模型测试输入和诊断，逐项核对核心/主机测试源码相同后复用，没有重复计为最新 tree 上重新运行。主机清单一度继承 C3 的负载描述，已改正实际 C4 参数，并保留原主机报告及其清单；测试源码和契约集合未变。详细结果见 [执行摘要](experiments/2026-10-31-lifecycle-stage-c-native-forget/summary.json)。

同一 Qwen3.8-27B v3 模型、RTX 3080 Ti Laptop 16 GiB / 驱动 596.08，LRU／redundancy × 普通解码／MTP4 共四组全部 exit 0、未 OOM。本批使用 B=2,048 tokens、生成预留 512、H=96 MiB、S=16 MiB、R=8,192 bytes、prefill=512、rk4v4、greedy seed 42；原始输入 1,024 tokens，两次 1,024-token 源重建形成三代家族，另有独立输入 512 tokens 和忙碌输入 768 tokens，retained_sessions=8。

四组均验证：遗忘前 5 条源和 5 个原生历史域，遗忘后剩 2 条/2 域；三代家族撤销数严格为 3/3；真正部分 prefill 中和未消费 handle 时均返回 Pending，旧 revision 返回 Stale；无关源仍保留原生历史。两份旧准备对象仍占 8,192-byte R，分别被准入拒绝后按实际 owner 退出降至 4,096、0，未产生新的源或模型重建。随后清理 last-source，重新提供原始 IDs 时新 source id=6、reused=0、8 个 greedy 输出与原请求相同。两组 MTP 实际执行投机轮，最后的新请求各为 2 轮。

首次真实模型检查失败于测试前置条件：2,048-token 根及较大的忙碌输入使既有 H 准入提前换出最老域，因此尚未遗忘就无法满足 5 个域共存。失败 tree `7ef0bd93…`、原始日志及 exit 1 保留。修正为能在同一 H 内共存的输入，未放宽 5 域共存、3/3 撤销或其他断言，没有改遗忘实现或历史淘汰算法。

## 实测资源代价与后续

普通解码的两组，遗忘后 Host payload 从 83,722,240 降至 25,231,360 bytes；MTP 两组从 88,954,880 降至 26,808,320 bytes。源 S 均从 79,104 降至 53,760 bytes，固定源目录仍计费。

checkpoint 记账在普通解码从 1,540,397,678 降至 924,051,868 bytes，MTP 从 1,540,423,338 降至 924,062,132 bytes；这些是当前 checkpoint 用量观测，不是 RSS。剩余包含仍有效的两个域和 pool 的空闲 buffer。H 并不限制这部分，更不能以“KV payload 已有界”宣称进程内存已控制。下一步优先给现有 checkpoint owner/pool 补完整预算及额度失败行为，再推进持久化与归档 fence。

本批输入短、输出仅 8 tokens，不证明性能、长时间或 Agent 质量；没有新 TSan/GPU sanitizer 结论。持久化归档、跨进程全局遗忘、长源重放、完整恢复事务与 D–F 门槛仍待完成。

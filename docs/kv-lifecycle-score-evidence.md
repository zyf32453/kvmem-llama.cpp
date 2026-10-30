# B3：直接代表与评分证据的提交契约

日期：2026-10-31。父提交 `420c5bfa3d1edfaf06ce32b5515b3764e0913ec4`。本项继续第二阶段可靠机制；不改变第一阶段冻结的 NInfer 首批支持范围，不代表新原生服务或完整生产 GC 已通过。

## 修复的问题

原 redundancy 策略已经计算直接代表，并在提案日志记录 witness；不能将此次改动描述成从零引入直接代表。缺口在于，manager 实际收到的删除命令只有 victim、round、reason 和内容版本，提交时不检查代表是否还存在。策略得到 ACK 后还从自己的缓存重建代表，可能与实际提交证据不同。

例如：worker 判断 A 可由 B 代表并准备删除 A；B 在前台消费批次前已退休并释放，旧 manager 仍然可以退休 A。此次用相同公共 API 回归程序分别链接 B2 与当前库：旧库报 `redundancy victim committed after witness was retired and freed`，当前库保留 A 并要求重新评估。测试同时核对真实 HostKvStorage 计费由 32 bytes 降到 16 bytes，避免只检查模拟状态。

## 命令携带的证据

`HistoricalKvDeleteIntent` 可携带固定大小的 `HistoricalKvScoreEvidence`：特征 schema、Query epoch、冻结评分 bank epoch、策略 round、是否实际使用特征评分、importance、retention score，以及可选的代表身份/内容版本、cosine 和阈值。

普通 LRU 不必伪造评分证据。Redundancy 必须有直接代表；评分策略的 LRU fallback 也带来源上下文，并明确 `feature_score_used=false`。manager 拒绝缺失、非有限数字、上下文不符、自代表、身份/版本错误、相似度低于阈值，以及代表也在同一删除批次中的命令。此处是可信策略提交协议的一致性检查，并不重新计算向量，也不证明两段 KV 在语义或推理结果上等价。

`feature_schema_id` 在 source/manager 生命周期内不可变，当前 schema 1 对应现有页面 cosine 特征契约。`score_epoch` 标识 `begin_round` 冻结的重要性 bank。每个 token 的原始 attention 更新不会使已冻结 bank 失效；重新冻结则撤销旧提交条件，避免每 token 都换代导致永远无法提交。策略仍可能保留同一 Query 下更早 bank 的 peak；这些字段是有限决策来源记录，不是可重放所有评分历史的日志。完整来源与恢复闭环仍属于后续阶段。

提交条件从最多 3 个原子 epoch 扩到 4 个，以同时覆盖 Query、评分 bank 和两类范围版本。source 未提供稳定评分 epoch/guard 时，评分提交不降级成前台任意回调。来源对象的原子地址必须活得比 manager 久；Query/范围事件与边界提交仍须由原生执行 owner 串行调用，沿用 B2 契约。

## 代表与 victim 的原子退休事务

仅为代表持有 read lease 不够：既有读取可以跨退休继续，所以 read lease 本身不阻止代表退出未来召回。

repository 增加 `retire_with_witness`。所有退休入口共同使用一个只尝试一次的原子事务标记；竞争者立即返回 Retry，不等 mutex、不自旋。事务内取得代表的 metadata-only pin，检查同一 incarnation、精确 slot/generation/content version、Live/sealed，然后在同一事务内提交 victim 的退休 CAS。Live 的内容不可增长，其他退休入口不能在检查和 CAS 之间退休代表。普通读、资源释放和分配不占用这个事务标记。

该 pin 不授予 Host 数据访问或 GPU 使用权，不增加 hotness，函数返回即结束。没有永久代表列表，也不阻止以后其他策略有损退休该代表。两个并发的 A→B、B→A 条件退休最多成功一个；后一方必须重新评估。普通非条件删除仍可以在之后删除代表，这与永久语义保留是不同承诺。

如果代表已退休但仍有旧 reader、已释放、槽位已换代，提交返回 `WitnessChanged` 并计数；竞争返回 Retry，重试保留原证据。ACK 用实际提交命令中的代表更新策略状态，不从旧缓存反推。诊断 ACK 在 worker 输出完整上下文、代表和分值，前台仍只执行固定字段与原子检查，不执行策略代码、stdio 或分配。

## 验证与范围

- `historical_kv_witness_lifecycle_test`：同一程序链接旧/新库的失效代表对照，包含实际 Host 释放。
- `historical_kv_score_evidence_test`：13 类无效证据在 worker-only/boundary 两种模式下被拒绝；同批代表冲突和循环；代表退休、释放、复用；冻结 bank 换代与原始 attention 更新；实际 ACK 代表；代表可在之后显式退休；100 轮交叉条件退休；暂停事务时竞争方直接 Retry。
- 候选清单继续固定源文件、24 个原生补丁和首批档位。旧 B2 检查器必须拒绝当前源码，不能复用旧通过结果作为新证据。

Windows Debug 本次测得 intent 为 176 bytes、条件为 128 bytes；不是整个 GC 的资源上界。批次、重试、审计、特征和策略暂存结构的统一预算仍须在 B4 完成。Linux 内存检查结果及完整源码/二进制身份在执行摘要中另行记录；不把主机测试替代新 GPU/Agent 验收。

后续仍需完成辅助预算、worker 故障路径和新原生服务接线验证，随后进入恢复/版本、任务保留、分层/多会话与完整验收阶段。

结构化复核见 [B3 执行摘要](experiments/2026-10-31-lifecycle-stage-b-evidence/summary.json)。完整证据位于 `artifacts/lifecycle-stage-b-evidence/evidence.zip`，SHA-256 为 `c4e94c8dc882a3e246a9971d8a2c7f8de2f537edebb71bec6113011f7ea0dc85`。

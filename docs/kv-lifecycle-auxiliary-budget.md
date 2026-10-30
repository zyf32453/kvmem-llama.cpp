# B4 辅助预算：特征与策略侧接入

日期：2026-10-31。父提交 `81aa105f40525021a9d475ef3713fd0615787d66`。本项是第二阶段辅助预算的第一批实现，完整阶段仍未完成。

## 账户与释放边界

既有 HostKvStorage 继续管理 KV payload，SSD/pinned 等资源仍按原 owner 结算。本项新增共享的 `HistoricalKvAuxBudget`，约束显式接入组件的 allocator 请求字节。它是辅助内存账户，不替代 HostKvStorage，也不声称约束整个进程 RSS。

账户容量创建后不可增加；分配前以原子 CAS 预占，失败抛出 `MemoryError(BudgetExceeded)`，不调用上游分配器。成功预占后，上游分配失败会退款并保留原异常；容器归还底层 allocation 后才退费。记录容量、当前计费、预占峰值、拒绝次数和分类计费。分类计数是各原子的观测值，并发读取不承诺一次事务内的分类总和。

扩容会同时计入旧 allocation 和新 allocation；不能按扩容后的最终差额预支退款。`clear`、truncate、计划回收和逻辑退休均不退款，保留的 vector capacity 继续计费。旧域容器及其副本通过 allocator 持有共享账户，owner 变量退出并不删除仍被引用的计费对象。copy/move/swap 的 allocator 传播保持 allocation 与原账户配对。

默认上游为 C++ `new_delete_resource`；自定义上游必须比账户和所有分配活得久。账户覆盖请求的容器存储字节，不测量 malloc 元数据、分配器缓存或 OS 页。预占峰值还包括申请中的额度；失败的底层分配也可能提高这个峰值。

## 本批接入

- `HistoricalKvPageFeatures`：页面目录、每页 descriptor、Query 分值、输出 peak 和冻结评分 bank，含嵌套分配。
- `HistoricalKvThinkingRanges`：thinking spans 与已完成回复 spans。
- `HistoricalKvRedundancyPolicy`：候选/行/参考/seen/特征矩阵、嵌套直接相似边，以及年龄采样、代表排序、支持位、先前代表 fallback、softmax 的临时数组。

三类对象可显式接收同一账户，从而共同竞争一个上界。未传账户的旧调用保持普通分配行为；不能将其记为“已经具备辅助上界”的配置。

同时将 Query 分值数量限制为 source 的逻辑页数，拒绝超出几何范围的输入。页面、Query、冻结 bank 或范围新增若在分配时失败，不发布半份数据，不提前推进对应版本。成功的冻结 bank 仍按 B3 的评分版本契约使用。

预算耗尽不会隐式删掉更多历史。策略步骤中的异常由既有 manager 停用该策略并丢弃本轮命令；已退休资源仍可继续释放。本批只验证此既有故障路径，还没有完成向原生请求入口发布统一健康状态和拒绝原因。

## 验证

新增 `historical_kv_auxiliary_budget_test`，使用独立的上游 memory_resource 观察实际分配/归还，核对账户值；测试观察器自己的记录不计为产品内存。

覆盖扩容瞬时峰值、跨组件共享容量、旧 owner 生命周期、copy/move/swap、超对齐、乘法溢出、部分构造退款、页面/Query/bank/范围的底层分配失败，以及实际 redundancy 回合预算不足后不发布删除且继续释放旧退休块。

正常评分路径先记录实际分配点，再逐点注入底层失败。本机观察到 66 个分配点；每个失败点都必须在 producer 边界明确失败或由 manager 停用策略，不能提交部分删除，结束后计费为零。另有 400 次确定的并发满额拒绝和 2000 次并发分配/归还，检查共享容量与退款。计数来自实际执行，不以固定编号绑定某个标准库实现。

## 后续接入与完成条件

manager 的观察快照、批次/重试/审计队列，repository/原生 bridge 的目录和固定数组，LRU 暂存、对象与回调 owner、原生 producer 暂存及旧域准入尚需继续纳入预算。当前新账户未接入原生 CLI，未确定生产默认额度，也未做新 GPU 长运行验收。

因此，本批只能证明已列出容器在显式账户下的分配上界。后续须将同一预算传入主后端全链路，在入口校验配置和初始化峰值，并覆盖旧域退出、worker 健康、失败/取消与容量不足。完整第二阶段仍以总体设计中所有完成条件为准，不能用本测试替代。

结构化复核见 [B4 执行摘要](experiments/2026-10-31-lifecycle-stage-b-budget/summary.json)。本批 Windows 17 项契约及 Linux 7 项 ASan/UBSan 通过。完整证据位于 `artifacts/lifecycle-stage-b-budget/evidence.zip`，SHA-256 为 `7c5ac1adb11a8d643ada9e2ea558def1452577eb066d43b8fa69d5bcd401e668`。

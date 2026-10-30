# B4 管理层、目录和原生桥接层预算

日期：2026-10-31。父提交 `5a79f8de48e063d16def88beb5d6c884f8200903`。在上一批特征/策略预算上继续接入公共管理层。新原生 CLI 与请求准入尚未接线，阶段 B 仍未完成。

## 本批改变

1. **管理层存储。** round/observation 元数据、决策缓冲、prepared/pending/ACK/retry 批次、提交条件、结果码、generation 去重及可选审计表都使用同一辅助账户。Retry 阻止新提案，所以 retry 暂存只需一个 batch；不再按整个 repository 的槽位数预留。swap 仍由 worker 执行，前台提交不触发 vector 扩容。
2. **repository 与旧域。** Slot 数组使用固定地址的计费数组，内部 Shared 及其 control block 使用计费的 allocate_shared。Reservation 句柄表、返回的 snapshot/observation 也计费。close 或外层 repository 析构不代表这些内存已释放；旧 lease 保留的 Shared/Slot 继续占额度。新域不能假借旧域“已经关闭”提前取得退款。
3. **原生 bridge。** 页面句柄表、slot→page 表、提交结果、保护位、释放代际、release ticket、detached record 表和策略注释快照都纳入账户。默认 LRU、ProtectedPolicy、manager 的堆对象通过计费 owner 创建；ReleaseState 和内部 repository 的共享对象/control block 也计费。使用 `historical_kv_aux_make<HistoricalKvPageGc>` 时，bridge 根对象自身同样计费。
4. **LRU 和账户身份。** LRU 暂存纳入账户。manager 默认继承 repository 账户；启用预算后，策略对象及其内部存储须声明同一账户。redundancy 的 source/range、LRU 的 range 也必须匹配，避免一层有预算而下层悄悄使用另一个或无账户分配。旧无账户调用仍可使用原接口，不将其算作新预算档位。

固定数组支持不可移动的 atomic/slot。部分元素构造失败时只销毁已构造元素，并归还 allocation。计费 owner 区分具体对象类型与接口类型，先执行正确的派生析构，再归还实际对象存储；不会在多态销毁时按基类大小退款。

## 释放回调的所有权

原生页面回调原先通过捕获 weak_ptr 的 std::function 绑定 ReleaseState，标准库可能为闭包另行分配。本次增加 `ResourceRelease::callback_with_owner` 固定函数指针和显式 `callback_owner`。owner 持续到 Metadata 回执完成；后台用固定入口释放真实 Host record 并等待原 release ticket。

原生 bridge 将整个 Host record 归入 PackedKv 组件，MeanK 字节为零，由同一记录 owner 结算；不是分别释放记录内部的地址。现有 page 保护与实际 Host record/reader 许可继续适用，回收目标仍是 CPU 历史资源。

普通 std::function 回调保留给旧调用。预算模式拒绝其无法计费的闭包；裸函数指针回调或显式 owner 的固定入口可用。原生 owner 由已计费的 ReleaseState 持有；外部 payload/IO owner 仍属于其原资源账户，不因 callback_owner 引用就改记到辅助账户。配置两种回调入口同时存在会被拒绝。

## 失败与兼容边界

所有 Reservation 句柄表分配都在修改槽位与 generation 之前。预算不足、底层 allocation 失败或策略账户不匹配时，部分构造返回所有已取得存储；manager 最后才 claim repository，不留下半个 owner。

公共历史快照、decision values 和 bridge 提交结果的容器类型现在使用辅助 allocator。本次主机契约调用方已随新类型重建，冻结共享探针也用原源码重新编译并执行；原生服务仍须重建验收。使用这些 C++ 类型的外部调用方须重新编译，不能混用旧头文件和新二进制。持有返回容器的容量也会持续计费，清空内容不代表退款。

预算仍按 allocator 请求字节定义，包含扩容预占，不能等同 RSS。外部调用方自行 new/make_shared 的 source、根对象或外部 callback owner 需要相应的资源 owner/工厂；声明预算参数不自动拦截全部框架内存分配。线程运行时/栈、libc stdio 等固定或外部资源须在请求/域资源清单中另列，不将它们谎记为本 allocator 已计费的字节。

## 验证

新增 `historical_kv_owner_budget_test`：

- 同一真实 PageGc/HostKvStorage 流程测试注册、保护更新、准备、提交、释放确认、实际 Host 退款和跨 generation 复用。进入热路径后禁止新的 C++ new/array/aligned-new，核对辅助账户也没有新的上游请求；关闭/开启审计分别运行。
- 先测出完整构造图的分配点，再逐点注入底层失败及对应额度不足。当前两档分别 32、33 个分配点；每个失败后账户归零、观察器没有遗留 allocation。
- 外层 repository 消失但旧 reader 仍存活时，新域在同一容量下创建失败；旧 reader 退出后新域成功，原 reader 未被失败构造污染。
- 预约句柄表失败不修改 Ready/代际；reset 的 retained capacity 继续计费，析构才退款。失败 manager 不占用唯一 owner，之后可成功构造合法 manager。
- 真实退休事务被另一线程暂停时，连续 5 轮、每轮 2 条 Retry 不扩容；事务解除后原批次恰好提交 2 条并释放 2 条，不额外选块。

完整 Windows 契约与 Linux ASan/UBSan 结果见执行摘要。测试中的全局 C++ 分配拦截不覆盖 libc stdio、进程 RSS 或 CUDA，不能由此推导整个原生请求路径的实时期限。

## 下一步

把共享辅助账户、计费 factory/source/policy owner 和有效额度接入 NInfer 原生配置、producer 暂存和新旧域的请求准入；发布 worker 健康/失败结果。之后重建服务并验收普通解码、MTP 和真实任务。现有服务默认尚未启用新辅助账户；上述主机检查不能替代这一步，完整阶段 C–F 也仍需继续。

结构化复核见 [B4 owner 执行摘要](experiments/2026-10-31-lifecycle-stage-b-owner-budget/summary.json)。完整证据位于 `artifacts/lifecycle-stage-b-owner-budget/evidence.zip`，SHA-256 为 `83070b0ffb2bc429f00e90462379402a002b5da8547233a3038b0f18224fa652`。

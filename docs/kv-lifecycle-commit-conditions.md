# 第二阶段固定提交条件

日期：2026-10-31。状态：B2 主机候选验证完成，第二阶段整体仍未完成。前置为 [B1 原子批次交接](kv-lifecycle-boundary-mailbox.md)。

## 移出前台的工作

此前 decode 边界调用 `policy->permits()`。LRU/redundancy 在该函数中读取内容范围；范围由 mutex 保护，单纯移除 B1 的批次锁仍不能消除这条等待。

现在 policy 只在 worker 执行 `prepare_commit()`，为每条意图生成 `HistoricalKvCommitConditions`。前台通过 repository 原有的精确代际 guard 和退休 CAS，执行固定条件检查：Live/sealed/mandatory、原生页面保护，以及最多三个原子版本比较。原有 expected_content_version 仍在同一个代际保护范围内核对。

前台不调用策略函数、不访问内容区间 vector、不获取其 mutex、不重新评分或记录策略日志。repository 的底层 RetirePredicate 入口仍保留，但边界模式传入的是固定 `conditions.matches()`，不是任意策略函数。纯后台模式和 shadow 的策略兼容路径留在 worker。

原生的 `on_boundary_retired` 通知仍保留：它在已有预分配缓冲区发布 exact-generation tombstone，不是策略代码。没有把所有 callback 类型都删除，也不把该改动宣传成硬实时保证。

## 后端保护与 Query

NInfer `ProtectedPolicy` 为条件加入对应页面的稳定原子保护位；如果页面在准备后、提交前被重新保护，退休被拒绝并返回 Reevaluate。不会只检查准备时快照，也没有删除 active、sink/recent、media、checkpoint/IO 的原生保护。

`HistoricalKvPageFeatures` 提供自己拥有的 Query epoch 原子地址。新问题、清空与 truncate 使旧条件失效；同一问题的 partial Query 更新不换代，不会让长 prefill 的已准备批次反复失效。版本饱和后保持失效，不回绕到旧身份。

源对象、内容范围和 protection 数组由 manager 的 policy/backend owner 保持至 GC 析构；条件只保存这些稳定原子的地址，不持有触发前台析构的 shared_ptr 或资源 owner。原生事件生产与退休提交仍由同一串行执行 owner 排序；本协议不承诺让任意外部线程的范围/Query 修改与退休 CAS 成为跨对象原子事务。

## 内容分类与持续追加

内容范围使用两类撤销版本：

- classification_revision：clear/truncate、旧前缀重叠修改、角色层级变化、arm/disarm 等需要撤销旧分类的事件。
- growth_revision：真正增长或新增 thinking 区间；重复 note、重复 arm 不换代。

worker 在范围锁内分类并生成证书。整块位于当时 thinking frontier 之前时，正常向后追加不改变该块分类，只需检查 classification_revision；位于 frontier 之后的块再检查 growth_revision。向旧前缀补入或扩展非末尾区间会换分类版本，不能借“正常增长”绕过失效。

因此连续生成 token 不会每步撤销所有老块的决定，同时未来范围扩展、旧前缀改动和回退仍能阻止过期决定提交。每个计数器都在所属 owner/锁下单写，饱和后禁止生成可用证书。

## 自定义策略与 Retry

`prepare_commit()` 返回 Ready、Reevaluate 或 Unsupported。未知自定义策略默认 Unsupported：worker 记录 unsupported_conditions/policy_failures 并停止新策略产出，仍推进以前已退休资源的释放。不能为保持表面兼容而退回前台调用任意 `permits()`。纯 worker 退休模式继续支持旧式自定义 predicate。

新增策略需要显式生成有界条件；动态 Query source 也必须提供稳定原子 epoch guard，默认没有该能力。两个既有测试 mock 已适配，生产 NInfer source 已接通。支持集合仍为已冻结的 LRU/redundancy；这不是任意第三方策略已经获得新接口兼容的承诺。

Retry 同时保留原意图和原条件，不能对重试的旧决定重新授权一个新 Query/分类版本。acknowledgement、pending、prepared、retry 的条件缓冲区随原意图一起转移。

当前 64 位主机实测每条条件为 72 bytes、最多 3 个 epoch guard。构造时为三份 batch 缓冲区和一份 repository-capacity retry 缓冲区预留元素，worker 交换复用这些池，前台不 resize。实际 allocator/capacity 与整体元数据账本仍须在第二阶段预算项统一计量；72 bytes 不含整个域或进程开销。

## 验证

旧 B1 与当前版本使用相同 `historical_kv_range_lock_test.cpp`。旧头文件/静态库保持不变；负面控制仅用 GCC `-fno-access-control` 允许测试持有旧私有 mutex，不改变产品同步或布局。旧版本以 `boundary waited for content range mutex` 退出 1；当前版本在锁仍被持有时提交一个合格的真实 LRU 块，随后正常确认和释放。

新 `historical_kv_commit_conditions_test` 覆盖：

1. 在边界模式提供有意不可调用的 permits，验证退休只使用编译条件且仍能真实释放。
2. 受控注入 Retry 后更换 epoch，旧条件被拒绝，重新评分的新决定仍可提交；重试没有重新授权。
3. 不支持的自定义边界策略明确失败，纯 worker 模式兼容性保留。
4. 895 次正常范围增长与重复 arm 不撤销旧前缀证书；truncate、clear、旧区间重叠、未来区间变更和回复层级变化撤销相应证书；饱和版本不能授权。
5. 真实 page bridge 在提交前重新保护页面会拒绝退休，解除临时保护后新批次继续工作。
6. 真实 PageFeatures 的 Query 换代使旧 redundancy 决定失效，同问题的 partial 更新仍允许新的合法决定。

Windows/GCC 16.2 Debug 的完整候选 14/14 项 CTest 通过，包含 B1 的 500 次实际后台交接与原有存储/版本/checkpoint 契约。

Linux 隔离目录 `/tmp/kvmem-lifecycle-conditions-20261031-e08la7ju` 使用已有 GCC 11.4、Debug、ASan/UBSan 与 leak detection/遇错退出。真实 page bridge、500 次后台交接、提交条件、内容锁四项目标 4/4 通过；构建与 CTest 均退出 0。源文件逐项比对本地指纹，运行实际链接 libasan/libubsan，未运行 GPU。

本轮未再运行 TSan：B1 中普通和非 PIE 诊断在进入测试前即因地址映射失败退出，仍没有竞态检查结论；没有修改 ASLR、驱动或 Windows 调试接口。新代码也还没有 CUDA 服务重建、真实 Agent 或性能合格结论。

## 候选与后续

第一阶段和 B1 清单保持原样；[B2 候选清单](contracts/kv-lifecycle-conditions-candidate-v1.json) 以 B1 提交为父版本，显式列出 14 项实现/构建/测试变更，固定 81 项指纹及原生 24 补丁哈希。复现时显式传入该清单：

```sh
python scripts/check_kv_lifecycle_baseline.py \
  --manifest docs/contracts/kv-lifecycle-conditions-candidate-v1.json \
  --build-dir build-lifecycle-host \
  --output artifacts/lifecycle-stage-b-conditions/new-run
```

B3 继续补 redundancy witness 的精确身份/评分版本和提交条件，随后完成辅助资源预算与 worker 故障路径，并在隔离构建中补原生服务验收。没有将 B2 的局部通过替代第二阶段或全部六阶段完成。

结构化复核见 [B2 执行摘要](experiments/2026-10-31-lifecycle-stage-b-conditions/summary.json)。完整证据位于 `artifacts/lifecycle-stage-b-conditions/evidence.zip`，SHA-256 为 `08271448fdcff8cc7c9b571db69742f526fb1edf7e68cc46103e122fe0a8798e`。

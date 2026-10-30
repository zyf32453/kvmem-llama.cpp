# 第二阶段边界批次交接

日期：2026-10-31。状态：B1 候选已完成主机验证；第二阶段整体未完成。目标与总进度见 [六阶段进度](kv-lifecycle-progress.md)。

## 本次行为变化

此前后台和 decode 边界都获取 `pending_mu_`。即使持锁区很短，后台被调度暂停时，前台仍可能等待这把锁。

现在用一个预分配批次和 lock-free 原子状态交接，前台不获取 worker 的 mutex，也不改变批次 vector 大小。每次边界最多做一次领取 CAS；没有就绪批次、另一提交正在进行或结果尚未确认时直接返回，命令留在交接槽中。

策略评分、释放、统计回执和下一轮产出继续由后台执行。已有 `tick_mu_` 只用于序列化 worker/manual_tick，前台不获取它。没有用 try_lock 将原来的等待换成可能丢命令的路径。

## 状态与内存顺序

- **Idle**：worker 独占交接缓冲区，准备意图及结果槽。
- **Prepared**：worker 用 release 发布完整批次；缓冲区不再由 worker 修改。
- **Committing**：前台一次 acquire CAS 领取成功后，读取意图、执行当前退休协议、填结果。重入调用跳过该批次。
- **Committed**：前台在 native tombstone 通知全部完成后用 release 发布结果；worker acquire 读取、移出到自己的 acknowledgement 缓冲区，再归还 Idle。

慢 `on_result` 和诊断输出作用于 worker 私有 acknowledgement 缓冲区，不占据前台锁。Prepared/Committing/Committed 状态下，worker 仍先推进以前已退休对象的释放，但不叠加新的策略批次。Retry 沿用原有保留与重发协议。

这不是任意多生产者/多会话协议。仍要求一个串行 worker、一个串行 decode owner，并在销毁前让前台退出。CAS 的 Committing 状态额外防止同一批次被重入消费，不表示原生后端已支持并发请求。

## 没有在本次关闭的问题

`commit()` 仍经 repository 调用 `policy_->permits()`；LRU/redundancy 的内容范围检查可能读取受锁保护的数据。原生 `on_boundary_retired` 通知也仍保留。因此只证明**批次交接不等待 worker mutex**，不能称整个前台 GC 路径已经无锁、无等待或达到硬实时期限。

下一项 B2 要由 worker 生成固定格式的提交条件，在前台用有界检查复核 life/sealed/mandatory、native protection 和相关版本；未知的自定义条件应明确不支持，不能删除旧保护后继续退休。结构化 witness/Query 版本与完整资源预算随后继续推进。

GPU 工作集容量、历史选块策略、物理回收条件、NInfer 原生补丁和首批支持范围未改变。生产服务仍需用新公共核心重新构建并做 CUDA/Agent 验收。

## 可执行证据

新目标 `historical_kv_boundary_mailbox_test` 覆盖：

1. 暂停持有全部 worker mutex 的测试线程时，已准备批次仍可提交。相同测试源码可通过既有 friend 在阶段 A 核心上持有旧交接锁，旧版本以 `boundary waited for worker mutexes` 失败，新版本通过。
2. 等待边界提交的批次不阻止后台释放之前已退休的其他对象，也不导致额外产出。
3. 慢策略回执不阻止空边界返回。
4. 提交通知中重入边界不能重复消费当前批次，未确认的批次也不能被再次提交。
5. 实际后台线程进行 500 次代际创建、提交、旧 reader 退出、回执和回收；校验提交/回执/完整回收均恰好 500，旧身份不能取得新读取。

Windows/GCC 16.2 Debug 的候选完整契约为 12/12 通过，检查器单测 9/9 通过。首次 500 轮压力测试因原定 15 秒期限失败；诊断版本在期限处已经推进到第 468 轮、929 个 tick，记录保留。测试使用生产 worker 的定时轮询，不能把 500 轮完成时间用作固定 15 秒的吞吐门槛。测试期限改为 45 秒，CTest 上限 60 秒，500 次精确计数条件不变；最终独立用例约 17 秒完成。期间未修改生产轮询参数或主机定时设置。

检查器单测首次在当前沙箱默认临时目录遇到权限错误；改为仅给测试进程配置工作区内 TEMP/TMP 后 9 项通过。原错误日志保留，没有将目录权限失败归为产品失败或直接写成通过。

第一阶段固定清单在当前候选代码上按预期拒绝，原因是 `kvmem/CMakeLists.txt` 指纹变化。没有改写第一阶段清单；新增 [B1 候选清单](contracts/kv-lifecycle-mailbox-candidate-v1.json)，显式列出四项核心/构建/测试变更并链接第一阶段清单哈希。它复核 79 项源码/测试指纹及原 24 个补丁哈希。

复现候选主机检查：

```sh
python scripts/check_kv_lifecycle_baseline.py \
  --manifest docs/contracts/kv-lifecycle-mailbox-candidate-v1.json \
  --build-dir build-lifecycle-host \
  --output artifacts/lifecycle-stage-b-mailbox/new-run
```

Windows 按 [基线文档](kv-lifecycle-baseline.md) 显式指定 cmake/ctest 路径，并将测试临时目录放在可写工作区。输出目录不覆盖旧运行。

## Linux 内存诊断

仅将所需公共头文件、四个核心实现和新测试上传到隔离目录 `/tmp/kvmem-lifecycle-mailbox-20261031-ymy178c0`，使用既有 GCC 11.4，未操作 GPU、驱动、注册表或 ASLR。

ASan/UBSan 使用 `-O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer`，打开 leak detection/遇错退出；编译与完整用例均退出 0，完成 500 轮，没有访问/未定义行为/泄漏报告。源码逐文件指纹与本地一致，实际链接包含 libasan/libubsan。

TSan 编译成功，但普通与 `-fno-pie -no-pie` 两种进程均在进入测试前以 `FATAL: ThreadSanitizer: unexpected memory mapping`、退出 66 结束。**没有取得 TSan 竞态检查结论**，这两次不能计为通过；后续应在兼容检测环境补齐。

远程完整源码与诊断日志归档 SHA-256 为 `ad458443e5bacf84c27798bfc6bdaa0958725b22457d7f3cb7f746aae11d8dde`，本地位于 `artifacts/lifecycle-stage-b-mailbox/remote-evidence.tar.gz`。本轮没有新 GPU、Agent 或性能合格结论，也没有关闭原偶发 CUDA 问题。

结构化复核见 [B1 执行摘要](experiments/2026-10-31-lifecycle-stage-b-mailbox/summary.json)。本地主机日志、旧版本反例、候选输入和远程归档封装在 `artifacts/lifecycle-stage-b-mailbox/evidence.zip`，SHA-256 为 `c58769da8958b8321fb530d61518db8e7621286f2fc65e378686b70392962ec7`。

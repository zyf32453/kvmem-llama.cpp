# C6：持久源归档与发布资格

日期：2026-10-31。范围是 Linux 本地文件系统上的独立源归档组件，单 owner 串行管理、后台文件写入。已实现真实文件、重开、版本/家族遗忘 fence、配额与失败处理；尚未接入 NInfer Engine，不代表已经能通过 Engine 跨重启推理或执行系统级 ForgetSource。完整阶段 C、D–F 继续开放。

后续 C7 已补原子源导入、当前读回执授权的 RAM binding 和持久保存序号，见 [C7 源导入与跨进程续存](kv-lifecycle-source-import.md)。[C8](kv-lifecycle-native-archive.md) 已接通 Engine 归档和统一遗忘，并在固定配置下提供跨进程模型证据。下文的 C6 运行结果及未完成项按其原候选、原时间理解，不将组件检查倒写成 Engine 验收。

## 本批交付与完成条件

在已有 SourceJournalStore、SourceImage、StorageIoQueue 和辅助 allocator 上建立持久的源归档权威。完成条件是：文件写完不自动授权恢复；晚到 writer 和旧版本不能覆盖当前资格；遗忘后重开不能授权旧 key；缺失/损坏/提交不确定时拒绝；在真实 reader、writer、receipt 退出前不提前归还相应配额。本组件不改变 CPU KV 退休、GPU 工作集、模型数学或原生 checkpoint 格式。

接口在 [source_archive.hpp](../kvmem/include/kvmem/source_archive.hpp)，实现见 [source_archive.cpp](../kvmem/src/host/source_archive.cpp)，测试见 [source_archive_test.cpp](../kvmem/tests/source_archive_test.cpp)。

## 数据和权威分开

每个归档有随机生成并持久保存的 128-bit namespace。源 key 是 namespace 加单调 uint64 ID；blob 有独立单调 ID。清单保存源的 durable parent、当前已发布 blob、准确大小、摘要和保存时的 SourceState。删除后保留 high-water mark，不复用旧 ID，也不积累无限的逐源墓碑。

这里的清单只授权读取某份**源数据**，不是另一套物理 KV 目录。普通历史退休仍不可逆；未来从归档重建必须走新计算域和正常资源准入，不能直接装回镜像内的 SourceRef、旧 native 域或 checkpoint。保存的 accepted/computed frontier 描述旧快照，computed 不证明任何 KV 当前驻留。

`create()` 只接受新目录；已有目录必须 `open()`。打开需要既有 lock、完整 catalog 与 seal，独占 flock 失败明确返回 Pending。不同 owner、进程和 namespace 的工单不能互用。读写 receipt 可以晚于包装 owner 退出，但它们保留实际表、文件 quota 和目录锁；不能同时打开另一 owner 来绕过这些权利。

重开时，磁盘源条目不会自动绑定到任何 RAM SourceRef。`enroll()` 必须从当前 SourceJournalStore 验证真实源；分支必须先登记现存祖先。后续 `import_source()` 通过当前 read receipt 将旧 key 显式绑定到原子导入的新 RAM 源，已有 binding 拒绝替换；不能只比较 decoded origin 的数值来取得资格，跨进程的 SourceRef 数值本来就可能重复。

## 保存、发布、读取与遗忘

1. owner 登记源，`save()` 从当前 revision 准备真实 R 中的不可变 SourceImage，并先提交 blob ID high-water mark，再投递现有 StorageIoQueue。queue 的 jobs/bytes 上限继续生效，运行中也计数。没有新增线程池。
2. worker 只创建独有的 `blob.ID`，流式编码、同步文件、同步目录，返回 I/O 完成状态。它不改清单、源目录、KV 退休状态或 Engine。失败和取消不会删除原源，也不会发布部分文件。
3. owner `publish()` 等待条件满足后，重新检查工单归属、当前源身份/配置、归档条目是否存在及版本。源只在末尾追加时，较早的不可变 accepted 前缀可作为明确旧版快照发布，避免连续生成让保存永久失效。后续 C7 使用持久 blob 序号拒绝旧工单覆盖新版；RAM revision 只在当前 binding 内校验，不能跨重启比大小。源已遗忘或绑定变化则拒绝，相同已发布工单重复确认幂等。
4. `read()` 先查询当前权威，再核对文件大小、SourceImage 校验/配置、完整文件摘要及准确 SourceState，成功才返回带 reader lease 的对象。adapter 在真正导入前必须再次 `validate()`；读取期间发生遗忘或新版替换后，旧 receipt 失效，但其 R 和旧文件仍随真实 owner 存活。
5. `forget()` 先提交删除指定 durable key 及全部子孙的清单，关闭新发布和新读取；`collect()` 再清理不再被清单引用、job 已结束、receipt/reader 都已退出的文件。进程重开后也用同一清单拒绝旧 key，不根据剩余 blob 重建资格。

本方法的 `forget()` **只撤销归档家族**，不偷偷调用 RAM store 局部遗忘。最终 Engine 接线必须先提交持久撤销，再撤销在线 KV/checkpoint/coverage 和 RAM 源，覆盖失败后的清理；本批没有将归档局部操作冒充系统级遗忘。局部 RAM forget 能使未发布 writer 的 owner 复核失败，但不会自动删除之前已经持久发布的归档。

## 提交协议和故障取舍

catalog 是固定大小、显式小端、有版本和 checksum 的完整当前清单；seal 固定 64 bytes，记录相同 namespace、提交序号、catalog 长度与摘要。一次提交依次执行：

```text
写 catalog.tmp → fsync 文件 → rename 为 catalog → fsync 目录
写 seal.tmp    → fsync 文件 → rename 为 seal    → fsync 目录
全部成功后切换 RAM 当前表
```

不能把原子重命名等同于持久化。这里依照 Linux [fsync 文件与目录的契约](https://man7.org/linux/man-pages/man2/fsync.2.html) 和 [rename 的可见性契约](https://man7.org/linux/man-pages/man2/rename.2.html) 实现；调用方需使用遵守这些语义的本地文件系统。进程强制退出测试并不能证明实际主机/设备掉电行为。

发生任意清单提交异常，该 owner 进入 unhealthy，停止后续读、发布、登记和清理，要求实际权利退出后重新验证打开。重开只接受 catalog 与 seal 一致，不选“看起来还能读”的旧清单。只回滚 catalog 或只回滚 seal 都会被拒绝。

这个设计选择安全优先，存在明确的可用性代价：崩溃发生在 catalog 已替换、seal 尚未完成期间，重开将拒绝整个归档，不自动回退到上一版。某个已发布 blob 缺失/长度错误、未知文件或超限也会使打开失败。保留文件供后续显式恢复/诊断，当前没有自动修复入口。

catalog/seal 都放在同一受信本地 namespace 中。**把整个目录连同两份一致权威一起恢复成旧备份，当前没有外部单调锚点可以识别**；本批不承诺抵抗整体备份回滚或密码学篡改。FNV 仅检测损坏。该边界须保留在系统级恢复/数据保留合同中；不能宣传成任意备份中的全局永久擦除。

## 配额与实际寿命

- D 按逻辑文件字节约束。提前保留两份最大 catalog 和两份 seal 的 control 空间，再计正在写入、已发布、旧 reader 仍持有及待删除的全部 blob 预约。它不是文件系统分配块、OS page cache 或 RSS 上界。
- 源目录和对象表的格数固定。当前表、提交 scratch、对象表使用已有 HistoricalKvAuxArray，实际数组字节受 metadata_bytes 限制；adapter 可传共享 HistoricalKvAuxBudget，使旧 reader/writer 保留同一 M 账户，最后 owner 才退款。缺省独立组件按相同固定几何上限分配。
- SourceImage 全程使用调用方共享的 RAM-only R。job 完成/失败/取消并释放捕获对象以后才退款；读镜像继续持有自己的 R。S 中源数据不因 D、M、R 或 queue 不足而自动遗忘。
- 包装 state、array/account/shared control、ticket、有限 lambda/路径、allocator 与文件系统运行时开销不包含在上述数组/D/R 字节内；其数量受固定对象槽和现有 queue jobs 限制。不能据此声称全进程内存有界。
- `collect()` 在 job 和所有 receipt 退出后 unlink 并同步目录，才归还 D。unlink 已完成但目录同步失败时，仍保守保留额度，重试成功只退款一次。重新打开时扫描文件只用于账本/孤儿清理，不能提升恢复资格。

清理结果单独返回 `quota_released` 和 `file_bytes_removed`：前者是实际结清的预约，后者是已经删除并同步目录的文件逻辑字节。queue 拒绝/取消但未建文件时，后者必须为零；部分写入失败时，它可以小于退款。目录同步失败后的重试保留先前已 unlink 的准确大小，不能重复计数。源码审查发现初版单一返回值混合了这两种口径，最终版本已拆开并用独立断言验证；不把未写出的预约宣传为实际物理回收。

文件额度或对象槽不足返回 BudgetExceeded；队列不足返回 Pending。当前保存会先消耗持久 ID，再投递队列，因此 queue 拒绝也可能产生一个无文件的待结清预约，需要 collect；不回退 high-water mark。清单更新包含同步磁盘操作，方法必须在 decode 边界之外调用。本批没有性能改善或无等待承诺。

## 验证范围

Linux/GCC 13.3、Python 3.11.18 下对 archive、image、journal、recovery、spill 五个组件目标运行 Debug 和整个 host library 的 ASan/UBSan/泄漏检查，两组实际各约 12 秒，结果及精确输入见 [执行摘要](experiments/2026-10-31-lifecycle-source-archive/summary.json)。早期版本按各自源码通过，最终包含共享 M、清理重试和统计分离的版本再次通过；没有失败被改记为通过。强制 `_exit` 的故障子进程不执行析构或 LeakSanitizer 退出检查；内存检查结论属于正常结束的父进程和组件路径。

归档测试覆盖正常文件往返与重开、owner 排他/跨 namespace 工单拒绝、版本反序与 append、reader/owner 寿命、家族三层遗忘与晚 writer、取消与 queue 额度、D/M/R/对象槽拒绝、独立共享 M 耗尽及最终退款、孤儿清理、缺失/损坏文件、单独回滚 catalog/seal 和同步失败重试。对真实 OS 调用注入 83 个清单错误点，以及保存中的 83 个清单/30 个后台 blob 错误点；另在六个 fsync/rename 成功边界直接退出子进程。save/read 各覆盖 7/4 个真实前台分配失败点。

尚无实际掉电、整个目录备份回滚、Engine 保存/导入、GPU 恢复、在线别名的持久遗忘、性能或 Agent 验收。NInfer C++20 Release 构建接入及现有八个程序的二进制对比单独记录；它不把组件测试计成新的模型运行。

最终 NInfer 接入下归档组件测试通过，八个既有原生程序前后逐字节相同，继续复用 C5 对应二进制的运行证据。当前 [组合合同](contracts/kv-lifecycle-source-archive-candidate-v1.json) 冻结 125 项归一化输入，Linux 实跑包另冻结 104 项精确 core 输入；source-only gate 不冒充 Linux 测试或原生补丁全量重放。预检初次误用不在本候选中的默认 profile，随后又因重复输出目录被 runner 拒绝；最终显式选取原冻结 profile 和新目录后通过，未修改源码或降低断言。原始输入、各轮日志和容器退出状态在 `artifacts/lifecycle-source-archive/evidence.zip`，SHA-256 为 `c267eba79506a218d10c97801ab09626982825c326958954284ed7ac771052a9`。

可在 Linux 主机构建目录复现：

```sh
cmake -S . -B build-source-archive -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_CUDA_TESTS=OFF
cmake --build build-source-archive --target source_archive_test source_image_test \
  source_journal_test source_recovery_test spill_io_test -j
ctest --test-dir build-source-archive --no-tests=error --output-on-failure \
  -R '^(source_archive_test|source_image_test|source_journal_test|source_recovery_test|spill_io_test)$'
```

ASan/UBSan 使用 `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`、同样 linker flags，运行时启用 `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`、`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`。源文件按实际冻结字节复现，故障注入通过测试可执行文件链接真实实现并 wrap 系统调用，没有向公共 API 添加测试开关。

## 接线落实程度与下一步

1. C7 已实现受 S 约束的原子源镜像导入，建立当前新根、computed 重置为零；归档 key 的新 RAM binding 只接受当前有效 read receipt，未接 Engine。
2. C7 已将保存新旧顺序改为持久 blob 序号，独立 exec 覆盖 counter 数值重复与 revision 重置后继续保存；SourceRef 只在当前 RAM owner 内校验。
3. 再接空闲 Engine 的保存/读取/重建入口，先验证持久撤销，再按现有 C4 顺序撤销所有在线别名。恢复失败保持已提交源或明确回滚新源，不发布半成品。
4. 最后用实际模型做两进程保存/恢复、已回收 CPU 页后的新域重建、晚到保存/读取中遗忘及损坏/容量拒绝；仅数据组件成功不能关闭阶段 C。

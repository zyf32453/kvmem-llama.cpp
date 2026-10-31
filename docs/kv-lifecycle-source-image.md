# 源记录镜像：离线组件与后续接入边界

日期：2026-10-31。本批从 C4 提交 `57a2f514baf73d7b6b5a08fca5bdd2b27aa67739` 建立独立离线分支。SSH 恢复前先实现不依赖 GPU 的源记录镜像组件；尚未接入 Engine、归档清单或持久恢复。另一个工作树中的 C5 checkpoint 预算候选仍待取得远程编译和测试终态，本分支没有包含或验收它。完整阶段 C 和生产系统都未完成。

## 本批解决什么问题

此前 `SourceJournalStore` 保存已接受的 token 和事件，只提供 owner 串行的同步复制读取。若直接持有引擎锁写大文件，磁盘等待会拖住引擎；若把未计费的完整向量交给后台，又会绕过内存预算。

新增 `SourceImage`，复用已有 `HostKvStorage` 的预约、真实 buffer 和最后 owner 退出退款。`prepare_source_image` 在 owner 串行区复制一份受 R 约束的不可变数据；完成后，调用方可以把这个对象移动给后台写入任务，写入过程不再读取原源目录。没有新建线程池、历史目录、allocator 或自动清理策略。

这里保存的是恢复所需的原始 token/event 数据，不是 CPU historical KV 页、GPU 工作集或可直接恢复的 checkpoint。源 token 仍在不代表派生 KV 已存在，也不重新授权已退休的 KV 身份。

接口见 [source_image.hpp](../kvmem/include/kvmem/source_image.hpp)，实现见 [source_image.cpp](../kvmem/src/host/source_image.cpp)。

## 三条数据路径

1. `prepare_source_image(store, current_ref, limits, R)`：核对当前源 revision 和格式上限，一次预约 token/event 两个真实 RAM buffer，复制后再次核对源引用，成功才返回不可变对象。调用方负责与源目录 mutation 串行；失败保留 S 中的原数据并归还临时 R。
2. `write_source_image(image, limits, sink)`：从不可变对象流式编码。对象必须在调用期间存活，同一个对象不能同时被移动或销毁。无论 sink 成功与否，R 都随实际对象所有权保留。另保留 `write_source_image(store, ref, ...)`，供调用方主动选择 owner 串行导出；它不适合持锁执行长时间磁盘 I/O。
3. `read_source_image(source, bytes, current_configuration, limits, R)`：先核对固定头、声明长度、配置及数量上限，再预约两个实际 buffer。逐块解码、验证事件与完整校验值后才返回对象。没有向源目录插入记录、创建模型域或发布 checkpoint 的副作用。

传输由调用方提供。sink/source 回调必须完整处理本次请求的字节数，失败抛异常，不能静默短写或短读。每次回调最多 1,024 bytes，内部固定栈缓冲不随源长度增长；函数不积累完整编码向量。测试中的向量只是测试传输，不是组件内部的额外无界副本。

## 格式、配置和语义校验

所有整数显式使用小端字节序，未直接写 C++ struct 或 padding。格式版本为 1，顺序如下：

- 8-byte magic：`KVSIMG1\0`；版本和完整文件长度各占 uint64。
- 原始及父 `SourceRef`，各为三个 uint64；父前缀、accepted frontier、computed frontier、事件数各占 uint64。
- 32-byte `SourceConfiguration`。
- accepted frontier 个非负 int32 token ID。
- 每个事件 26 bytes：id、begin、end 各为 uint64，kind 与 complete 各为 uint8。
- 8-byte FNV-1a-64 footer，覆盖此前全部字节。

总长度严格为 `144 + 4 * token_count + 26 * event_count`，解析会拒绝溢出、截断、多余字节、未知版本、缺失身份/配置、非法父关系、computed 超过 accepted、负 token、非法事件类型/布尔值、空事件和事件重叠。延续已有源记录契约：事件 ID 不要求唯一或单调，也不擅自把“非末尾 incomplete 事件”判为非法。

配置 digest 必须与调用方当前配置完全一致；这个比较只允许解释源数据，不证明任何派生 KV 可复用。footer 复用现有 SnapshotReader/Writer 的 FNV，仅用于意外损坏检测，不是密码学认证，更不是退休或遗忘许可。

镜像中的 `SourceRef` 是来源注释。现有 id/incarnation 是进程内身份，不得在重启后当作当前引用使用；父引用也不证明归档中的整个家族仍存在或合法。

## 预算和失败行为

独立 `max_bytes / max_tokens / max_events` 限制传输与处理工作量。R 对实际 `token_count * sizeof(int32_t) + event_count * sizeof(SourceEvent)` 计费，包含正在解码、已经返回以及交给后台尚未销毁的所有对象。编码长度和 R 长度不同，不能拿磁盘格式大小代替实际 RAM 占用。

预约发生在两个 buffer 分配前。额度不足返回 `BudgetExceeded`，分配失败保持异常；两种情况都不删除原源、不调用普通历史退休来偷换“永久遗忘”。已准备和已解码的对象共享同一调用方 R 账户；移动不重复计费，覆盖移动先归还旧对象，最后 owner 销毁才退款。账户可随 buffer 存活，不依赖外层 HostKvStorage 包装对象的寿命。

本组件要求 RAM-only staging，发现带 SpillFile 的账户会在分配前返回 `Unsupported`。这来自本地检查发现的实际接口限制：HostKvRecord 的同步 `data()` 不接受分层记录。SSD 归档由流式传输处理，不在这里隐式把解析缓冲变成冷存储。

R 不包含固定 SourceImage 对象、HostKvStorage 的 account/control 对象、allocator 开销、固定栈缓冲或调用方文件/传输缓冲。这些边界与现有恢复 staging 一致；它不是整个进程的 RSS 上限。未来任务队列还必须另有限长和自有 metadata 预算，不能只限制单个镜像大小后就允许无限排队。

失败的写入可能已经产生部分字节，组件不负责删除、重命名或提交文件。部分输出必须保持未发布状态。没有文件存在即恢复、保存成功即删除源、或校验成功即重开旧历史资格的路径。

## 持久化接入必须满足的契约

以下是下一批的接口约束，尚未实现，不能据此宣称已经可跨重启恢复：

接线先冻结为当前 NInfer/Linux、单 Engine owner、文本 token/event、显式新域重建。后台传输复用 `StorageIoQueue` 的 jobs/bytes 上限及完成票据，不建立第二个线程池；完成票据只证明 I/O 结束，不直接发布归档资格。现有 `SpillFile` 的合同明确为 ephemeral/process-local、无 crash/restart 保证，不能把临时 extent 和 checksum 升格为持久恢复权威。

持久清单使用自己的稳定 namespace、单调 ID/revision 和已提交 high-water mark；映射到现有 RAM 源目录的当前引用。删除记录后不复用旧 durable ID，缺失记录即拒绝授权，避免新增无限增长的逐块墓碑表。清单打开失败不能通过扫描 blob 自动重建资格；文件扫描仅用于核对用量和清理孤立文件。读取 lease、排队写入和临时文件均应在实际所有权退出前保持相应 R/D 计费，磁盘额度不足不自动永久遗忘源。

1. 在现有 owner 上建立有界的持久归档清单和 durable identity 映射。它记录源 lineage、提交序号、配置、内容摘要、归档位置及当前遗忘/版本 fence。进程内 SourceRef 不充当 durable key；不另建与现有历史目录争夺权威的第二套 KV 目录。
2. owner 从当前源发出带 revision/fence 的保存任务，任务持有实际计费的 SourceImage。工单须有 owner 产生的许可；解码镜像里的 origin 只是来源注释，不能因它的 ID 与当前源相同就冒充保存权限。后台仅写未发布对象并返回完成回执；完成回执进入 owner 的有界交接队列，worker 不自行修改源、退休状态或归档资格。
3. owner 在提交清单前重验任务身份、内容版本、配置与持久遗忘 fence。源数据快照需要区分“同一源的追加”和“身份/内容失效”：已有 SourceJournalStore 保证 accepted token 前缀不可改写，因此同一 owner/source、配置和遗忘 fence 均有效时，较早的不可变前缀可以作为明确的旧版快照发布；不能因每次末尾追加就永久饿死后台保存。新任务不得用较旧版本覆盖较新的已发布快照；过期身份、已遗忘、取消或不完整写入只成为待清理的孤立对象。这个规则不放宽现有 C3 的当前 SourceRef 准入校验，也不授权旧 checkpoint；恢复方必须明确知道快照的保存 frontier，不能冒充最新接受状态。清单更新与进程重启后的恢复次序必须有明确的原子提交和持久化保证，不能把 rename 等同于所有平台上的掉电保证。
4. ForgetSource 先持久关闭家族的后续发布和恢复资格，再撤销在线别名及清理归档；故障时宁可留下不可恢复的待清理文件，也不能留下仍可授权恢复的旧清单。遗忘 fence 不能因对象文件删除、目录槽位复用或重启而倒退。
5. 导入前先查询权威清单及当前配置/遗忘/退休规则，再用本 codec 解析数据；发布前再次核对。显式源重建取得新身份和正常资源准入，不能把旧 origin 直接装回 repository，不能把 computed frontier 当作已有 KV 覆盖证明。派生 checkpoint 恢复仍需要自己的完整覆盖与 MemoryStamp 校验。
6. 缺失/损坏权威清单、旧清单回滚、晚到 writer、读取中遗忘、配置变化及清单提交失败均须明确拒绝并回收暂存。单个镜像通过 checksum 不允许绕过这些条件。普通内存压力仍只回收允许的副本或退休历史，不自动永久遗忘原始恢复依据。

已有 C4 遗忘只覆盖当前 Engine。源镜像是尚未接入 Engine 的独立数据 owner，因此本批测试特意保留一个边界事实：源目录局部 forget 后，一个早已取得的镜像仍能写完数据，R 也不会提前退款。这不构成合法的归档发布或恢复；接入前必须落实上面的持久 fence，才能把它纳入系统级 ForgetSource。

## 本地验证与复现

[source_image_test.cpp](../kvmem/tests/source_image_test.cpp) 覆盖内存与真实临时文件往返、固定小端字节、256-token 分块边界、分支/空源、精确额度与共享账户、移动及最终退款、当前配置与 stale 引用拒绝、prepare 后源变更/局部遗忘、独立异步任务写入和账户包装对象退出后的实际数据寿命。

损坏测试对 2,274-byte 样本逐字节翻转，以及逐长度截断；另构造 11 类重算 footer 后仍不合法的语义镜像。故障注入逐一中断读取、直接导出和镜像导出的 33 个传输回调位置，并枚举 prepare 与 decode 的实际内存分配失败点，检查旧源和真实计费保持一致。

执行结果及源码/二进制身份见 [本批摘要](experiments/2026-10-31-lifecycle-source-image-offline/summary.json)。本地 Windows/GCC Debug 的结果只归属公共组件和主机契约。SSH 恢复后，另在固定构建镜像的 Linux/GCC 13.3、Python 3.11.18 下对整个公共 kvmem host library 和 source_image_test 启用 ASan/UBSan；启用 detect_leaks 后的一项组件目标全部通过，实测 0.28 秒。已核对动态链接包含 libasan.so.8 与 libubsan.so.1，所有 102 项输入与 Windows 冻结源码逐字节一致；两种环境的故障注入数量一致。

新增 Linux 原始日志、JUnit、运行脚本与 CMake 配置保存在 `artifacts/source-image-offline/linux-evidence.tar.gz`，SHA-256 为 `ba3191f19801173697b516373fb9c05cd31efaf708fe97aa2b97ecbfb6ade1eb`。输入清单保留初次 Windows 冻结时的状态；后续资格以执行摘要为准。尚未运行本组件的 CUDA、真实模型、掉电恢复、性能或 Agent 检查，尚无 Engine/持久清单接线；本次 sanitizer 通过不代替 C5 的预算或模型验收。

本批最终重新构建并运行 23 项主机契约，全部通过，CTest 总耗时 31.52 秒。prepare/decode 各覆盖 4 个真实分配失败点，三条传输路径共覆盖 99 次回调故障。原始证据保存在 `artifacts/source-image-offline/evidence.zip`，SHA-256 为 `550bbfcb8e6ca9a6e5f5b8143b2b3c9e34f485b420e90f3656c8a5f6df01222a`；[离线组件清单](contracts/kv-lifecycle-source-image-offline-v1.json) 冻结 102 项主机/core 源输入，未冻结或验收新的原生 GPU tree。源码 hash 按实际输入字节计算；Git 换行转换可能改变 checkout 字节，逐字节复现应使用证据包内的冻结副本。

可在独立主机构建目录复现：

```powershell
cmake -S . -B build-source-host -G Ninja -DCMAKE_BUILD_TYPE=Debug -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_CUDA_TESTS=OFF
cmake --build build-source-host --target source_image_test --parallel
ctest --test-dir build-source-host --output-on-failure --no-tests=error -R '^source_image_test$'
```

完整本批主机集合为 C4 manifest 的 22 项 host_tests 加 source_image_test；实跑命令、JUnit、输出及源文件冻结在本地证据包内。

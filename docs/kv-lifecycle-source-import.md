# C7：原子源导入与跨进程续存

日期：2026-10-31。接续 [C6 源归档](kv-lifecycle-source-archive.md)，本批实现源镜像到 S 的原子导入、当前归档读回执授权的 binding，以及重启后保存的持久顺序。公共组件已通过 Windows/Linux 功能、Linux ASan/UBSan/泄漏和四次独立 executable 检查；原生 C++20 回归及 GPU 结果见本批执行摘要。C7 当时尚无 Engine 归档入口；后续 [C8](kv-lifecycle-native-archive.md) 已接通并提供独立模型证据。下文保持 C7 原候选归属，阶段 C 和整个 A–F 目标仍未完成。

## 修改前后

原先归档只能返回 R 中的 SourceImage 数据和读回执。重开后没有 RAM binding，也没有把镜像转成 SourceJournalStore 的入口；若调用方自己先转成完整 vector，再 capture，会产生没有纳入 S/R 的大副本。此外，C6 的保存新旧比较只适用于同一存活 RAM binding，不能用于重启后复用同一个 durable key。

现在增加两条不同层次的接口：

- `SourceJournalStore::import_image(image)`：只复制源数据，建立新的 RAM 规范根。它本身不授予归档许可，不复活任何旧 KV、checkpoint 或源引用。
- `SourceArchive::import_source(read_receipt, store)`：先验证当前 owner 签发、仍指向当前已发布版本的 read receipt；S 导入完全成功后，再把原 durable key 绑定到新 RAM 源。已有 binding 返回 Pending，不静默覆盖。过期、已遗忘、已被新版替换、foreign 或 moved-from 回执都不能导入。

实现分别位于 [source_journal.cpp](../kvmem/src/host/source_journal.cpp) 与 [source_archive.cpp](../kvmem/src/host/source_archive.cpp)，声明见对应公共头文件。SourceImage 仍然只是数据对象；没有根据其 origin 数值自动取得权限的路径。

每个 durable key 在同一 owner 内只绑定一次；即使调用方单独执行 RAM store 的局部 forget，组件也不会自动解除旧 binding。当前需关闭/重开 owner，或由后续明确的 detach 协议处理；不能用重复导入静默替换源。这项限制不影响分配/配额失败后的重试，因为失败不会建立 binding。

## S 的原子性和真实资源

`import_image` 先核对镜像状态、源数量及 token/event 上限，再按已有 SourceJournalStore 的 buffer capacity 规则，一次预约两份真实 S buffer。两个分配成功后，将 token 以 256 项固定栈缓冲逐块复制，事件逐项复制并检查。整个准备对象留在目录之外；所有可能失败的工作完成后，才发布源条目和推进 ID。

额度或分配失败不留下半条目，不消耗源 ID，不改变旧源及其计费；正在读取的完整镜像仍由原 R owner 计费。这里没有完整临时 vector、typed raw-byte alias、重复线程池或新 allocator。S 包含 token/event 的实际 capacity 和已有固定目录，R 包含镜像；固定对象/allocator/运行时开销仍按原合同在外，不能说成全进程 RSS 上限。

导入保留配置、接受的 token 与所有事件事实，包括重复事件 ID、非末尾 incomplete、已裁剪分支事件；创建新 SourceRef，revision 从 1 开始，parent 为空，computed 为 0。旧 computed frontier 不代表新 Program 已有 KV。durable parent 继续保存在归档清单中，不能把旧进程 SourceRef 当作当前 RAM 父引用。

这种 RAM 根与 durable lineage 的差别必须由下一批 Engine 统一遗忘处理：不能只用 RAM parent 遍历，就声称已经撤销全部持久家族。本批归档 forget 仍只处理归档资格，已经导入的 RAM 内容不会被它自动删除。

## 版本顺序使用持久序号

保存工单只能从 owner 当前验证的 SourceRef 准备，源接受内容只允许追加；blob ID 在 worker 创建文件之前已经持久提交，单调且不重用。发布改为按这个持久序号拒绝早到/晚到的旧工单。当前 RAM revision 仍用于校验同一存活 binding 的源，但不再与磁盘上另一个进程的 revision 比大小。

独立进程测试刻意构造了真实反例：第一次运行保存 revision 4；重启后 RAM counter 重新从 1 开始，新源的 id/incarnation 数值与旧来源相同。当前有效 read receipt 批准导入以后，新 revision 1 追加为 2，再保存必须成功；旧的 `4 > 2` 比较会错误拒绝，持久序号能够正确接续。

SourceRef 的“新身份”是新的当前 owner 中由正常目录发放的身份，不是跨进程全球唯一数值。镜像中相同数字仍只是来源注释；当前回执、namespace/key、owner binding 和正常新域准入共同决定允许执行的动作。没有把旧 native domain、KV 地址或退休 generation 安装回系统。

## 四个独立进程

`source_archive_test` 通过 fork 后 exec 当前可执行程序，实际运行四个互不共享 RAM store 的进程：

1. 创建归档、保存 revision 4，并把 durable key 作为客户端句柄保存在归档目录之外。
2. 打开归档、验证回执、导入为当前新根，追加并保存更大的 durable 序号。
3. 再次打开和导入，核对追加的实际 token，持久遗忘；旧 reader 马上失效。
4. 再次打开，旧 key 被拒绝；剩余 blob 只允许作为孤儿清理，不成为恢复资格。

这证明的是源数据、授权和续存，不是模型推理跨重启恢复。Linux 检查器也执行同一流程；强制 `_exit` 的故障子进程依然不执行析构或 LeakSanitizer 退出检查，不能把它们记成退出泄漏检查通过。

## 验收与不利结果

Windows/GCC 16.2 Debug 四项目标（image/journal/recovery/spill）全部通过。Linux/GCC 13.3、Python 3.11.18 的五项目标（再加 archive）在 Debug 与整个公共 host library 的 ASan/UBSan/泄漏检查下通过。新增普通导入和归档授权导入各枚举四个实际分配失败点；另覆盖精确 S 额度、少一字节、行/事件/目录上限、重复 binding、移动/过期回执、分支父关系以及 durable 家族遗忘。C6 的文件错误、清理计费及进程退出断言全部保留。

公共头文件变化触发了 77 个原生编译/链接步骤，八个既有程序中七个二进制发生变化，因此没有沿用 C5 原二进制结果来证明当前模型回归。当前原生候选重新执行 12 项契约与两项组件检查，并使用原来的七组 GPU 负载和强断言做既有功能回归。它们不覆盖 Engine 归档入口，因为该入口尚不存在。

首个原生任务的编译和 14 项检查通过，但最后输入闭环核对失败：验收脚本改写了已冻结的 `inputs.json`。修正方式是先生成当前元数据、再冻结并在结束时核对同一份字节；相同代码/编译结果在新目录复验，源断言、模型与工作量均未放宽。失败现场和旧日志保留，不能把首个任务写成整体通过。

最终重新验证 3,446 项精确输入和 125 项合同输入，12 项原生契约、两项组件检查和七组新 GPU 回归全部通过。四组 LRU/redundancy × 普通/MTP4 压力各发生八次 checkpoint 触发回收，384 MiB 的 C 峰值分别为 308,084,854/308,089,986 bytes；1 MiB 拒绝后的 Engine 仍可使用。C3 前缀页实际释放与 C4 三个源/三个域撤销断言均保留。测试使用显式 1/384/2,048 MiB 档位，没有另跑默认 1 GiB GPU 档位，也没有性能或 Agent 质量结论。

复验构建只执行 build-id stamp，没有再次编译或链接 C++；导出的原生目录没有 `.git`，已有 stamp 提示 `build unknown`，本批以固定源码/补丁树、实际二进制及模型 hash 确认归属。整理记录时纠正了对 Ninja “no work”文字的假设，并补齐 Docker stderr 与时间戳，确保失败 Traceback 被原样保留；没有因报告脚本的错误重复运行或改动已经通过的测试。

本批精确输入、输出、当前二进制、容器状态和最终结果见 [执行摘要](experiments/2026-10-31-lifecycle-source-import/summary.json)。当前组合合同是 [source-import-candidate-v1](contracts/kv-lifecycle-source-import-candidate-v1.json)，source-only gate 仅验证源码与补丁身份，不能替代运行测试。

原始证据保存在 `artifacts/lifecycle-source-import/evidence.zip`，SHA-256 为 `5afa13ca74c2e0072f1bc72f8e4b966a37163008bbf3694e8adec337de1a785e`。执行时合同与验收后补充结果元数据的合同分别保留，各自源码 pin 完全相同；125 项源码/30 个补丁的最终 source-only 核对通过，没有新增补丁全量重放或改变测试工作量。

仍未验证或交付：Engine 归档 attach/save/import、从归档驱动的真实模型新域重建、持久与在线别名统一遗忘、checkpoint 持久化、长源重放、完整性能/长时间/Agent 验收。C6 的 catalog/seal 不一致时拒绝打开、完整一致目录备份回滚无法识别、同步元数据 I/O 成本等边界继续有效。

下一批将把这些公共机制接进当前单 GPU、单 Engine owner 的原生控制面。配置校验必须由实际 Frontend/Program 重新产生；不能仅相信归档携带的 digest。尤其原生源 digest 包含 raw/chat、模板参数、thinking、KV/模型及执行配置，不能用任意 raw 示例冒充不同 chat 配置。

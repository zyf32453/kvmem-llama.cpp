# Checkpoint 持久恢复：权限组件与原生保存候选

更新：2026-11-01。C11a 的**归档权限组件**已验收；C11b 已实现原生编码及 Engine 保存候选，
本地 CPU 检查通过，最新原生完整编译和真实模型检查因 SSH 再次中断尚未完成。
后续 C11c 已补私有 CPU 读取准备组件，C11d 已补实际逻辑对象及其 C/M 准备，均只有本地组件证据；
C11e 已补私有原生 Window/目录及 owner 转移，CPU 数据装配检查通过；
C11f 已接通 Engine CPU checkpoint 导入候选与新身份发布协议；C11g 已补请求级显式续算候选及源分支，
完整 Linux/模型恢复仍未验收，
阶段 C 与完整六阶段目标继续开放。
执行范围、结果和不利发现见 [本批摘要](experiments/2026-10-31-lifecycle-checkpoint-authority/summary.json)。

## 为什么不能直接复用普通 session 保存

NInfer 已有 `session_snapshot.cpp`，保存 catalogued continuation 的 KV、StateImage、ledger 和
checkpoint 目录；它没有保存 `KvmemWindowState` 的历史退休页、稀疏页映射、Host 版本、检索/
反馈状态和源归档权限。现有 `checkpoint_image.hpp` 则只是进程内 SpillExtent 编解码，整数
identity 不是跨进程恢复许可。二者都不能直接作为 GC 历史的持久恢复接口。

本批先把恢复权限落实到既有 `SourceArchive`，继续复用目录锁、source namespace/key、已提交
源 blob 序号、固定对象表、共享 M/D 计费及 `StorageIoQueue`。没有另建 checkpoint 数据库、
存储池或后台线程。模型状态编码由 NInfer Program 负责；C11b 候选尚未完成原生验收。

## 一份 checkpoint 绑定什么

每个附件严格绑定 `(namespace, source key, 当前已提交 source blob 序号, backend schema)`。
只比较 RAM SourceRef 或磁盘文件名不够；读取必须使用当前 SourceArchiveRead，发布还要核对
当前 RAM binding/revision 与已发布源快照一致。更新源快照或永久遗忘源家族后，旧读取许可失效。

一个 source blob 最多发布一份不可变附件。更新 checkpoint 时，先发布新的源快照，取得新的
单调 blob 序号；即便 RAM revision 和 token 数据没有变化，也不覆盖旧附件。这样可复用既有
持久版本时钟，避免再维护一套可回退的 checkpoint 当前指针。

附件 framing 为小端固定头和校验尾，总长度 `104 + payload_bytes`。头包含上述稳定身份及
非零 256-bit backend schema，载荷由 backend 定义；FNV 完整性校验沿用可信本地文件系统模型。
这不是原生模型状态格式，schema 的完整性与模型数学语义仍须由后续 adapter 验证。

## 保存、发布、读取和退休

- `save_checkpoint`：要求完整当前源快照，先检查 frame/D/队列完整占用上限，保留调用方已计费
  的不可变载荷，后台只写 `checkpoint-pending.N`。producer 过长或不足都会失败，不产生读取许可。
- `publish_checkpoint`：I/O 完成后由同一串行 owner 再验源版本、当前 blob 和 binding，才 rename
  为 `checkpoint.N` 并同步目录。仅 worker 完成不允许恢复；转冷后迟到的旧 writer 不能发布。
- `read_checkpoint`：先验证当前源回执、附件身份、schema 和长度，以固定 8 KiB 栈缓冲流式交付
  给调用方的**私有未发布空间**，最后验证完整校验和并再次复核权限。sink 可能收到未校验字节，
  不能提前改写正在服务的 GPU 状态或发布 native 域。
- `retire_checkpoint`：独立撤销派生状态，保留源数据。writer 未完成时返回 Pending；成功后旧
  read/write 回执失效。文件改名为 `checkpoint-retired.N`，不再允许同一 N 重新发布。
- `collect`：等待对象的所有句柄和 I/O 退出，再回收额度与文件字节。仍为当前源的退休附件会
  缩为零字节 tombstone，保持不可逆退休；源 blob 换版或遗忘后，连同旧源对象一并清理。

源转冷保留磁盘数据和读取资格，但删除旧 RAM 身份；后续模型恢复必须建立新身份，不能把磁盘
元数据里的旧 session、SourceRef 或 MemoryStamp 当作本次进程的许可。永久遗忘则立即关闭
整个持久家族的读取许可，文件和 D 额度仍按真实 owner 退出结算。

## 资源边界与失败行为

`max_checkpoint_bytes=0` 默认禁用附件 I/O；正值限制完整 frame。附件与源对象共享已有槽位和
D 总额度，不占新的无限目录。新增对象字段按实际 `sizeof(Object)` 进入已有 M 表计费，即使
功能关闭也有这部分固定元数据成本。零字节 tombstone 的目录项由既有对象数量上限约束。

Core 不分配完整 checkpoint staging。调用方必须实际保留并计费不可变 C/H/M 载荷，声明真实
`retained_bytes`，队列按它做完整准入；这个声明不能代替后续 native adapter 的真实预算。
读取目标也由调用方先取得预算并保持私有。已验收的归档组件仍只覆盖小型载荷；C11b 的
数百 MB/GB 原生保存候选尚待模型验收，不能将实现或本地持有关系测试当作大载荷验证。

写入失败或队列拒绝留下有界未发布 claim，待 collect 清理；未完成文件不能读取。发布发生
rename/目录同步错误会使该 owner 不健康并返回未确认失败。重新打开时，已完整 rename 的
附件可能有效，不能把未确认失败报告成“写入必定没发生”。源数据仍由原 catalog/seal 控制。

源 catalog/blob 仍为 v1，新实现可读旧源目录；禁用附件功能时仍会计入附件实际 D 占用。
加入新文件名后，旧实现会拒绝打开该目录，因此不承诺向旧程序降级读取。完整目录备份回滚
仍需外部单调锚点才能识别；进程退出测试不等于硬件断电验证。

## 验证与不利发现

组件测试覆盖流式 payload、后台完成未发布、真实载荷 owner 释放、D/队列/frame 精确边界、
producer 错长、schema/损坏/截断拒绝、转冷后迟到 writer、源换版及永久遗忘、读取中撤销、
退休 tombstone 与无 checkpoint 的源数据恢复路径。独立 exec 验证保存、重开读取、退休、
再次重开拒绝；实际系统调用注入覆盖后台 I/O、发布错误及进程中断，分配失败也检查退款。

第一轮 24 项公共组件及同 24 项 ASan/UBSan/泄漏检查通过。补充边界后，对最终归档测试在
Debug/内存检查中重跑；其余 23 项的相关代码未变化，沿用对应结果。NInfer C++20 Release 的
同一归档组件也通过，**没有构建或执行本批模型恢复程序**，不能沿用旧模型二进制作恢复证明。

整理时发现一次核心重跑的脚本把新测试覆盖到 `/work/kvmem`，而构建实际使用
`/work/source/kvmem`，该次结果已作废。修正后核对 Ninja 的实际翻译单元路径、源码摘要并
重新运行；原错误 overlay、实际旧源码、日志和作废标记保留。另修正了“配置上限变小”被误报
为文件损坏的问题，现以 BudgetExceeded 拒绝并保留合法源数据。

## C11b：原生保存候选与本地验证

候选增加 `Engine::begin_historical_checkpoint_save`，由已有
`finish_historical_save` 分辨工单类型后完成 owner 发布；
`retire_historical_checkpoint` 只撤销归档附件，保留规范源，不把它当作 RAM 页 GC。
`HistoricalArchiveOptions::max_checkpoint_bytes` 默认 0，须显式启用；首批实现限定 Linux、
单 GPU/单请求、文本/RAM、none/MTP，以及关闭旧 context cache 的 KVMem 路径。
这些是实现范围，当前没有新的生产或模型验收结论。

Program 在空闲 owner 下冻结 retained history：复制原始位置、退休/保护标记、Host 版本、
Mean-K、selector/LRFU、图/反馈/thinking 状态和有限 checkpoint 目录；共享仍合法的 Host
KV/scale 与去重后的完整 main/MTP/recurrent StateImage。已退休但尚有物理读者的页不进入
新载荷；引用缺失或已退休 resident 页的旧 checkpoint 不作为合法恢复点写出。
这不修改在线目录，也不把文件内旧 SourceRef/MemoryStamp 当作未来恢复许可。

不可变 metadata 的内存副本先进入 C 配额，H 和 pinned image 保留原有真实 owner。
队列按完整 C/H/R 持有上限准入，修正原先只够单批 KV 搬运的队列上限；没有新增存储池或
I/O 线程。保存有同步元数据复制、前缀核对和额外 D 写入成本，未测性能。

生命周期检查发现，单把 pinned image 的 shared owner 交给 worker，可能让最后释放在文件
线程上间接执行 CUDA。现以一个有界 Program export lease 持有冻结对象；worker 释放全部引用
后才发信号，Engine 的既有维护/控制路径再释放最终 owner。Program 关闭时先 join 队列。
真实 C/H 占用会保留到实际完成及 owner 清理，I/O 完成不能冒充所有资源已退款。

逻辑快照读取器只构造独立、付费的私有特征/范围对象，不保存旧 worker 的冻结视图和 epoch
certificate。Windows 上逻辑快照、既有 native-page GC、真实 export lease 三项通过；后者覆盖
成功/错误/取消及 5 个真实分配失败点，证明相同持有关系的最后销毁发生在 owner 线程。
该证据使用 CPU 跟踪对象，不能代替 CUDA 固定页或真实模型验证。

本地首次构建修正了测试声明问题；owner 检查又碰到 Windows 路径过长，改用同一 workspace
内较短构建路径后通过。远程已观察到相关 9 项公共组件及同 9 项内存检查完成，首次原生构建
发现新工单类型缺少前置声明并已修正；重跑期间 SSH 握手持续重置，用户也确认无法登录。
最新 owner 修正尚未上传，新原生完整构建/模型结果均不作通过处理。

代码、补丁回放、本地证据与未验收项见
[C11b 候选摘要](experiments/2026-11-01-lifecycle-native-checkpoint/summary.json)。

## C11c：有预算的私有读取准备组件

已补 NInfer 模型专用的 `KvmemCheckpointInput`：沿用 C11b 文件格式，先读取固定头，申请实际
metadata 目标，再以无堆分配的只读视图校验完整元数据。检查源快照身份、模型/布局、原始位置、
退休标记、Host 版本、统计/图/反馈、特征与 thinking 范围，以及所有 checkpoint/目录/镜像引用。
非法或缺失依赖在准备大载荷目标之前拒绝；出错的输入不能继续接收或报告成功。

之后先预约整组状态镜像和实际 H 页，流式数据直接进入私有目标，不构造另一份完整 opaque
checkpoint。模型侧 `read_historical_checkpoint` 使用原有 C pool 与 H reservation；在完整外层
校验及当前 SourceArchive 权限复核成功后，才返回私有准备对象。它没有导入 S、分配新 native
身份、创建在线 history 或上传 GPU。后续发布者必须再次复核许可，并落实私有 typed 元数据的
M/C 准备与正常执行资源准入，因此不能把该组件称为完整恢复入口。

沿现有 prefill/restore 代码核对了两项易误判语义：开放输出的结束位置使用模型容量；旧
checkpoint 的图消息边界可能包含已丢弃后缀，正常恢复会重新绑定当前请求。读取按容量约束
这些边界，不把它们当作当前源的读取许可。另校验尾页 Mean-K 的有效行数，避免损坏数值
进入后续 float-to-token-count 恢复。

独立固定数据与实际 canonical source、RawKvStore、C/H storage 的 Windows C++20 测试通过：
逐字节/7-byte/8-KiB/整块四种输入一致，6,568 个截断位置均拒绝，覆盖错误模型/位置/退休引用、
默认与 coverage 目录、空图、缺省特征、共享镜像、配额边界、失败后退款和持有读者的延迟退款。
12 个真实分配失败点也退款；已用禁止堆分配的检查确认元数据遍历不分配堆内存。

该证据使用小型已知状态字节，不是实际 StateImage 数学或 CUDA pool 的恢复验证。私有
NInfer adapter 已写入但尚未完成完整原生构建；Engine 发布、GPU 上传、新域身份和模型续算
均未完成。没有性能或全进程 RSS 结论；C pool 的实际空闲镜像仍可能保留。尝试本机已有
Docker 补跑 Linux 内存检查，但后台未就绪，未修改 WSL/驱动设置，Linux sanitizer 仍待运行。
见 [C11c 组件与验证范围](experiments/2026-11-01-lifecycle-checkpoint-input/summary.json)。

## C11d：独立逻辑状态及其 C/M 准备

字节视图之外，现已增加模型专用的 `KvmemCheckpointLogicalState`：还原实际 `RawKvStore`、
`KvMemStore`、LRFU、图、输出反馈/采样、请求与 capture plan、特征/thinking 范围，以及每个
checkpoint 的 token、三轴位置、执行切分、resident 页、部分 Mean-K 和状态镜像索引。
这些对象独立拥有数据；以后可移动到原生 continuation，不需要再次复制全部逻辑内容。
该类型没有当前 SourceRef/MemoryStamp、worker、GPU 地址或执行许可，不能直接作为可运行
Window 发布；源/目录/权限仍由外围私有 read 保留。

新增无堆分配的容量计算，按实际编码中的块数、图行/边、消息、反馈和前缀长度预约 C；共享
控制对象使用保守余量。Raw mean-only 表复用现有 H account，但不另存 packed KV。模型侧
先为整组 typed metadata 预约 C，再申请状态镜像/H 大目标；完整文件校验后才构造逻辑对象。
特征与范围的实际数组继续由原有 M allocator 计费，M 中途不足也整体放弃私有准备。C charge
先声明、后析构，覆盖对象清理；没有新增线程、通用注册表或公开恢复接口。

Core 增加两项必要支持：mean-only 快照的无分配容量计算，以及仅允许空私有 selector 接收的
逻辑统计恢复。后者保留三种评分和 LRFU，验证几何、数值及 runtime budget；旧 tier slot、
in-flight、dirty 与 working-set 标记均丢弃，拒绝覆盖已有非空 selector。

Windows GNU16.2.0/C++20 下三项检查通过：扩展后的实际输入/逻辑准备组件，及现有
`kvmem_store_test`、`raw_kv_store_test`。独立 fixture 验证所有还原字段，并在清空原始 wire
metadata 后继续访问；沿用的 6,568 个截断、12 个输入分配失败检查保留。新覆盖 79 个实际
普通堆分配失败点、7 个实际 M 上游分配失败点、C/M 精确与少一字节拒绝、失败退款和旧物理
标记清除。rich/legacy 两个小 fixture 的 C 预约分别为 9,601/8,377 bytes，测得普通堆峰值
8,201/7,217 bytes；M 数组独立计费，rich fixture 实际 M 上界为 344 bytes。这些数字只证明
相应小 fixture 的对象准备，不代表真实模型或完整进程的内存上界。

成本是 wire metadata 与 owned logical metadata 在准备期间同时存在，C 还保守保留共享控制
及临时对象余量；逻辑解码额外遍历 metadata，延迟/RSS 未测。Linux 完整 Native adapter、
实际 pinned pool/GPU、大模型状态、重启续算及 sanitizer 仍未验证，旧 C10 模型结果不作为
当前代码的证明。仍需把这些私有数据装配成原生 Window/checkpoint 目录、绑定正常 graph
和执行资源预算，重新准入并以新身份发布。
见 [C11d 代码、结果及范围](experiments/2026-11-01-lifecycle-checkpoint-state/summary.json)。

## C11e：原生 Window/目录与 owner 转移

私有逻辑数据现可装配为项目实际使用的 `KvmemWindowState` 与 `KvmemPrefixCheckpoint`。
原有历史状态定义移到独立数据头文件，供 Program 和独立 CPU 检查共同使用。目录保留原
slot、空槽、purpose/protection 以及主目录/coverage 对同一 CP 的别名；不同 CP 也可共享一份
状态镜像。coverage 恢复严格验证依赖、数量与字节额度，按相对顺序重置 recency 时钟，避免
旧时钟接近上限导致下一次 touch 回绕；策略仅比较这些时钟的先后关系。

装配完成的对象仍是私有数据：SourceRef、session/publication key 与 MemoryStamp 保持无效，
旧 GPU working set 和执行中的 query 状态清空。Host 退休标记、原始逻辑页/位置及合法 CP
依赖保留；没有重新创建旧 GC worker 或旧许可。新域发布必须在后续事务中重新准入和复核。

原生 wrapper、身份副本、图预算及目录临时空间现在与 typed metadata 一起，在大镜像/H
读取前预约 C。`MetadataCharge::split` 只转移已有额度，不先退款再申请；CP 的独立 charge
和 image owner 随 detached alias 保留，Window 则持有其 imported charge。Window 的移动
赋值先完成旧对象析构，再移动新对象，避免第一成员 charge 提前退款。图及 feedback 同时
接入正常的每历史 32 MiB/父 graph budget，实际 M 标记继续用现有 allocator。

原先整批镜像检查会把已计入 C 的匹配 idle 镜像再次当作新增需求，导致明明可复用也拒绝。
现按实际可复用数量检查新增需求，之后仍用原 acquire；准确满额的两份 idle 镜像复用已覆盖，
没有取消真实计费或把 live 镜像当空闲使用。一般的非匹配 idle 镜像仍占 C，不承诺自动腾空。

Windows GNU16.2.0/C++20 三项检查通过：输入/逻辑组件、原生 Window 装配、coverage policy。
原生装配直接编译实际数据类型、prefix identity 和装配函数；测试专用物理 buffer 是普通 CPU
内存替身，不是 CUDA pinned 内存。检查 legacy/coverage、共享镜像、原始位置/字节、无效
身份、graph 拒绝、精确 C/M 与少一字节拒绝、20 个实际装配分配失败点、poison 后禁止重试，
以及持有 CP alias、移动 Window 时的真实释放顺序。此前 6,568 个截断、12/79/7 个输入/
typed/M 上游失败点仍通过。小 fixture 的 C peak 为 101,573 bytes、M peak 为 351 bytes，
只代表该平台/数据/预算口径，不是模型、Linux 构型、RSS 或性能证据。

首次本机链接因私有对象引用 Linux 专属归档析构接口失败；许可成员现按既有 Linux 支持范围
限定，Windows 不构造归档许可，修正后复验通过。为编译实际原生数据头，测试目录使用官方
NVTX v3.4.0 固定头文件，未修改主机设置。源类型布局已改变，后续必须完整重编译当前 Native，
不能混用旧二进制。完整 Linux 文件/Engine adapter、sanitizer、pinned pool/GPU 与模型仍待
连接恢复验收；没有公开恢复入口或 S/native 发布。

成本包括多份 CP 图的独立还原、额外目录/图预算租约与保守 C 预约；临时余量随 Window owner
保留到销毁，恢复全部目录可能因额度不足拒绝。同步 I/O、构造耗时和完整进程内存未测。
见 [C11e 结果及限定](experiments/2026-11-01-lifecycle-checkpoint-window/summary.json)。

## C11f：新源/原生身份的发布候选

新增 `Engine::import_historical_checkpoint(key)` 候选：要求空闲 Linux 单 owner、归档 key 尚未
绑定 RAM 源、存在空原生历史槽。先准备全部私有状态，整理到最新完整 CP 的计算边界，再
导入新 S 根并发布新 native domain。该入口导入 CPU 历史；没有 GPU 上传，后续 generate
仍走正常资源准入及精确前缀匹配，不承诺强制命中或完整请求恢复。

整理同时回退 Raw mean/selector、图/feedback、尾页版本与 thinking/features；文件的顶层
statistics 不能被当成一个未保存的完整状态。还会检查所有可召回页的实际 H/版本，避免只有
CP resident 依赖合法、下次 Query 却选到缺页。额外图复制/临时空间在大载荷读取前纳入 C。

普通 SourceImage 导入继续 computed=0。只有 Core 的新 checkpoint 授权入口，才可把后端
证明的 CP frontier 写入新根（revision=1、parent=0）。源和 CP 回执必须属于同一当前 blob；
S 准备后再次复核源/CP 及绑定条件，失败删除新根，成功才设置当前归档 binding。原生最后
一步只写入身份/有效位并移动已构造对象，无新增分配；空槽不够时拒绝，不先淘汰无关历史。
这里的新身份属于当前 owner，文件中的旧数值不是权限，跨进程数值重复也不能替代回执检查。

Windows 四项检查通过：输入/逻辑、原生 Window/发布协议、SourceJournal、SourceImage。
新增 6 个实际发布准备分配失败点、真实 S 额度拒绝、无效 H/version 拒绝和 S 返回后禁止新
分配的提交检查；此前各组件失败点保留。测试使用实际 Native 对象、实际 S 数据操作及 CPU
物理缓冲替身，**没有执行 Linux 归档授权入口**，不能当作最终权限验收。当前小 fixture 的
C peak 为 104,111 bytes、M peak 为 351 bytes，不是模型或全进程内存上界。

Linux 测试已加入同 blob 双回执、已计算边界、分配失败、S 分配期间 retire/forget/抢占绑定
后的最终回滚。真实模型测试新增独立 `save`/`restore` 进程，以原始 native continuation 作
输出基线，并要求实际 history hit、同 owner 转冷/再导入及附件退休拒绝。模型测试仅完成
Windows 语法检查，Linux/链接/GPU 尚未运行。原生 import/规范前缀重放属于不同路径，不能
用旧 SourceReplayAsNew 结果代替这一批。

成本是恢复时的私有整理/图反馈副本及保守 C 余量；全部合法目录和 S 数据仍须有额度，
I/O、性能、RSS 未测。接口/源码现为候选，完整 Native build、文件最终授权与实际模型循环
均待 SSH 恢复验证；请求级显式续算、取消/资源故障及长期质量仍未完成。
见 [C11f 实现、证据与未验证项](experiments/2026-11-01-lifecycle-checkpoint-publish/summary.json)。

## C11g：显式续算与源分支候选

新增 `Engine::prepare_checkpoint_resume(source, prefix_tokens, output_reserve_tokens)`，先准备，
再走既有 submit/generate、模型资源预约及原生恢复路径。它既能引用已有的完整 retained CP，
也能引用 C11f 导入的新 CPU 域。首批仍为 Linux、单 GPU/运行请求、文本/RAM、普通或 MTP、
关闭 legacy context cache；只重用 raw-token 配置的规范源前缀，不能把 chat 配置默认为 raw。
需要至少一个已接受、位于所选 CP 之后的规范 token，用正常前向取得 logits。

- 准备对象保存真实 R 前缀副本，其固定对象/控制开销另计 C；不持有历史 Window、CP image 或
  H 页。缓存失效后旧准备对象仍可能占 R/C，但不能阻止淘汰或获得新的恢复许可。
- 固定当前 SourceRef/revision、配置、Window stamp 和所选 CP stamp。准入核对规范 token、
  输出及 MTP workspace 预留，不接受关闭 reuse 或 branched read。历史查找只能选择该源/域/CP；
  准入或 materialization 找不到时拒绝，禁止转为普通的隐藏重算。
- 沿用原历史的 Query 来源及 CP 恢复后的 attention parser 状态。普通 raw 输入仍走自己的
  可选缓存路径；此次显式续算不能冒充 SourceReplayAsNew。
- 实际 native restore 后才创建 S 子分支，记录 parent/prefix 和真实计算边界；分支创建/
  计算进度提交失败撤销新子分支。成功接受输出后结束请求持有的恢复准备，退款取决于所有
  真实共享 owner。取消/失败未完成整段提交时不保留该请求的半成品 native history。

Windows GNU16.2.0/C++20 下四项 CPU 检查通过：输入/逻辑准备、实际 Window/发布/续算组件，
以及 SourceJournal/SourceImage 回归。续算新覆盖 3 个准备分配失败点、4 个 S 分支分配失败点，
真实 C/R 额度拒绝及退款、规范 token/输出预留/owner/revision/stamp/家族遗忘 fence、父分支与
计算进度。物理 image 使用普通 CPU buffer 代替 pinned buffer，未执行 CUDA。完整前端与真实
模型测试源的 Windows 语法检查通过；`-Wall -Wextra` 报告三处未改动的前端初始化警告。
Program/GPU 文件因本机缺少 `cuda_runtime.h` 未完成语法/链接检查。

真实模型 save/restore 流程改用显式接口：原始 retained CP 续算作为保存/恢复对照，要求实际
history hit、冷却后的旧准备拒绝、子分支关系和 R 退出退款；新子分支先持久保存，再做家族转冷。
该测试及取消、准入/执行资源不足的完整 Engine 场景仍未运行。SSH 未恢复，未重试/替换旧任务。
这些是候选代码，不能据 CPU 检查开启 `state_checkpoint` 能力或宣布跨进程模型恢复成功。

代价包括一个额外 R 前缀副本、C 固定准备对象、逐次规范行比对、S 子分支的复制/容量和恢复时
正常 CPU/GPU 工作；prepared-input 工作区另计，不承诺全进程 RSS 或性能。真实恢复中发生
资源失败可能已消费原缓存，当前不承诺撤销此前的缓存消费；完整故障/取消门槛尚待验证。
见 [C11g 证据与缺口](experiments/2026-11-01-lifecycle-checkpoint-request/summary.json)。

## 下一步的完整原生链路

1. SSH 恢复后先取回已启动任务的最终状态和日志，再编译最新保存候选。真实模型检查完整
   文件、退休标记、全部合法目录引用、C/H/D 边界与迟到工单拒绝，不复用旧模型二进制。
2. 验收 C11f 的最终授权与 S/native 原子发布/回滚，以及 C11g 的实际请求准入、资源预约、
   GPU 上传、取消及资源失败，核对最终许可和源分支，不以 CPU 组件代替 Engine 证明。
3. 执行普通/MTP4 与 LRU/redundancy 的真实保存—回收—跨进程恢复及原始 native continuation
   对照，旧版本、遗忘和退休不得绕过新身份许可。完成前不打开恢复能力标记。
4. 再完成自动源保留策略及任务价值、分层/多会话、长期资源/性能/Agent 验收。当前组件不能
   证明任意旧稀疏轨迹精确回放、全进程 RSS 上界或完整生产 GC。

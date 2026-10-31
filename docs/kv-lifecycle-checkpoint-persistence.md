# Checkpoint 持久恢复：权限组件与原生保存候选

更新：2026-11-01。C11a 的**归档权限组件**已验收；C11b 已实现原生编码及 Engine 保存候选，
本地 CPU 检查通过，最新原生完整编译和真实模型检查因 SSH 再次中断尚未完成。
后续 C11c 已补私有 CPU 读取准备组件，Engine 原生恢复入口及跨重启模型恢复仍未实现，
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

## 下一步的完整原生链路

1. SSH 恢复后先取回已启动任务的最终状态和日志，再编译最新保存候选。真实模型检查完整
   文件、退休标记、全部合法目录引用、C/H/D 边界与迟到工单拒绝，不复用旧模型二进制。
2. 在 C11c 字节/元数据读取准备上，完成 private typed state、M/C 分配与正常执行资源预约、
   GPU 上传及最终许可复核；成功才发布新 S/native 身份，失败不暴露半恢复状态。
3. 接通 Engine 新域恢复入口；验证普通/MTP4 与 LRU/redundancy 的真实保存—回收—跨进程
   恢复及独立输出基线，旧版本、遗忘和退休不得绕过新身份许可。
4. 再完成自动源保留策略及任务价值、分层/多会话、长期资源/性能/Agent 验收。当前组件不能
   证明任意旧稀疏轨迹精确回放、全进程 RSS 上界或完整生产 GC。

# Checkpoint 持久恢复：源快照权限与发布组件（C11a）

日期：2026-10-31。当前完成的是持久恢复所需的**归档权限组件**，尚未完成原生状态编码、
Engine 恢复入口或跨重启模型恢复。阶段 C 与完整六阶段目标继续开放。
执行范围、结果和不利发现见 [本批摘要](experiments/2026-10-31-lifecycle-checkpoint-authority/summary.json)。

## 为什么不能直接复用普通 session 保存

NInfer 已有 `session_snapshot.cpp`，保存 catalogued continuation 的 KV、StateImage、ledger 和
checkpoint 目录；它没有保存 `KvmemWindowState` 的历史退休页、稀疏页映射、Host 版本、检索/
反馈状态和源归档权限。现有 `checkpoint_image.hpp` 则只是进程内 SpillExtent 编解码，整数
identity 不是跨进程恢复许可。二者都不能直接作为 GC 历史的持久恢复接口。

本批先把恢复权限落实到既有 `SourceArchive`，继续复用目录锁、source namespace/key、已提交
源 blob 序号、固定对象表、共享 M/D 计费及 `StorageIoQueue`。没有另建 checkpoint 数据库、
存储池或后台线程。后续的模型状态编码仍归 NInfer Program 负责。

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
读取目标也由调用方先取得预算并保持私有。当前只验证小型组件载荷，尚未验证真实数百 MB/
GB 原生状态的预算、复制与恢复成本。

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

## 下一步的完整原生链路

1. 在 Program 内编码完整 checkpoint 依赖：main/draft/recurrent StateImage、原始位置与稀疏映射、
   仍合法的 Host KV/scale、统计和检索/图/反馈状态、退休标记及相关有限 checkpoint 目录。
2. schema 绑定实际模型、执行配置、载荷布局及状态语义版本。准备、解码、C/H/M 分配、上传都
   先保持私有；完成并复核当前许可后才发布新 S/native 身份，失败不能暴露半恢复状态。
3. 接入 Engine 保存和恢复入口；验证普通/MTP4 与 LRU/redundancy 的真实保存—回收—跨进程
   恢复及独立输出基线，旧版本、遗忘和退休不得绕过新身份许可。
4. 再完成自动源保留策略及任务价值、分层/多会话、长期资源/性能/Agent 验收。当前组件不能
   证明任意旧稀疏轨迹精确回放、全进程 RSS 上界或完整生产 GC。

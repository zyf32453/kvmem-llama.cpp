# KV 生命周期六阶段进度

更新时间：2026-10-31。完整目标仍为 [总体设计](kv-lifecycle-system-design.md) 中的六个阶段；单项通过不代替完整生产验收。

- **A 基线与契约：已完成。** NInfer 主后端、模型/源码固定身份、首批范围、动作及资源 owner、可执行契约已建立。证据：[阶段 A](kv-lifecycle-baseline.md)，提交 `ca2bbfe`。
- **B 可靠机制与预算：首批支持档位已完成。** B1 原子交接、B2 固定提交条件、B3 精确直接代表/评分证据和 B4 GC 自有辅助预算、旧 owner 计费、worker 健康均已实现。真实线程与交叉退休、禁止新分配的前台提交、分配故障注入、版本/保护撤销及 Linux 核心 ASan/UBSan 有独立证据。TSan 因工具在进入应用前失败仍无结论；不因此宣称竞态检测通过。证据：[B1](kv-lifecycle-boundary-mailbox.md)、[B2](kv-lifecycle-commit-conditions.md)、[B3](kv-lifecycle-score-evidence.md)、[特征预算](kv-lifecycle-auxiliary-budget.md)、[owner 预算](kv-lifecycle-owner-budget.md)、[B4 原生接线](kv-lifecycle-native-budget.md)。

  服务验收发现并修复了生成中辅助额度不足会使整个 Engine 故障的真实缺陷。第 26 补丁区分有效 sequence 的有序 abort 和已消费 membership 的结清，保留等待 FIFO。当前版本 20 项主机契约、5 项原生测试、两环境各 6 项协议测试通过；真实模型 8 项活动/排队停止、2 项生成中额度不足、12 次 M=10 MiB 独立会话轮换全部通过。之前的四组强压力回收、非流式拒绝/后续恢复、断连与中间二进制 48 次轮换按各自版本保留，不混记。见 [服务故障及完整结果](kv-lifecycle-service-failures.md)。

  这里完成的是冻结的单 GPU、单运行请求、文本/RAM 档位的机制和有界 GC 自有资源；M/H 不是整个进程或 checkpoint 的上界。新测试实际观察到冷域/checkpoint 带来的 RSS 增长，完整 checkpoint/源记录预算继续由 C/E 处理。长时间、固定工作量性能、先前偶发 CUDA 和 Agent 质量属于后续门槛，完整生产系统尚未完成。
- **C 恢复与版本闭环：实现中，未完成。** C1 独立 RAM SourceJournalStore、真实配额、分配失败原子性、revision/分支/局部遗忘 fence 已完成组件验证，见 [C1](kv-lifecycle-source-journal.md)。C2 已接入 NInfer：实际 frontend 模板/配置及消息来源、接纳输入与最终提交输出、独立 `--kvmem-source-mib` 和用量，以及取消/失败清理。最终二进制通过 21 项主机契约、7 项原生检查、4 组真实内容记录、2 组源额度耗尽/FIFO 和 4 组取消边界检查，见 [C2 完整说明](kv-lifecycle-native-source.md)。源记录默认关闭、当前按请求保留（含预热/失败输入）；当时没有公开遗忘，后续 C4 已补进程内家族接口，仍无自动过期/去重。

  C3 的规范前缀新域重建已接入 NInfer：独立 R 真实配额、正常 Program 模型资源预约、新鲜域身份、逐单元源版本校验、取消不保留半成品、请求级源回执和父分支关系。最终候选通过 9 项原生检查、Linux 三项 ASan/UBSan/泄漏检查；22 项主机契约在确认核心源码未变后复用其覆盖。LRU／redundancy × 普通解码／MTP4 四组真实模型全部通过：重建前缀内 CPU 页确实释放后，规范重建与独立原始 IDs 输出一致；旧 native window 被换出后仍可仅从源重建。另覆盖实际部分取消、准备额度耗尽、准入 reserve 拒绝和首 token 停止。详见 [C3 组件](kv-lifecycle-source-recovery.md) 和 [原生重建及不利结果](kv-lifecycle-native-recovery.md)。

  这次只支持完整前缀可装入窗口的显式 raw-token 新域重建，不是旧稀疏/recurrent 轨迹的精确恢复，也没有 HTTP 或持久化接口。R 副本之外的 prepared-input 与源分支另占请求工作区/S；仍无全进程内存上界或性能结论。超长源稀疏重放、checkpoint/归档 fence、完整发布/回滚故障、持久化及系统级遗忘仍待完成，阶段 C 不关闭。

  C4 已接通显式进程内源家族遗忘：空闲 Engine 下统一撤销原生 KV/checkpoint/coverage 别名，再删除源与子孙；旧准备对象在准入失效，R 随真实 owner 退出退款，忙碌或 stale 不部分修改，无关源保留。最新候选 9 项原生检查、四组真实模型均通过；22 项主机契约及 3 项核心 sanitizer 结果在确认源码未变后复用。本批同时暴露了测试输入超过 H 会提前逐出目标域，修正前置负载且保留失败。更重要的是 checkpoint 在遗忘后仍约 0.92 GB，H 不覆盖它；下一步优先收紧 checkpoint 总预算。详见 [C4 完整说明及代价](kv-lifecycle-native-forget.md)。这个接口不覆盖持久化、其他 Engine 或调用方 transcript，也不承诺安全擦除。

  **C5 checkpoint owner 总预算已完成首批功能验收。** 原生第 30 补丁、108 项合同输入及 3,440 项实际源输入固定；独立 C 按真实 image、metadata、私有 rollback、空闲池和 alias 生命周期计费。12 项原生检查及定向 ASan/UBSan/泄漏检查通过，22 项已有主机契约在确认 89 项输入相同后复用。七组 GPU 检查全部通过：真实 pinned pool；LRU／redundancy × 普通／MTP4 的 1 MiB 请求级拒绝和 384 MiB 反复回收、源重建；C3/C4 各一组 2,048 MiB MTP4 回归。压力组各回收八个旧域，C 峰值约 308.1 MB，源可恢复；原物理释放与 3／3 家族撤销断言保留。SSH 恢复后确认原任务 I/O 失败，补齐恢复包缺少的 probe，并在持久目录完成约 90 分钟冷构建；等待时限调整及原有 CUDA 警告均保留。详见 [C5 完整结果与资源边界](kv-lifecycle-checkpoint-budget.md)。C 不等于 RSS；较小 C 会改变 cache 保留数量，性能和完整生产仍未验收。

  源镜像组件：有界不可变 R 副本、流式导出/解码、配置及损坏拒绝、失败退款，已通过 23 项 Windows 主机契约、一项 Linux ASan/UBSan/泄漏检查及 NInfer/C++20 构建接入测试。合入后八个原生程序逐字节不变，C5 运行结果按相同二进制复用；该次合同冻结 111 项输入，当时没有持久清单/遗忘 fence。后续 C6 在此数据组件上落实独立归档权威。源镜像实现与独立证据见 [源记录镜像](kv-lifecycle-source-image.md)。

  **C6 持久源归档组件已实现并通过 Linux 功能及内存检查。** 稳定 namespace/单调 ID、高水位、catalog/seal、真实 R 快照、现有 StorageIoQueue、owner 发布复核、只追加旧版快照与反序完成拒绝、重开后的读取资格和归档家族遗忘均有独立组件测试。固定表可接共享 M，writer/reader/receipt 保留真实 quota 与目录锁，最后退出再清理；退款额度与实际删除文件字节单独统计。五项组件 Debug 和五项 ASan/UBSan/泄漏检查通过；83 个清单故障点、30 个后台 blob 故障点、六个进程退出边界及 7/4 个分配失败点覆盖关键异常。详见 [C6 契约、结果与代价](kv-lifecycle-source-archive.md)。

  C6 当时仍没有 Engine 保存/导入或跨重启模型恢复，也未把归档 forget 冒充系统级遗忘。清单与 seal 提交中断可能使整个归档拒绝打开；整体目录备份回滚没有外部单调锚点，不能识别。这些可用性/信任边界、同步元数据 I/O 成本与尚未验证项必须保留，C 不因此关闭。

  C6 合同冻结 125 项输入；五项 Debug 与五项内存检查各约 12 秒，NInfer/C++20 Release 下同一归档测试也通过。当时八个原生程序逐字节保持 C5 身份，GPU 结果按原二进制复用。源码检查发现并修正清理单一返回值混合额度退款与文件删除量的问题，接口和测试分别验证。后续 C7 因公共代码变化重新做当前二进制的原生/GPU 验收，见下。

  **C7 原子源导入与跨进程续存组件已通过本批验收。** SourceImage 直接在 S 中准备两份真实 buffer，成功才发布当前新根，父引用不继承、computed=0；当前归档 read receipt 才能建立 durable key 的 RAM binding。以持久 blob 序号排列保存，不跨进程比较 RAM revision。四个独立 exec 实际覆盖保存、数值身份重复/revision 重置后的导入续存、再次导入遗忘和重开拒绝；普通/授权导入各四个真实分配失败点。

  Windows 四项回归、Linux 五项 Debug/五项 ASan/UBSan/泄漏、12 项原生契约/两项组件均通过。七个原生程序发生变化，七组 GPU 回归已重新执行并全过，原物理释放、额度不足后可用与三源/三域撤销断言保留；没有沿用旧二进制作为当前证明。首个原生任务在检查完成后因验收脚本更新了自身冻结元数据而失败，修正顺序后在新目录以相同代码复验；失败日志和构建标识警告保留。见 [C7 完整结果](kv-lifecycle-source-import.md)。该批仍无 Engine 归档入口、归档驱动的模型恢复或统一持久/在线遗忘，后续 C8 见下；C 继续开放。
  **C8 Engine 源归档已接通，首批模型功能检查通过。** 当前 Frontend/Program 配置验证、原有 I/O 队列上的保存工单、当前回执授权的 S 导入、新域推理、持久家族与所有在线 RAM 根/子孙的统一撤销均已实现。四种 LRU/redundancy × 普通/MTP4 配置的保存/恢复遗忘/重开拒绝共 12 个独立进程，以及配置、R/D 额度、磁盘提交中断、损坏重开五个进程全部通过。每组恢复都保留真实 CPU 前缀页释放和独立原始 IDs 输出断言，统一遗忘实际清掉 4 源/2 域/2 归档，无关家族保留。

  Windows 4 项、Linux 5 项 Debug 与同 5 项内存检查、原生 12 项契约及 2 项组件检查通过；八个既有二进制均已变化，七组旧功能 GPU 回归已重新执行并全过，完整结果按 [C8 完整说明](kv-lifecycle-native-archive.md) 和执行摘要记录。测试准备曾用空提示作配置参考，且复用缓存带入旧结果目录；修正输入/结果隔离并保留失败和主动停止记录，未放宽负载或断言。固定配置、同步清单 I/O、一次 binding 及磁盘部分提交会拒绝整个归档的代价保留。此批只从 32K 源恢复可装入窗口的 1K 规范前缀，超长重放、checkpoint 持久恢复等尚未完成，C 不关闭。
  **C9 完整长前缀的新域分块重放已通过首批验收。** 保留原有严格完整窗口重建，新增显式 adapter 能力及 ReplayAsNew。物理 selected+append 窗口、单元上限与逻辑 completion 分别校验；新域走正常资源预约、逐单元源版本复核和请求完成协议。四组 LRU/redundancy × 普通/MTP4 在 2,560-token 设备窗口重放完整 32K 源前缀，实际普通 64 单元、MTP4 65 单元、每步不超过 512、区间无缺口，输出与独立原始输入一致；跨窗口取消、R/完成额度拒绝、完整准备遗忘失效及重启拒绝均通过。

  Windows 23 项契约、Linux 24 项 Debug 与同 24 项 ASan/UBSan/泄漏检查、原生 12 项契约及 2 项组件检查通过。九个既有程序均变化，当前二进制重新执行新重放/兼容 14 个进程、归档 17 个进程和既有 GPU 七组，全部通过；真实 C8 文件用于当前读取/长准备兼容，不冒充旧文件的独立长推理。详见 [C9 完整说明](kv-lifecycle-source-replay.md)。

  分块不等于单块 CPU 空间：R 仍持有完整 131,072 bytes 前缀，prepared-input 工作区独立；计划增加 16 bytes 固定元数据。第二次长请求准入并取消后，前一个 inactive native 域已不在缓存中，不能承诺取消撤回先前缓存淘汰。新域重算不等于旧轨迹或 dense 等价，真实复杂内容/非对齐长尾及长期性能仍待扩展。保留归档但腾出 S 再同 owner 导回尚无接口，checkpoint 持久恢复和精确兼容旧轨迹闭环也未完成，C 继续开放。
- **D 任务感知保留：未完成。** 仍需结构化任务事件、有限锚点、衰减价值策略及质量比较；已有角色/thinking/冗余能力保留为基础。
- **E 分层与多会话：未完成。** 已有 HostKvStorage/SpillFile 等组件不等于 GC+SSD/并发组合已验收。
- **F 完整生产及研究验收：未完成。** 仍需新版本 GPU/Agent、长时间多次回收/恢复、资源边界及性能验收。

下一步优先补齐保留归档的源家族冷却、显式解除 binding 和同 owner 导回，再处理 checkpoint 持久恢复与兼容旧轨迹闭环；完整长前缀新域重放按 C9 固定负载验收，其他配置和真实任务继续由后续门槛验证。只有验证旧快照不能复活退休身份并完成这些条件，才关闭 C。短功能检查不代替 F 的配对性能、长时间与 Agent 质量验收。

用户“暂不修改主机调试设置”和“DeepSWE 暂缓”的约束继续有效。SSH 已恢复，C5–C9 按各自范围记录；Engine 的规范前缀跨重启恢复已提供模型证据，完整系统尚未完成。

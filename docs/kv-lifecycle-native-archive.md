# C8：Engine 源归档与统一家族遗忘

后续 [C9 长前缀新域重放](kv-lifecycle-source-replay.md) 已增加独立 ReplayAsNew 能力和真实模型检查；原严格短前缀入口及本页原批次结果保持原范围，不倒写成精确旧轨迹或完整生产验收。

日期：2026-10-31。接续 [C6 归档组件](kv-lifecycle-source-archive.md) 和
[C7 原子导入](kv-lifecycle-source-import.md)。本批目标是把已有机制接入真实 Engine，
完成保存、重开、规范前缀新域推理以及持久/在线家族撤销。新增功能的 17 个独立进程模型检查和
七组现有 GPU 回归全部通过；这不代表阶段 C 或完整生产系统完成。
最新总体状态见 [六阶段进度](kv-lifecycle-progress.md)。

## 修改前后

C7 提供组件层 SourceArchive 和原子 S 导入，但 Engine 不能直接附加、保存或导入归档。
归档 forget 也不会自动撤销已导入的在线源。现在由 Engine 控制顺序、Frontend 验证输入配置、
Program 持有一个可选归档，并复用原有 StorageIoQueue、R 和共享 M；没有新增线程池或平行源目录。

公开接口位于原生 `include/ninfer/engine.h` 和 `historical_source.h`，随第 31 个补丁
`backends/patches/ninfer-source-archive.patch` 固定；本批契约是
[kv-lifecycle-native-archive-candidate-v1.json](contracts/kv-lifecycle-native-archive-candidate-v1.json)。

- `attach_historical_archive(options, reference)`：明确创建或打开一个固定配置 namespace。
- `begin_historical_save(source)`：准备不可变 R 镜像，预约 D 并排入原有队列，返回可移动工单。
- 工单的 `wait/status/cancel/key`：只操作文件 I/O，不取得恢复许可，也不借用 Engine 指针。
- `finish_historical_save(ticket)`：当前空闲 owner 复核源和家族，成功后发布持久版本。
- `historical_archives/historical_archive_usage`：查询当前目录、binding 和独立归档资源。
- `import_historical_source(key)`：用当前 read receipt 导入新 RAM 根，再交给已有
  `prepare_source_rebuild` 和正常生成请求计算新域。
- `forget_historical_source` 与 `forget_historical_archive`：统一持久家族和在线别名撤销；
  `collect_historical_archives` 只清理不再授权且没有 owner 持有的文件。

## 配置与并发边界

首批限定 Linux 本地文件系统、Qwen3.5 主后端、单 GPU、max_concurrency=1、纯文本和 RAM KV，
要求 S/R 已启用，并明确指定目录、D、归档元数据预算。其他模型族和操作系统没有获得支持声明。
归档元数据接入 Program 现有 M，目录和 object 数量均有固定上限。

附加参考必须是当前兼容 Frontend 实际生成的 PreparedPrompt。内部检查 artifact、基础 tokenizer/
模板、raw/chat 及实际模板参数，并结合 Program 的 KV、MTP、RoPE 和相关执行配置重算 digest。
不从磁盘自报的配置推导许可，不把 raw 与 chat 混用。为此 prepared 元数据增加一个 32 字节
Frontend stamp，占请求工作区，不属于 S/R 的 buffer 计费。

变更归档的控制操作要求没有活动、排队、正在准备的请求或未消费 handle，且 Program 没有未结清
事务/传输。begin 完成后可提交推理，文件 worker 只访问有界不可变镜像；finish 必须回到空闲 owner。
wait 不会持有 Engine 执行锁等待磁盘。清单 fsync 仍是显式同步控制操作，没有放到 decode 边界。
这说明允许共存的所有权关系，不构成已经测得 I/O/推理重叠收益的性能结论。

## 为什么不能只遍历 RAM parent

导入后的父归档和子归档各自成为 RAM 根，旧进程的 parent/ref 不能安装回来。因此遗忘先确定持久
家族，再查找其中每个 durable key 的当前 RAM binding，展开其在线子孙，选择互不重叠的 RAM 根。
同时预检查 continuation、native history、checkpoint/coverage 和 source_last，预分配替换对象。

所有可失败的在线准备完成后，先持久撤销，再执行不分配的在线 owner 清理。旧准备对象在准入
失效；迟到的保存完成也不能重新发布。R 镜像、工单及磁盘对象仍按实际 owner 退出退款，遗忘
不能把仍被持有的空间提前记成空闲。无关家族保留。

若 catalog 已替换而 seal 提交失败，调用方不能知道永久遗忘是否完整提交。此时归档标记不健康，
已经预检查的在线家族仍被撤销，原错误继续向上传递；不报告“永久遗忘成功”。不完整的权威对
重开时拒绝整个 namespace，不能偷偷加载旧清单。本批不提供修复、降级选旧副本或跨 Engine 撤销。

## 资源与拒绝行为

源镜像和读取使用真实 R，完整导入使用 S，归档文件使用 D，固定表受元数据额度并可计入共享 M。
begin 检查单个镜像能否装入现有 I/O 队列总字节容量；结构性超大返回 BudgetExceeded，避免永远
重试 Pending。暂时队列满依然返回 Pending。入队失败后未发布的持久 ID 和 D 占用仍由 collect
按实际 owner 状态清理，不复用旧 ID。

导入不恢复旧 computed/KV/checkpoint。一次导入成功即是完整有效的 S 源；之后模型准入或恢复准备
失败不意味着该有效源也必须消失。这个分步 API 不构造半个 native domain，调用方可以明确重试。

## 验收入口与未完成项

`ninfer_engine_source_archive_real_test` 在多个独立可执行进程中测试保存、恢复、遗忘和重开拒绝；
比较新域输出与独立原始 token IDs 推理，检查恢复前确有 CPU 前缀页释放，并保留 MTP 真正执行的
断言。还包含配置、R/D 不足和 catalog/seal 中断场景。

本批在 RTX 3080 Ti Laptop 16 GiB、固定 Qwen3.8-27B `.ninfer` 模型上实际执行：

模型 SHA-256 为 `29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`。
新归档检查使用 B=2048、生成 reserve=512、H=96 MiB、C=2048 MiB、S=16 MiB、R=1 MiB、
M=64 MiB、归档 D=64 MiB、归档元数据上限=1 MiB，raw 输入 32,768 tokens，恢复前缀 1,024，
生成 8 tokens。R/D 拒绝组分别改用 64 bytes 和 64 KiB；这不是默认配置或任务质量的验收。

- LRU/redundancy × 普通解码/MTP4 四组，每组由保存、恢复与遗忘、再次重开拒绝三个独立进程组成，
  共 12 个进程全部通过。保存前确有恢复前缀内 CPU 页物理释放；新域输出与独立原始 IDs 推理一致，
  MTP 分组确实执行 speculative rounds。
- 恢复进程导入的三个源均为 computed=0、无旧 parent/native domain 的新 RAM 根。统一遗忘实际
  清理 4 个 RAM 源、2 个 native domain、2 个归档条目；独立家族保留。两个失效准备对象仍真实占用
  8,192 bytes R，待 owner 退出才退款；迟到保存不能发布，最后工单退出后 D 才释放。
- 配置不匹配、R 不足、D 不足、catalog 已替换/seal 失败、后续重开拒绝五个额外独立进程全部通过。
  额度拒绝和磁盘失败后的普通推理仍可使用；磁盘失败不被记作永久遗忘成功。
- Windows GCC 16.2 Debug 四项相关回归通过；Linux 五项 Debug 和同五项 ASan/UBSan/泄漏检查通过；
  原生 C++20 的 12 项接口/预算检查与 2 项组件检查通过。
- 八个既有原生二进制均相对 C7 发生变化，因此重新执行七组 GPU 回归：真实 pinned pool、四组
  checkpoint 额度/反复回收、C3 恢复和 C4 遗忘均通过。原来的物理前缀释放、额度拒绝后继续推理、
  三源/三域家族撤销等强断言保留。没有借旧二进制结果代表当前候选，也没有性能或 Agent 结论。

编译与执行输入固定为 142 项合同源码、3,449 项实际输入及原生第 31 补丁；补丁也从已验收 C7
源码树独立重放得到相同结果。首个原生构建完成 146 步；测试修正后的增量只执行 build stamp、
测试编译和链接三步。导出的原生源码仍有 build-id unknown 提示，不能用该字符串代替精确身份。

第一组模型检查曾在保存前退出：测试使用空 PreparedPrompt 作为配置参考，被 Engine 正确拒绝。
改用真实 token 参考后复验，原工作量、模型、输出及物理释放/遗忘断言均保留。复用构建缓存还带入
旧结果目录，发现后在运行新模型前停止本次控制器，保留 exit137/OOM=false 和旧证据，改用独立
结果目录。另有源码预检误用默认 profile、独立索引差异检查误用无 HEAD 基准的脚本问题，均修正为
实际冻结配置/C7 基准；它们没有被记成通过的应用测试。失败记录及最终结果按本批执行摘要保留。

完整执行摘要为 [summary.json](experiments/2026-10-31-lifecycle-native-archive/summary.json)。
原始日志、脚本、JUnit、实际输入和失败轮次在本地 `artifacts/lifecycle-native-archive/evidence.zip`，
准确 SHA-256/大小由摘要记录。归档目录由 Docker owner 建立，收集器最初直接从 SSH 用户读取受限
目录失败，最终改用相同容器身份读取；没有改变主机或归档权限，没有因此重跑或改写模型结果。

仍未完成：checkpoint 持久恢复、超长源稀疏重放、自动冷热源 detach/rebind、跨 Engine/多会话、
HTTP/CLI 入口、长期/固定工作量性能及真实 Agent 质量验收。一致目录备份回滚没有外部单调锚点，
仍不能识别；本地 FNV 也不是密码学完整性保护。S/R/M/H/C/D 不构成全进程 RSS 上界。

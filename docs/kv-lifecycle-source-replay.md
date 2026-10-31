# C9：长前缀的新域分块重放

后续 [C10 源家族转冷](kv-lifecycle-source-cool.md) 已补齐保留归档的显式解绑与同 owner 再导入，并以真实 C9 归档执行完整 32K 重放；同时修正了取消测试在回调中同步等待控制接口的问题。下文保留 C9 原批次结果和当时未完成项，不倒写历史证据。

日期：2026-10-31。接续 [C8 Engine 归档](kv-lifecycle-native-archive.md)。本批实现显式
`prepare_source_replay`，让已接受的完整长前缀在固定 KV 窗口内重新计算。本批已通过所述固定配置的功能与模型
验收；阶段 C 和六阶段总目标继续开放，不能把新增入口当作完整生产验收。

## 相对 C8 的变化

C8 可以从 32K 源归档导入全部 token/event，但 `prepare_source_rebuild` 只支持完整前缀及输出
reserve 能装入设备窗口的重建，真实模型只验了 1K 前缀。本批保留该严格入口，新增独立的长前缀
重放模式 `ReplayAsNew`，不让原来的容量拒绝静默变成稀疏计算。

新入口仍产生**新的当前策略计算轨迹**。它不恢复旧的 sparse/recurrent/spec 状态，不撤销旧 KV
退休，也不承诺与 dense/full-attention 结果等价。后续 checkpoint 持久化与旧轨迹恢复需要各自的
兼容及授权证据，不能从 token 相同推导。

## 两类预算分别检查

新增 `MemoryCapabilities::windowed_source_replay`，默认关闭。仅有 sparse KV transfer 不授予
该执行能力。首批 NInfer adapter 明确限制 Linux、单 GPU、单请求、纯文本/RAM、启用 S/R、
none/MTP；其他后端不会因为新增枚举就自动获得支持。

`SourceReplayBudget` 区分：

- `chunk_tokens`：一次 native prefill 单元的最大 token 数，须非零、页对齐，且不超过物理 append
  reserve。selected history 与按页向上取整的 append reserve 之和必须装入设备窗口。
- `completion_tokens`：整次逻辑输出承诺，包含 backend workspace。它与整个前缀之和不能超过
 逻辑 context，可跨多个物理单位；不会把总输出错误地当成必须一次驻留的设备空间。

旧 `RebuildAsNew` 仍检查整个前缀的页面数；`ExactState` 仍明确拒绝，不以 token 重放降级。
未知模式、缺失能力、未对齐/超大单元、物理窗口不足、context 或整数溢出在执行前拒绝。

## 复用现有执行所有权

Engine 将准备交给 Program，再交给原有 Frontend 和正常 submit/generate。Program 仍预约真实
main/draft/state/checkpoint/scratch/Host 资源，并使用现有 canonical chunk grid、Pressure/Query
选择和请求完成协议；没有新的模型执行循环、GPU allocator、线程池或历史目录。

准备对象的源 revision/configuration 在准入和每个实际执行单元重查；另核对当前 selected、append、
设备窗口和 chunk 与计划一致。旧域复用、branched reads 继续关闭。完整重放提交前不保留新 native
history；部分取消保留正常的 accepted/computed 源事实及回执，R 则按真实 owner 生命周期退款。

R 仍持有整个源前缀的不可变副本，随后 prepared-input 还有独立请求工作区。本批“分块”指计算和
设备驻留，不代表只占一个 chunk 的 CPU 空间。计划增加两个 uint64 标量，prepared/request 元数据
相应增长；这些固定对象不等于 R buffer，也没有全进程 RSS 上界或性能不变的承诺。

## 验收入口

原生 `ninfer_engine_source_replay_real_test` 通过独立进程保存 32,768-token 输入及其独立原始输入
输出；重启后导入源，在 2,560-token 物理窗口内重放完整前缀，比较输出，记录每个实际 prefill
单元，并检查完整区间覆盖、每步不超过 512 tokens、MTP 真正执行和旧 CPU 页确实释放。

同一测试覆盖：短前缀入口仍拒绝长源；逻辑 completion 不足及 R 压力拒绝；跨多个窗口后实际取消；
未完成 native history 不发布；家族遗忘使完整长 R 准备失效；再次重启拒绝已遗忘源。
另用真实 C8 namespace 验证新版的授权读取/长 R 准备，不把重新生成的文件当作旧版本兼容证据。

本批实际通过 Windows 23 项契约、Linux 24 项 Debug 与同 24 项 ASan/UBSan/泄漏检查、原生
12 项接口/预算检查及两项组件检查。新重放/兼容 14 个进程、C8 归档 17 个进程、既有七组 GPU
回归全部通过；九个既有可执行程序相对 C8 均变化，没有用旧二进制结果代替当前证明。

四组实际普通重放各 64 单元，MTP4 各 65 单元，每步不超过 512，覆盖完整 32K 前缀；完整成功
域曾被保留，第二次长请求准入并取消后，旧 inactive 域已不在缓存中。部分新域没有发布，但取消
不会撤回先前发生的缓存回收，也不保证避免后续 miss/重建代价。所有源事实直到显式家族遗忘才删除。

组合合同冻结 146 项输入，原生实际输入 3,452 项，第 32 补丁从 C8 源码树独立重放一致。首批
工具准备中 Windows 子进程找不到 CMake，改为既有工具的显式路径后通过；一次只读 SSH 元数据
查询引号错误已修正。最终档位元数据还曾混淆“公开可选策略”与“已验收策略”，源码预检拒绝后
修正为分别记录 off/LRU/redundancy 与已验收的 LRU/redundancy。未重启模型任务、改代码输入、
负载或放宽断言。导出原生源码的 unknown build-id 提示
仍保留，以精确源码/补丁树/二进制摘要确定身份，不宣称该提示已修复。

实际结果和所有资源代价见 [执行摘要](experiments/2026-10-31-lifecycle-source-replay/summary.json)。
真实 C8 文件只验证当前授权导入及完整 R 准备；该旧 fixture 未保存 32K 原始输出 oracle，
因此没有将它写成旧文件的独立长推理通过。四组新的完整长重放则均实际执行模型及独立输出比较。

原始日志、输入、脚本、JUnit 与三组 GPU 报告在本地 `artifacts/lifecycle-source-replay/evidence.zip`，
SHA-256 和大小由执行摘要记录。性能、全进程 RSS、复杂真实输入及非对齐长 accepted-output 尾部
尚未单独验收；不能从 generic 3,073-row 部分尾页复制或 32K 后生成，推导其全部 native 场景通过。

完整 checkpoint 持久恢复、精确旧轨迹重放、自动冷却/重新绑定、任务价值、多会话分层以及长期/
固定工作量性能与 Agent 质量仍未完成。当前 repeated-token 负载是机制检查，不是任务质量证据。

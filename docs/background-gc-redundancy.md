# CPU 历史 KV：redundancy 管理策略

本次在 `codex/background-gc-vllm` 的基本 GC 上增加可选策略，默认仍为 LRU。
策略决定哪些 CPU 历史块退休；原 manager 继续负责身份复核、关闭新访问、等待已有读写结束和释放。
已经准入的 GPU 工作集仍按原有规则运行。本次没有替换 GPU retrieval selector。

多轮游戏开发的补充案例、测试干扰与实际结果见
[浏览器 FPS 质量案例](background-gc-browser-fps.md)。该案例不能替代下述原配置的验收。

当前 IQ3 27B / 32-token 块的接入与最新验收见 [导师配置记录](background-gc-mentor-27b.md)
及 [参数复测记录](background-gc-parameter-retest.md)；后者也是本次提交的代码检查与数据入口。

## 用法与支持范围

CLI 和 server 在原有 historical 参数上增加：

```sh
--kvmem-historical-gc-policy redundancy
```

可选值为 `lru` 和 `redundancy`，必须搭配 `--kvmem-historical-capture N`。
`off/shadow/delete` 模式含义不变。选择 redundancy 后，即使 mode 为 off，也采集特征，
便于测量相同采集路径下的增量 GC 开销；想关闭新增特征开销应选择 lru。
策略在上下文创建时确定，不提供运行中切换。server 的 `/props` 发布 `historical.policy`。

沿用首批单 GPU、单 lane、串行文本、recency、Flash Attention、f16 KV、RAM 历史配置。
NVMe、MTP、多 lane、混合模型、DeepSWE 不在本次验收范围内。
更新已有 llama.cpp 工作目录时，需重新运行 `scripts/apply-patches.sh`；
新加入的 `0014-historical-redundancy-capture.patch` 打通 recency 下的特征采集，
并支持单 token 的输入 Query。Windows 构建脚本也包含此增量补丁。

## 选块方法

这是受 R-KV（arXiv:2505.24133v4）启发的 **Block 级近似**，不是论文 token 级算法的复现。

1. 每个已封口块保存 pre-RoPE Mean-K。为复用既有 RawStore 完整发布协议，
   Mean-K 仍在所有 attention 层采集、保存；评分只使用其中最多 4 个均匀分布的层。
   各层各 KV head 分别归一化后拼接，两个描述向量的余弦相似度等于这些 head 相似度的平均值。
2. Importance 使用最近一次完整输入 Query 样本：最后一个成功发布的 microbatch 中最多 8 个 token 的均值。
   每组 KV head 对相应 GQA Query heads 取最大点积，除以根号 head dimension，再跨层/head 平均；
   最后在本轮有效块间 softmax。**生成期间不逐 token 更新 Query**。
3. Redundancy 使用最多 32 个较新的有效块作为参照，排除自身，平均余弦相似度后在块间 softmax。
   不创建完整 N×N 矩阵；这与论文的全量 token 比较有区别。
4. 保留分数为 `0.1 * Importance - 0.9 * Redundancy`，低分优先退休。
   当旧块与较新参照的相似度至少为 0.9 时，本轮保护它选中的较新代表，避免一批删掉两个副本。
   该保护是本轮局部规则，不承诺在后续无限压力下永久保留某类信息。
5. 只提出 Live、已封口、非 mandatory 块的退休意图，sink/recent 保护沿用现有规则。
   无 Query、特征缺失、零向量、非有限值时保护相关块，不悄悄改用 LRU。
   因而保护过多或没有有效特征时，容量不足仍可能拒绝新请求。

这些参数目前固定在策略 Config 默认值中；没有增加一组尚未经验证的命令行调参入口。
高 key 相似度是冗余的代理信号，不等同于文本语义完全重复；Mean-K 会损失块内部细节。
当前各块等权，包括短尾和生成片段，没有再按块内 token 数校准分数。

## 线程、生命周期和开销

`HistoricalKvMeanKeySource` 只读特征，`HistoricalKvRedundancyPolicy` 只产生意图，
`HistoricalKvGc` 仍统一执行回收。策略不持有释放回调，也不直接修改 GPU 槽。
每条意图携带仓库代际、槽代际和 content version；manager 提交前再次校验身份与保护条件。

Query 按所有采样层的相同结束位置整体发布，后台评分冻结本轮 Query。
发布者使用 try-lock，竞争时跳过该次样本并保留上一份完整样本，不等待后台计算。
Mean-K 通过专用后台只读租约复制；租约仍阻止 CPU 数据释放，但不计入前台请求完成计数。
复制结束后立即释放租约。自有缓存按代际和 content version 校验，槽复用不能沿用旧特征。

后台沿用 5 ms 轮询，每个观察/特征切片最多 16 块、比较切片最多 512 对，
每批最多 16 条意图。切片限制工作量，不保证某个固定微秒上界；排序和元数据扫描也有成本。
仍使用既有后台调度，不新增“严格只能在两次 decode 之间运行”的同步屏障。

新增开销包含前台计算图保留特征、设备到 CPU 传输、Mean-K 存储和后台评分。
Mean-K 存储计入既有历史 payload 预算；描述缓存、Query 缓存与评分工作区是单独的固定容量内存，
不能把 payload 指标宣称为整进程内存上界。设评分维度 D、容量 C，
两份主要缓存约为 `2 * C * D * sizeof(float)`，另有 Query 和 O(C) 元数据。
在实测 48 层、4 KV heads、128 head dimension、容量 64 的模型上，D=2048，
这两份缓存约 1 MiB；Mean-K 全容量上界另为 6 MiB。

## 测试方法

- `historical_kv_redundancy_test`：受控重复/独有信息选择，与 LRU 对照；importance 权重作用；
  代表保护、mandatory、缺失/非法特征、压力取消、pending credit、参照数量上限；
  真 RawStore 的释放及槽复用；并发 Query 发布的一致性；后台读租约保护数据且不阻塞请求完成。
- 既有历史 KV 测试：回归原 LRU、GC、repository、capture session、active set 等生命周期规则。
- 真实模型 `--gc-redundancy-bench 5`：一个模型/上下文，每模式预热一次，随后五轮轮换
  off/shadow/delete；256 输入 token、32 decode、容量 64、低/高水位 32/48。
  GPU 槽覆盖全部历史，逐步检查窗口和输出一致，并要求 delete 在 decoding 期间确实释放 CPU 数据。
- 同配置 `--gc-bench 5`：保留 LRU 的不采集特征路径，避免仅比较 redundancy off/delete 而漏掉采集成本。
  分别加载模型的比较可能受显存换页影响；补充 `--gc-policy-bench 2`，共享一次模型加载，
  按 LRU、redundancy、redundancy、LRU 顺序创建上下文，每段分别预热，再测两轮。
- `--gc-redundancy-bench` 另在同一组真实 Mean-K/Query 上比较两策略各删 4 块后的
  Query 权重留存和两两相似度。这是选择信号诊断，不是独立回答质量指标。
- `scripts/test_server_historical_gc.py --policy redundancy`：从公开 HTTP 入口验证
  取消、reset、精确前缀、SSE、容量拒绝和后台 CPU 释放。约束输出 Z 的用例只测流程。

性能结果需同时报告吞吐、单步延迟、CPU 实际释放和特征采集成本。
固定 GPU 窗口下的输出相同不能证明后续需要重新读取被删历史时的质量无损；
没有真实 agent 任务集之前，不宣称 redundancy 优于 LRU 的任务正确率。

## 2026-10-21 验收记录

使用同一台 RTX 3080 Ti Laptop（16 GiB）和用户提供的 Qwen3-30B-A3B Q4_K_M GGUF；
不是 Llama 模型。CPU 线程 4，batch 64、microbatch 16，f16 KV，GPU budget 1536、generation reserve 256。

本地相关回归 15/15 通过（另有用户自带 demo 通过，但不属于本次提交）；
远程五项相关测试通过；redundancy/delete 的 HTTP 生命周期验收 33/33 通过，退出码 0。受控样例中，LRU 删除较老的独有块，redundancy 删除两个重复块，
保留独有块及较新的重复代表。真实 RawStore 测试同时确认 CPU 数据释放与代际复用。

redundancy 五轮正式对照通过，另有每模式一次预热；所有模式的输出 token 相同、GPU 窗口稳定。
每次 delete 请求在 decoding 期间退休并提前释放 27 个块的 CPU payload，完整槽回收数为 0，
这是仍持有 GPU 许可时的预期行为。历史 payload 从 139,493,376 降至 51,904,512 字节，
差额约 83.53 MiB，不将其解释为进程 RSS 下降。

- redundancy/off：吞吐中位数 24.081 token/s，单步 P95 中位数 45.399 ms。
- redundancy/delete：吞吐中位数 24.032 token/s，单步 P95 中位数 45.686 ms。
- 按同轮 off 配对，delete 吞吐变化中位数 -0.50%，单步 P95 +0.26%；
  五轮 P95 变化范围 -0.24% 到 +4.65%。这只是该短负载的观测，不是普遍性能保证。

在同一组真实特征上各删除四块，LRU 删除原位置 `[32,64,96,128]`，
redundancy 删除 `[160,128,64,32]`。剩余 Query 权重从 0.4324 增至 0.4572，
但剩余块两两平均相似度也从 0.9224 升至 0.9287（该项没有改善）。
这说明组合评分改变了选择，不能据此称它在所有指标上优于 LRU。

证据暂存于本地 `artifacts/redundancy-gc/` 及远程隔离目录
`/tmp/kvmem-gc-99bef27.3CD9MU/redundancy-gc/`。预验收发现的 recency 采集开关缺口已修复；
缺失真实 Query 的检查保留在测试中。原始失败日志也保留，没有把重跑结果覆盖成首次通过。

### 包含特征采集成本的补充对照

随后在一次模型加载内进行 LRU → redundancy → redundancy → LRU 的 ABBA 对照，
每段每模式预热一次，再测两轮，总计 36 次请求，24 次计时样本。两策略输出 token 全部相同，
每个 delete 样本也都真实释放 CPU 数据。

但该主机出现明显的跨上下文速度漂移：四段 delete 吞吐分别约为
24.03、18.88、18.94、16.31 token/s。同一种 LRU 在首尾两段差异很大，
不能把它简单归因于策略。日志中的 GPU 模型分配量为 17,587 MiB，超过本机 16 GiB 物理显存；
换页、频率等因素未受控，确切成因尚未定位。

本次 ABBA 中，redundancy 输入处理耗时约 4.715 秒，LRU 约 3.998 秒，
配对增幅中位数约 18.0%；说明新增特征路径有值得关注的成本。
整体 decode 吞吐配对变化中位数 -2.85%，但范围从 -21.95% 到 +17.12%，
这个区间不足以支持可靠的整体性能结论。前面的约 0.50% 只描述同一特征路径内开关 GC 的增量，
**不能宣称新增 redundancy 策略的总开销只有 0.50%**。

因此，本版交付结论是：可选策略功能和后台释放路径已验证，默认仍保持 LRU；
回答质量收益与稳定的整体性能收益均未证明。若后续优化，应优先减少不参与评分层的特征采集，
并在显存充足或另行验证过的模型放置配置下复测，之后才考虑扩展更多策略参数。

### 连续容量复用与交付校验

同一历史仓库连续 40 次 HTTP 请求，redundancy/delete 压力验收 128/128 检查通过，
退出码 0；观察到最多 54 个已退休块、52 个完整槽位回收。请求在实际空位出现后恢复准入，
回收后仍保持同一仓库。该结果区别于稳定 GPU 窗口测试的“只释放 CPU 数据、暂不复用槽位”。

远程将提交的产品/测试源码归一化 SHA-256 与本地清单逐项相同；增量计算图补丁
在临时副本中完成反向还原、重新应用及字节一致性检查。Linux 补丁脚本语法和
Windows PowerShell 构建脚本语法通过检查。

本地完整日志归档：`artifacts/redundancy-gc/evidence.tar.gz`；
共享模型对照汇总：`artifacts/redundancy-gc/policy-summary/summary.json`；
源码/二进制核验：`artifacts/redundancy-gc/verification.json`。

日志归档 SHA-256：`b9012145f6ba81d2defb59ebfd9f53b7598510b8e91a192ce5f2c4ecde909304`。

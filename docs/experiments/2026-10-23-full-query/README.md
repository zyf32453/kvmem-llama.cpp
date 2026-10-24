# 第 4 点：完整输入 Query 的实现与分阶段验证

**2026-10-24 阶段性提交说明：**本页较早的“已启动/正在运行”段落保留启动时点记录；r5 GC 长任务现已结束，**未通过质量验收**。八轮中前七轮 HTTP 200/stop，第八轮因前缀失配触发 reset，需重建 71,735-token 输入，但 CPU historical 容量仅 1,024 块，服务端以 HTTP 503 拒绝预约，未进入本轮 decode。独立检查前七轮原样网页为 **27/29**，缺少第八轮及其最终 `round` 字段；零修复，不能算完整生成。此前发生真实 CPU payload 释放（累计 1,501 次）和槽位回收（累计 1,705 次，生命周期计数重叠），最大候选 56，无 worker/policy failure；这些只证明路径执行，不证明质量合格。[终态审计](gc-prefix-refusal-r5/audit-summary.json)、[浏览器报告](gc-prefix-refusal-r5/browser-checks.json)及[前缀证据](gc-prefix-refusal-r5/continuation-evidence.log)已归档。此前自检的“间接相似却同时删除两块”反例仍未修复；两次 decode 间完成 GC 的时序保证和正式 p95/吞吐也未验收。本提交是供审查的阶段版本，不是完成声明。

2026-10-23 的实验快照为 **`full-query-all-decode-r3`**，25 项源码指纹见 [r3 manifest](source-manifest-all-decode-r3.json)。本轮移除生成 Query 捕获的 thinking armed/range 限制，并修复答案阶段重复发布 thinking-close 事件引发的高频 flush；启用 thinking 且 tracker 已 armed 时，回收候选仍为 thinking-only。该快照以 HEAD `660e46ae4a1a2079e771b8c43be5ceb03c4d75b5` 为基底，当时改动尚未提交；本文件开头记录后续终态。

本地证据：事件去重前全回归 **18/18**、去重后受影响测试 **2/2**、Python **27/27（5+11+11）**。r3 已完成远端 CUDA 构建、原生测试 **3/3**、25 项 hash 与补丁 0014 核对；三轮 no-thinking 采集探针及 **6 请求的原生路径诊断均通过**。后者观测到真实 CPU payload 释放，但仍不是完整任务质量或正式性能验收。

截至2026-10-24，r5 off 已完成 **8/8轮、零截断、零修复、原检查器29/29通过**；但第8轮前缀未命中、重算了完整输入，因此严格连续历史检查失败，driver/controller均exit1。这只是关闭GC的功能质量参考，不是GC质量或连续缓存验收。相同任务的 `redundancy-thinking-native-262k-r5` 已结束且未通过质量验收，详见上方终态审计；正式 p95/吞吐实验尚未启动。此前旧r1移动错误、pilot输出截断及r4的65K上下文拒绝均保留，见后文。

## 改动范围

**2026-10-24 自检新增未修复问题：相似关系的传递合并不能保证保留直接替代块。** 最新本地构建重新运行18项相关CTest全部通过，源码与r3/r4合并manifest的27项指纹一致；但新增合成反例失败：A–B、B–C余弦均0.939693，A–C为0.766044，门槛0.9。A为mandatory，thinking候选池包含三块，准入压力路径禁止有损兜底；策略仍将B和C都以Redundancy原因删除，只保留A。C没有任何达到门槛的保留替代块。原因是相似边通过并查集合并成连通分量后仅保护一个winner，原有“完全相同副本”的测试未覆盖这种非传递情形。原始[反例源码](self-audit-20261024/transitive-substitute.cpp)、[结果及指纹](self-audit-20261024/result.json)、[本次18项回归日志](self-audit-20261024/ctest-18.log)已保存。此时未修改产品算法，运行中的r5仍是该版本；即使网页功能通过也不能消除此反例，应修复直接替代关系并补回归后再宣称策略完成。

导师指出的第 4 点有两个独立要求：生成期间使用新的 Query；输入问题不能只用末尾几个 token 代表。此前只在 thinking 范围内更新生成 Query，输入也只保留末尾 32-token 均值。本次先实现全部输入分段，再进一步移除生成捕获的 thinking armed/range 限制；在 thinking 已 armed 的实验路径中，回收候选仍为 thinking-only，不能把采样范围与可回收范围混为一谈。未 armed 的非 thinking 路径保留原策略；若没有显式设置 candidate cap，它仍使用旧的参照范围，不能声称也受 56 块上限约束。

这里“整个输入 Query”指本轮定位出的完整用户问题 token 范围，不是把全部聊天历史当作问题，也不是全文语义无损编码。服务端在 `clamp_query_span()` 之前保存完整范围，通过 `llama_kvmem_set_gc_query_span()` 交给 GC。GPU 检索的 `query_max_tokens`（默认 512）仍只限制检索通道，不能再把 GC 问题截成末尾 512。

对选中的 attention 层（沿用最多 4 层）进行以下处理：

1. 在 prefill 收集问题范围内的**每一个** Query token，连续 32 个组成一个分段，保留本轮全部分段。microbatch 边界不切断分段；不足 32 的尾段也保留。因此 4,096-token 输入有 128 个分段，不是仅保留最后 32 tokens。
2. 等 prefill harvest 全部排空，核对各层分段位置和 token 数一致、采集数等于服务端完整范围长度，再发布不可变快照。收集未完成时，不使用上一个问题或半份问题进行新的回收决策。
3. 对每个历史块逐段评分，取各分段的重要性最大值。每段内部沿用原有 GQA 分组最大值和层/head 聚合。不同段之间不再先求全局 Query 均值，避免开头的重要方向被很长的后缀稀释。
4. 完整问题评分与生成期间的观察评分共同取最大值；生成侧仍保留慢窗 32、快窗 8、最近 4 个慢窗。当前 adapter 的生成 Query 捕获不再要求 thinking 已 armed 或位置属于 thinking 区间，使答案/no-thinking 位置也可进入观察窗口；r3 的 no-thinking 模型探针及合成路径诊断已验证生成采集。thinking 已 armed 时，候选块仍按 thinking-only 规则筛选。
5. 长问题首次评分可能跨过很多次后台轮询。完成问题评分和块间相似度比较后，在选代表块/排序之前，再冻结一次最新生成 Query，分步刷新候选的重要性。沿用问题评分缓存，并保留本轮先前已观察到的重要性峰值，避免最后用十几秒前的生成关注点删除。
6. r3 修复 thinking 结束后的事件重复发布：此前每个答案 token 都会再次通知 closed，触发 `flush_observation()`，把正常窗口变成高频刷新。现在 `ThinkingSpanTracker::take_update()` 仅在边界或 closed 状态变化时发布，服务端 6 处路径统一经过这一接口。同一边界从 open 变 closed 仍必须发布；空区域、重新进入 thinking 和新问题重置分别处理。

实现入口：`kvmem/include/kvmem/historical_kv_redundancy.hpp`、`kvmem/src/host/historical_kv_redundancy.cpp`、`src/adapter/llama-memory-kvmem.cpp`、`tools/llama-kvmem-server.cpp`。

## 计算与内存约束

令 T 为问题 token 数，S 为分段数（连续输入时为 ceil(T/32)），L 为所选层数，W 为每层 Query 向量宽度，B 为本轮候选块数。

- 收集及聚合为 O(T×L×W)，问题表示为 O(S×L×W)；首次对 B 个块评分为 O(B×S×L×W)。这是 Query 与块的比较，不引入 Query token 的 T×T 矩阵。
- 后台 `read_bounded()` 每次最多计算 8 个新问题分段。未算完整就返回 `Yielded`，策略保存游标；**半份评分不得参与排序或发出回收命令**。当前 adapter 每次 policy step 最多读 16 个特征，所以上界是每步 128 个分段评分，而不是“整步最多 8 段”。长问题通常在第一个未完成块处直接 yield。
- 评分按问题快照 epoch、块 handle/generation/content_version 缓存。相同问题和内容再次访问不重复问题点积；新问题和槽位复用会使缓存失效。完全相同问题已存在于可验证的精确缓存前缀内时可复用整个问题快照。
- 排序前的刷新只对最多 4 个慢观察均值和 1 个快观察均值重新评分；每步仍受 `budget.scan` 限制，不重复全部问题分段点积。生成 Query 在刷新快照之后仍可继续变化，因此这是有界延迟的观测，不是与每个 decode 完全同步的评分。
- 一份问题表示的逻辑预算为 64 MiB，包括分段元数据和 float 数组；vector 容量与分配器开销另计。生产者构建、当前发布、后台冻结的旧快照最多同时保留三份，因此它**不是整个进程 64 MiB 的硬上限**。通常旧快照随下一次后台轮次替换，仍须远端测实际 RSS。
- 超预算、缺层、漏采、非有限数或无法定位完整文本范围时拒绝该问题，暂停该轮 redundancy 选块并记录 `full_question_invalid=1`；不静默截断，不自动回退 LRU。内存紧张时因此可能无法满足后续请求准入，应把此情况视为验收失败并分析原因。
- 分段数越多，完成第一次评分所需的后台轮询也越多。5 ms poll 下，56 个全新块、每块 512 段会需要约 3,584 次分段读取；大部分需要 yield。其墙钟完成时间可能达十几秒，不能用下面的纯计算时间代替回收响应时间。远端须测首次回收等待和 Ready 水位能否跟上生成。
- 将采样扩展到全部 decode 会增加最多 4 个所选层的 Query D2H；现有两槽 harvest 都忙时可能等待，原捕获路径也包含同步，因此**不能宣称零阻塞或无前台开销**。单 token 捕获依赖维护补丁 `patches/0014-historical-redundancy-capture.patch`；本地 `llama.cpp` 尚未应用该补丁，CPU 测试和服务端单文件语法编译不验证这条 GPU 路径。远端 r3 已核对该补丁，后文的 no-thinking 探针与合成诊断提供实际 GPU Query 捕获证据；完整任务质量与正式性能尚待验收。

所有问题段都参与，并不代表信息无损：段内均值仍可能稀释或抵消少数 token；Mean-K、层抽样及 pre-RoPE 评分仍是近似。最大值聚合也可能增加保留量。不能据此宣称安全删除、全文语义等价或复现了论文全部机制。这个改动消除的是“仅取末尾”的固定位置盲区。

## 本地验证

### 最新 r3 的测试归属

[ctest-all-decode.txt](ctest-all-decode.txt) 的 **18/18** 是移除生成捕获 thinking 限制后的全回归，执行于事件去重修复之前；r3 去重后重新运行受影响的 Query coverage 与 thinking tracker，结果 **2/2**，见 [ctest-query-events-r3.txt](ctest-query-events-r3.txt)。旧 closed 通知循环先使回归失败；改用变化事件后，256 个普通答案 tokens 只增加 **8 个慢窗口样本**。测试同时覆盖同一边界 open→closed、空 thinking 区域、新 thinking 区域与新问题重置，不以简单“样本数大于零”代替事件正确性。

Python 日志覆盖 **5/5**、长 Query probe 检查器 **11/11**、性能解析器 **11/11**，合计 **27/27**；服务端单文件语法编译通过。上述均为本地证据，不能证明 r3 已完成 CUDA 链接、单 token D2H、答案/no-thinking 持续更新或性能验收。`source-manifest-all-decode-r2.json` 只保留曾打包上传的版本记录，r2 未构建/未运行；后续应以 25 项 [r3 源码指纹](source-manifest-all-decode-r3.json) 及新构建二进制 hash 标记结果。

r3 随后已在远端完成全部 CUDA 目标构建，并通过 redundancy、Query coverage、thinking tracker **3/3 原生无模型测试**。25 项源码 hash 与 manifest 一致，单 token 捕获补丁 0014 已核对；隔离树没有 `.git`，上游来源以原仓库固定 pin 加实际 graph 文件 hash 核对，不把隔离目录误写为已通过本地 Git commit 验证。新服务二进制 SHA-256 为 **`824f31ba118e6baeae288b3a8e4ffe663246629958a6446cdf2646cccfd466b3`**，见 [构建/来源核验](build-all-decode-r3-result.json)、[构建日志](build-all-decode-r3.log) 与 [原生 CTest](ctest-all-decode-r3.log)。r3 三轮 no-thinking 探针的独立结果见下文；构建证据与模型运行证据分别记录。

### 首次 full-query 实现的既有本地证据

MinGW-w64 GCC 16.2.0、Release、Windows x86_64，CPU-only。执行：

```powershell
$env:PATH = 'C:\mingw\mingw64\bin;' + $env:PATH
cmake -S . -B build-gc-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_ENABLE_NVME=OFF
cmake --build build-gc-host -j 2
ctest --test-dir build-gc-host -R '^(historical_kv_|thinking_span_tracker_test|raw_kv_gc_release_test)' --output-on-failure
build-gc-host/bin/historical_kv_query_coverage_test.exe --bench
python -m py_compile scripts/test_server_historical_workload.py
python tests/test_historical_query_log.py -v
python tests/test_full_query_probe.py -v
```

相关 CTest **18/18 通过**，原始输出为 [ctest.txt](ctest.txt)。新 `historical_kv_query_coverage_test` 覆盖：

- 4,096-token 问题的开头、中间、结尾各放一条可区分方向的信号；microbatch 分别为 1/17/512/4096，验证评分不因位置或切批而丢失。它是构造的向量测试，不是自然语言语义测试。
- 跨段相反方向不被全局均值抵消、33-token 尾段、分离范围、完整缓存复用、新问题失效、生成关注点更新。
- 缺层、所有层同时漏 token、超预算、NaN、重复结束时长度矛盾拒绝发布；下个合法问题恢复。
- 长问题分步计算时没有提前回收命令；问题切换会否决旧轮次/延迟命令；并发构建新问题不修改后台旧快照。
- 在长问题评分的第 5 次 step 改变生成 Query 方向，验证**同一轮**最后的回收决策使用新关注点，同时断言全部问题点积只计算一次。这一用例先复现旧实现失败，再在加入排序前刷新阶段后通过。

既有 redundancy 测试同时验证了实际 CPU payload 释放、槽位 generation 复用后的特征失效等，不能只运行新测试代替回归。

另有 **5/5 Python 日志验收测试通过**：逐请求把完整 Query 范围长度与结束时实际采集数对应起来，拒绝非零但不完整的后缀、用重复结束日志冒充多轮、漏掉某轮结束、范围矛盾或采集失败。它们验证验收脚本不会把明显不完整的日志判为通过，不代替模型测试。

新增三轮长问题探针 `scripts/fixtures/full-query-coverage-workload.json`，每轮把互不相同的三个字段约束放在问题头部、中部和尾部，要求模型只返回精确 JSON；各轮字段值不同，以检查是否误用前一轮答案。目标输入约 2K–4K tokens，实际长度由模型 tokenizer 和运行日志确定，验收最低要求每轮 1,024 tokens。`scripts/check_full_query_probe.py` 独立核对原生聊天模板、实际请求的完整多轮历史、前缀缓存复用、每轮完整问题范围、实际采集 token 数、ceil(T/32) 个分段，以及输出的三个字段。生成侧 Query 采样次数仅记录，不把计数非零当作关注点及时更新的质量证明。

探针检查器的 **11/11 本地无模型测试通过**，涵盖旧轮字段冒充当前输出、重复 JSON 键、额外解释、后缀段数伪装成全量、问题范围重复、截断、缺少前缀复用、worker 失败和证据缺失。直接运行 Python 时，本机沙箱默认临时目录有访问限制；将 `tempfile.tempdir` 指向工作区现有 `tmp/full-query-20261023` 后重跑通过，未修改测试逻辑。这 11 项是**检查器本身的测试**，与下文三轮真实模型探针是两类证据。

该短探针只验收“长输入 GPU 采集与输出约束”，允许没有 GC 活动。现有通用 workload driver 可能仅因没有回收而返回非零；须检查原始摘要，再以独立 `full-query-probe-check.json` 判定本探针，不能忽略采集、请求或 worker 错误。即使探针通过，正确回答也可能来自普通 attention；不证明 GC 排序保留了这些约束，也不代替有实际 CPU 释放的游戏质量与性能验收。

服务器 `tools/llama-kvmem-server.cpp` 已针对仓库固定 upstream `7fe450e19305b828c199d602c23a8337aaa1f03b` 及维护补丁完成单文件编译（Windows 需 `_WIN32_WINNT=0x0A00`）。完整 adapter/服务构建因本机缺 `cuda_runtime.h` 未完成；`GGML_CUDA=OFF` 不能把这个 GPU adapter 变成纯 CPU 版本。单文件编译不能证明链接、GPU Q 捕获和实际模型路径正确。

### 合成 CPU 开销探针

[local-cost.jsonl](local-cost.jsonl) 为本机一次运行的原始数据：4 层、head_dim 128、8 KV heads、32 Query heads，一个历史块，固定合成向量，256-token 批量输入；每个尺寸测一次首次评分和 100 次缓存读取均值。没有模型、GPU/D2H、HTTP、GC poll 等待或 decode。CPU 标识 `AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD`，16 逻辑处理器；未固定频率。

- 512 tokens：16 段，约 1.00 MiB；收集 9.24 ms，首个块评分 0.291 ms。
- 4,096 tokens：128 段，约 8.02 MiB；收集 78.62 ms，首个块评分 2.65 ms。
- 16,384 tokens：512 段，约 32.09 MiB；收集 323.98 ms，首个块评分 9.45 ms，分 64 次读取；该次最大读取耗时 0.207 ms。缓存读取均值约 0.009 ms，不再计算问题点积。

这些数用于暴露成本，**不是 p95、前台延迟或吞吐验收**。首次评分还要乘实际新评分块数；收集时间也未包含 GPU 传输。真实参数需要模型验证后才能确认合适。

## 远端接入进度与下一步验收

2026-10-23 远端已恢复。以下 r1 运行所用的前一 revision 同步到隔离目录 `/tmp/kvmem-full-query-20261023`，初批 17 项源码 SHA-256 与本地核对一致，随后补入当时的 full-query benchmark 文件；归档的 [源码指纹](source-manifest-full-query.json) 共记录 18 项。该指纹不包含后续移除生成捕获 thinking 限制及更新 benchmark 的改动。旧实验目录保留。CUDA 配置为 GNU 11.4、CUDA 13.2.86、CUDA architecture 86、Release；硬件为 RTX 3080 Ti Laptop 16 GiB。沿用模型 `/home/yifan/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，文件大小 12,120,016,960 字节；环境、驱动及探针文件指纹见 [远端环境记录](environment-full-query.json)。

该 revision 在远端的 `historical_kv_redundancy_test`、`historical_kv_query_coverage_test`、`thinking_span_tracker_test` **3/3 无模型测试通过**，见 [CTest 输出](ctest-query.txt)。当时的完整 CUDA 服务端与 full-query benchmark 均已构建通过，后者见 [构建日志](build-full-query-bench.log)；该构建日志不覆盖最新版本。探针使用 `build-cuda/bin/llama-kvmem-server`；启动前记录的 SHA-256 为 `0799b3f6b667ce8dff192ba5dd43df6f08756cf2ad459de71e90690b2d4eb672`。此前旧尾窗源码的 `redundancy-thinking-final-r3` 已恢复为 8/8 轮完整生成、28/29 浏览器检查、0 修复，详见交接文档；它不能覆盖此次 full-query 改动。

### 三轮长问题真实模型探针：r1

远端输出为 `/tmp/kvmem-full-query-20261023/redundancy-gc/full-query-probe-r1`，原始副本位于本地 `tmp/full-query-20261023/full-query-probe-r1/`。工作负载 driver 和独立检查器均正常退出（exit 0）。参数沿用 IQ3、32 tokens/block、36K/16K/1K/recent 0、原生 thinking、CPU 1024 槽、Ready 352/384、redundancy/delete；[实验配置](probe-r1/experiment-config.json) 和 [实际服务命令](probe-r1/command.json) 已归档。

- 三轮均 HTTP 200、正常 stop、无截断；实际用户问题分别为 **1,813 / 1,813 / 1,812 tokens**，每轮 **57 段**，完整范围与采集计数逐轮一致，`invalid_questions=0`。实际长度低于 fixture 的 2K–4K 目标区间，但超过 1K 最低门槛及旧 512-token 截断长度，不能描述成已测 4K 输入。
- 头部、中部、尾部共 **9 个输出字段全部匹配**；两次续轮前缀命中分别为 **2,005 / 3,934 tokens**。结论见 [独立探针报告](probe-r1/full-query-probe-check.json)，日志证据见 [范围与结束计数摘录](probe-r1/capture-evidence.log)。
- 第三轮观测到实际 GC 活动：累计 `committed=4`、`host_released_early=4`、`reclaimed=0`，无 worker/policy 失败。前两项计数重叠，不能相加；它们说明 4 个块的 CPU payload 已提前释放，不代表 4 个槽位已完成最终回收。问题评分累计 350 次、43 次 yield；生成慢/快采样累计 6/152 次。它们证明相关路径被执行，不能单凭计数证明实时关注点更新质量。
- 各轮问题表示均为 **5,614,272 bytes（约 5.35 MiB）**。1 秒采样、51 个记录的进程 RSS 峰为 **12,222,456 KiB（约 11.66 GiB）**，最后一次为 **3,483,484 KiB（约 3.32 GiB）**；这包含模型加载与全部服务内存，不能视为 Query/GC 增量开销或长期内存上界。原始短内存记录见 [进程内存](probe-r1/process-memory.jsonl) 与 [GPU 内存](probe-r1/gpu-memory.csv)，计算摘要见 [审计摘要](probe-r1/audit-summary.json)。

这是一次**完整输入采集与小规模输出约束 smoke**。当时生成 Query 捕获仍受 thinking armed/range 限制，采样计数不能证明答案或 no-thinking 阶段持续更新。虽然本次碰巧发生了 GC，普通 attention 仍可能解释字段提取成功；它不证明长期任务质量、GC 删除的语义安全、全部槽位回收完成或 p95/吞吐达标。原始 [运行摘要](probe-r1/summary.json) 与独立报告均保持相应质量/性能未验收标志。文档目录只归档精简证据；原始 requests、results 和完整 server trace 留在上述本地/远端目录，SHA-256 已写入审计摘要。

### 新版八轮游戏：r1 因已确认功能失败提前结束

此次沿用零自动修复的八轮工作负载，远端目录为 `/tmp/kvmem-full-query-20261023/redundancy-gc/redundancy-full-query-thinking-r1`。在第二轮完成后，对模型原样生成的 `state.js` 与 `physics.js` 执行原检查器中的单项 `diagonal movement normalized`：输入 forward=1、strafe=1、dt=0.1、speed=3.5，预期位移 0.35，Chrome 实测 **0.4949747468**，误差 **0.1449747468 > 0.01**，判定失败。没有修改生成代码；浏览器无异常和非预期网络请求。证据见 [单项浏览器报告](game-r1-early-stop/report.json)、[原样 state.js](game-r1-early-stop/state.js) 和 [原样 physics.js](game-r1-early-stop/physics.js)。

这个错误在**尚未发生任何 GC 提议、提交或 CPU 释放**的前两轮就已生成；前两轮结束计数全部为 0，见 [GC 与采集日志摘录](game-r1-early-stop/capture-gc-evidence.log)。`physics.js` 的 SHA-256 为 `dd649052899485b889d264d4f50265a5b12c85305d5d46aca202bc7384842d57`，与旧 final-r3 中同文件完全相同。文件内先归一化方向、随后仍以原始长度计算移动距离，导致斜向加速。现有证据不能把该错误归因于 GC 删除；也不能据此证明 GC 对后续质量完全没有影响。

原 [八轮文件契约](game-r1-early-stop/workload.json) 在后续六轮不再修改 `physics.js`，[控制器命令](game-r1-early-stop/controller-command.json) 又明确 `--max-repairs 0`，因此已确定本轮无法满足全部功能检查，主动对本次 driver 发 SIGINT 提前结束，保留 [终止原因](game-r1-early-stop/early-stop-reason.json)。这不是 GC 崩溃，也没有执行完整 29 项检查；不得报告为“28/29”或“8/8 已完成”。

最终归档以 [运行摘要](game-r1-early-stop/summary.json) 为准：实际为 **3 个完整回复、尝试了 4 轮、计划 8 轮、0 修复**。在检查和结束的间隔内，第三轮 `weapons.js` 也已正常 stop；第四轮 `bots.js` HTTP 已建立，但无完整回复，因主动终止而中断。前三轮问题采集分别 227/102/144 tokens、8/4/5 段，`invalid_questions=0`；第四轮只有问题范围，没有结束日志，所以整轮组的 `full_question_coverage_observed=false` 是不完整运行的结果，不能冒充全八轮覆盖通过。

第三轮结束时累计 `committed=170`、`host_released_early=170`、`reclaimed=0`，无 worker/policy 失败；`max_round_features=53`，生成慢/快采样累计 359/12,918 次。这一较晚发生的 GC 活动不改变第二轮缺陷发生时尚无回收的事实；计数重叠，不能相加。主动中断后控制器退出码为 1，已完成归档，自有 driver/server 进程均退出。逐轮状态、源文件指纹和未归档大文件的 SHA-256 见 [失败审计摘要](game-r1-early-stop/audit-summary.json)，完整 trace 与请求/回复保留在 `tmp/full-query-20261023/redundancy-full-query-thinking-r1/remote/`。

### 旧二进制 off movement pilot：r1 未完成生成

为排除基础移动错误，使用独立 fixture `explicit-movement-v1` 仅澄清第二轮原有的总速度/斜向归一化要求，原八轮 fixture 没有修改，API、参数、其余轮次和原 29 项检查均保持。逐字段比较确认仅 turn 2 的文本不同，见 [审计摘要](movement-baseline-pilot-r1/audit-summary.json) 和 [实际工作负载](movement-baseline-pilot-r1/workload.json)。该两轮 pilot **GC=off**，使用旧 r1 二进制 `0799b3f6...`，不是当前 r3；[配置](movement-baseline-pilot-r1/experiment-config.json) 与 [服务命令](movement-baseline-pilot-r1/command.json) 已归档。

第一轮正常 stop，生成 1,363 tokens；第二轮生成 **16,384 tokens** 后以 `finish_reason=length` 结束，耗时 **1,034.745 秒**，最终 `content` 为空，仅有 53,334 字符的 reasoning，**没有生成 `physics.js`**。driver 退出码为 1，[运行摘要](movement-baseline-pilot-r1/summary.json) 记录 1 轮完整、1 轮截断。四项提前移动检查因第二轮输入不完整而全部未执行，[检查报告](movement-baseline-pilot-r1/movement-checks.json) 为 **0/0，不是 0/4**；也未执行完整 29 项检查。

没有 GC 回收，不能将此结果归为 GC 质量失败，更不能据此反复修改 redundancy 算法。该 pilot 也不能证明新 fixture 已消除移动错误或 r3 已通过验收。原始 reasoning 和回复保留在 `tmp/full-query-20261023/movement-baseline-pilot-r1/`，文档只保存精简摘要和原文件 SHA-256。`summary.json` 的 `required_gc_activity_observed=true` 在 off 模式只是“无需回收”的条件满足，不能解释成实际发生 GC。

### r3 no-thinking 探针：输入全覆盖与生成采样通过，无实际 GC

该探针使用 r3 二进制 `824f31ba...`、原生模板，`enable_thinking=false`、thinking policy 未 armed，仍开启 redundancy/delete。[独立报告](probe-no-thinking-r3/full-query-probe-check.json)、[配置](probe-no-thinking-r3/experiment-config.json)、[服务命令](probe-no-thinking-r3/command.json) 与 [运行摘要](probe-no-thinking-r3/summary.json) 已归档。

三轮均正常 stop，分别生成 **52/52/51 tokens**，reasoning 为空；问题仍为 **1,813/1,813/1,812 tokens、各 57 段**，`invalid_questions=0`，头/中/尾 9 个字段全部正确，后两轮前缀命中 **1,918/3,796 tokens**。慢窗口累计 **1→2→3**、快窗口累计 **45→90→134**，证据见 [逐轮结束日志](probe-no-thinking-r3/capture-evidence.log)。这证明在没有 thinking 标记、policy 未 armed 时，生成 Query 仍进入了观察窗口；计数是每轮结束的累计值，不是每个 decode 的采样时刻或延迟轨迹。

本次 `proposed/committed/host_released_early/reclaimed` 全为 0，worker/policy 无故障。通用 driver **exit 1** 对应唯一未满足的 `required_gc_activity_observed=false`；其余请求完成、完整采集、前缀复用与错误检查通过。独立采集/输出 checker **exit 0**，按本探针预定范围不要求回收，不能把这两个退出码混写成模型或捕获失败。也不能用无 GC 的探针证明删除质量、回收进展或 p95/吞吐。简要输出、计数和原始文件 SHA-256 见 [审计摘要](probe-no-thinking-r3/audit-summary.json)；长 requests/results/trace 留在 `tmp/full-query-20261023/full-query-no-thinking-r3/`。

### r3 原生合成路径诊断：通过，非正式性能验收

`full-query-path-gate-r3` 已结束，进程 exit 0、严格 `full_query_validation.status=passed`。使用 r3 的 `kvmem-historical-access-test`，off/shadow/delete **各 1 次 warmup＋1 次 measured，共 6 个请求**；[启动记录](path-gate-r3/launch.json)、[结束记录](path-gate-r3/result.json) 和 [路径检查报告](path-gate-r3/path-gate-check.json) 均已归档。

- 每个请求的完整输入均为 **6,144 tokens、192 段、18,911,232 bytes 逻辑问题表示**，`invalid_questions=0`。合成 thinking 仅标记输入 `[0,6144)`；256 个生成位置不标 thinking，off 明确 disarm。6 个请求的 256 步 decode 均新增 **8 个慢窗口样本、249 个快窗口样本**。
- 有效策略候选池最大 **56**，worker/policy failure 均为 0；off 没有 GC worker，shadow 没有真实释放。两次 delete（warmup 和 measured）分别在 decode 期间提前释放 **48 个 CPU payload**，每次有 3 个发生释放的 step，`first_release_step` 分别为 **129/111**；两次 `slots_reclaimed` 都为 **0**，不把 payload 提前释放当成槽位最终回收。
- 六组输出 token 一致，GPU 工作窗口稳定。这个结果验证了所构造负载中的全输入采集、非 thinking 生成观察、有界选块与实际 CPU 释放路径，不能证明自然语言任务质量或删除语义安全。

该诊断明确保持 `gc_quality_accepted=false`、`formal_performance_accepted=false`。只有每 mode 一个正式计时样本，原始时延保留在 `tmp/full-query-20261023/full-query-path-gate-r3/` 的 run log、summary 与 runs，**不据此作 p95 达标结论**。本次没有 RSS 采样，18,911,232 bytes 是逻辑问题表示，不能当作进程 RSS 或内存上界。历史非 thinking 的 29/29 结果也不能替代当前原生 thinking 任务验收。

### 尚待完成

1. 修复第八轮前缀失配及 reset 后容量不足的问题，再重跑 `redundancy-thinking-native-262k-r5` 的严格连续质量验收：八轮完整生成、零修复、原29项检查通过、发生真实CPU释放后继续多轮生成，并且全部续轮复用历史。不能用off的29/29或本次GC的27/29代替这一验收；还要修复上方的直接替代反例并加入回归。
2. 保留并调查off第8轮的精确前缀失配。已确认服务端重算66670-token输入；原生模板空白整理和重新分词是待区分的可能原因，尚未证明根因。GC组若发生重算，严格检查必须失败，不能恢复已删除内容后仍声称持续GC质量通过。现有小探针也不能替代这一要求。
3. 按“先质量、后性能”的顺序，**本次未启动正式 p95/吞吐配对实验**。当前 `kvmem-historical-access-test <模型路径> --gc-full-query-bench <1..20>` 仅把合成输入 `[0,6144)` 标为 thinking；256 个生成位置不标 thinking，off 明确 disarm thinking policy。所有 mode 都要求慢/快 Query 观察计数随这 256 步增加；shadow/delete 的回收候选仍仅来自 thinking 块。这属于**合成性能测试**，不是自然模型 thinking 或网页任务质量验证。三组都包含 Query 收集，因此应同时报告 prefill、后台增量 decode p95/吞吐、首次回收等待和 RSS，不能把它当成未接入 Query 收集的原始模型基线。保留此前目标 p95 增幅不超过 5%、吞吐下降不超过 3%，未测前不宣称通过。

完整执行入口和旧证据边界见 [测试交接](../../historical-gc-testing-handoff.md)。

### 正在核验的质量基线：服务端采样配置

既有 workload driver 固定发送 temperature=0，覆盖服务端 thinking 默认的 temperature=1、top_p=0.95、top_k=20。为判断这一测试设置是否影响持续生成，新增显式 `--sampling-profile server-defaults`；默认仍为 greedy，旧实验不会被静默改义。新 profile 不发送采样覆盖项，仍固定 seed=42，保存请求与实际 sampling trace。

两轮 `movement-native-sampling-pilot-r4` 已完成，driver exit 0。使用 r3 服务二进制、未改动的原八轮 fixture、GC=off、native thinking 和全部原 KV/16K 参数，没有设置 reasoning budget。实际 trace 为 temperature=1/top_p=0.95/top_k=20、seed=42；请求中没有采样覆盖项。两轮分别生成 2433/9925 tokens，均 stop，第二轮命中前缀 2888 tokens。未改动的四项移动检查 **4/4 通过**，包括此前失败的斜向速度，实测位移正好 0.35。证据见 [pilot 审计](movement-native-sampling-pilot-r4/audit-summary.json) 和 [浏览器检查](movement-native-sampling-pilot-r4/movement-checks.json)。这只是前两轮基线通过，不能代替完整 29 项、GC 质量或性能验收；旧失败实验还存在构建/mode/提示差别，不把本次观察写成已证明贪心采样是唯一原因。

`off-thinking-native-full-r4` 已结束，控制器与 driver 均 exit 1。实际完成 **6/8 轮**，尝试7轮：前六轮全部HTTP 200、stop，无截断，生成2433/9925/15634/9380/9440/4659 tokens，所有已完成续轮都复用了前缀。第六轮结束累计 **52,555 tokens**，即使不计下一条用户消息，再加16,384预约输出也已达68,939，超过65,536；第七轮实际返回 **HTTP 400：`prompt + max_tokens exceeds n_ctx`**，第八轮未执行。这不是GC失败，也没有增加thinking限制或重试同一配置。原fixture、server-defaults、seed42、native thinking、0 repairs、CPU4096及原GPU预算均保留在[审计记录](off-context-limit-r4/audit-summary.json)和原始目录中。

对六轮原样生成的完整网页使用原检查器，得到 **27/29**：失败项为全部轮次完成，以及初始状态的最终`round`字段；后者由尚未执行的第八轮要求新增。原移动、射击、换弹、机器人、渲染、键盘/鼠标和重启等检查均通过，浏览器异常与非预期网络请求为0。详见[浏览器报告](off-context-limit-r4/browser-checks.json)和[原样文件](off-context-limit-r4/app/)。这是未完成任务的诊断结果，不是完整29项验收或GC质量证明。原始`summary.json`中的`refused_requests=0`只统计特定503情形，不能掩盖本次实际HTTP 400；判定须同时查看逐请求错误。

3894次内存采样中，进程RSS峰值12,222,604 KiB，末次5,261,576 KiB；GPU峰13,891 MiB，包含加载和整套服务，不能当成Query/GC增量开销。原始日志及内存采样保留在`tmp/full-query-20261023/off-thinking-native-full-r4/remote/`，文件hash见审计记录。

产品二进制与核心源码保持 r3；只有两份测试脚本增加显式采样 profile，指纹见 [quality-sampling-r4 manifest](quality-sampling-r4-manifest.json)。旧 r3 归档不覆盖新测试脚本。运行输入使用独立目录，未覆盖 r3 原始源码快照。

### 长任务准入上限修正：r5 off 与 GC 对照均已结束

在确认 r4 的模型进程退出、GPU回到空闲后，启动 **`off-thinking-native-262k-r5`**。逻辑上下文改为 **262,144**，对应模型加载日志中的`n_ctx_train=262144`；无GC质量参考的CPU capacity改为 **8192块 = 262144 / 32**，以保留完整历史。待验GC组仍为 **1024块、Ready 352/384**，将使用相同262K逻辑上下文。CPU容量不同的off组只作质量参考，不能直接用于同容量性能阈值结论。

GPU budget **36,864**、generation reserve/每轮最大输出 **16,384**、sink **1024**、recent **0**、块长 **32**、q8_0、原fixture、server-defaults、seed42、native thinking、0 repairs均保持，没有增加reasoning budget。核心源码和服务二进制仍为r3，两份测试脚本仍与r4指纹一致；不是新的算法revision。

[实际启动核验](quality-context-r5-launch.json)已确认`context_actual=262144`，GPU工作池仍为 **53,248 cells、1664 slots、1,853,882,368 bytes**，与r4一致；启动后GPU占用约13,891 MiB。这个核验只证明配置与工作池正确，不等于任务质量或性能通过。

`off-thinking-native-262k-r5` 已结束，八轮均HTTP200/stop，生成2433/9925/15634/9380/9440/4659/13913/1371 tokens；第2至7轮前缀均命中，第8轮为0。服务端第8轮执行`reset=1`、重新prefill完整66670-token输入，prefill383.399秒、生成126.069秒、整轮512.001秒。driver/controller因此exit1，控制器未进入自动浏览器反馈阶段；原始`summary.json`的`browser_all_passed=null`保留不改。模型进程已确认退出。原始目录为`tmp/full-query-20261023/off-thinking-native-262k-r5/remote/`。

随后对八轮原样输出独立运行**未修改的原检查器**，结果 **29/29、零修复、浏览器异常0、非预期网络请求0**。这是off功能质量参考通过，**严格连续历史检查未通过**，不是整个driver成功，也不是GC质量证明。证据见[审计](off-prefix-reset-r5/audit-summary.json)、[功能检查](off-prefix-reset-r5/browser-checks.json)、[前缀与reset日志](off-prefix-reset-r5/continuation-evidence.log)、[原样网页文件](off-prefix-reset-r5/app/)。5731次内存采样中RSS峰值12,222,320 KiB、末次9,037,976 KiB，GPU峰13,901 MiB；包含加载与重算，不是Query/GC增量开销。

已从本次GGUF提取[原生模板](off-prefix-reset-r5/native-chat-template.jinja)，确认模板保留thinking但会trim内容和reasoning。前缀失配的根因仍待区分，不能仅凭模板代码断言是空白整理，也可能涉及生成token的重新分词。该off结果允许作为无回收功能参考；**不能把它用于证明删除后历史未恢复，也不放宽GC组的严格前缀检查**。

`redundancy-thinking-native-262k-r5` 当时从控制器会话 **16266** 启动，现已终止；旧off会话22708也已结束，不要再轮询或重启。启动前确认模型进程已退出、GPU324 MiB/0%占用、r3服务二进制hash一致。GC使用CPU1024块和Ready352/384，其他模型、任务、采样、262K上下文及GPU参数相同；实际GPU池仍为53248 cells/1664 slots/1,853,882,368 bytes。见[配置预检查](quality-gc-r5-preflight.json)和[实际启动核验](quality-gc-r5-launch.json)。本地输出`tmp/full-query-20261023/redundancy-thinking-native-262k-r5/`，远端`/tmp/kvmem-full-query-20261023/redundancy-gc/redundancy-thinking-native-262k-r5`。启动不等于质量或性能通过。

GC组当时前四轮HTTP200/stop，生成2433/9925/15648/15244 tokens，耗时122.921/631.981/1138.067/1187.209秒，续轮前缀命中2888/12926/28729 tokens。前两轮无实际回收；第3轮开始实际CPU释放，见[第3轮检查点](quality-gc-r5-round3.json)。第4轮结束时累计committed775、host_released_early650、reclaimed391（生命周期计数可能重叠，不相加），worker/policy失败均0。完整问题100 tokens/4段，invalid_questions=0，慢/快观察累计1355/43226，max_round_features=56。第5轮曾以`reset=0 cached=44084`继续执行，证明实际回收之后完成了一轮并继续复用历史；**此处只记录当时的中途检查点**。第4轮结束时ready=0，尚不能据此宣称内存容量验收通过。原始计数、逐轮输出量和第5轮准入日志见[第4轮检查点](quality-gc-r5-round4.json)。最终第八轮拒绝及质量未通过，以本页开头的终态审计为准。

### off 前缀失配的词表复现（无推理）

使用同一GGUF的词表及隔离构建的原生`llama-common`模板渲染库，设置`vocab_only=true`、`n_gpu_layers=0`，不创建推理context。2.271秒内完成八轮请求重放，**八个输入token数全部与运行日志匹配**，七次从API reasoning/content重建的文本也都与模板中的canonical生成文本相同。前六轮的canonical生成token数与实际生成数相同；第七轮实际生成**13913**，canonical重新分词为**13912**，对应下一轮前缀**66571与66570**不一致。见[结果](off-prefix-reset-r5/prefix-vocab-probe-result.json)、[源码](off-prefix-reset-r5/prefix-vocab-probe.cpp)和[命令及指纹](off-prefix-reset-r5/prefix-vocab-probe-audit.json)。

这将不一致收敛到第七轮输出的往返表示，但**尚未保存原始采样token IDs或解析前字节**；API重建一致不能排除parser消费了额外格式空白。因此不能宣称已证明具体哪一个token发生变化，也不能把所有可能性归结为模板trim。本次探针没有修改产品源码或当时运行的GC配置。后来的GC组确实发生失配，已按严格条件判为失败；下一次需要保留原始边界证据，不能放宽续轮复用条件来通过验收。

# Historical KV 后台 GC 测试交接（给接手智能体）

更新时间：2026-10-24。本文记录**测试方法和证据边界**，不是对当前工作树的最终验收声明。开始执行前先运行 `git status --short`、`git rev-parse HEAD`，保存本次源码版本和测试二进制版本；本地可能存在未提交实验代码，旧报告不自动覆盖新代码。

最新长任务状态：r5 off 已完成8/8轮，原样网页29/29、零修复；但第8轮前缀失配导致完整历史重算，严格连续性检查失败，driver/controller exit1。它只作关闭GC的功能质量参考。r5 GC组现已结束且**未通过质量验收**：前7轮HTTP200/stop，第8轮前缀失配后试图reset，71,735-token输入超过CPU历史容量，HTTP503、没有开始decode；独立网页检查为27/29，缺第8轮及最终字段，零修复。此前已发生真实CPU payload提前释放1,501次、槽位回收1,705次，最大候选56，worker/policy失败为0；计数重叠且不证明质量合格。见[GC终态审计](experiments/2026-10-23-full-query/gc-prefix-refusal-r5/audit-summary.json)和[浏览器报告](experiments/2026-10-23-full-query/gc-prefix-refusal-r5/browser-checks.json)。完整连续质量、decode间隙时序与正式p95/吞吐仍待验收。

**2026-10-24自检发现新策略缺陷，尚未修复：** A–B、B–C相似度0.939693但A–C仅0.766044，门槛0.9；A为mandatory、thinking池三块、准入路径关闭有损兜底时，当前策略仍以Redundancy原因同时删除B/C。连通分量只保留一个代表，未保证每个被删块都有直接相似的保留替代。相关18项原回归仍全通过，新增反例exit1。见[自检证据](experiments/2026-10-23-full-query/self-audit-20261024/result.json)和[反例源码](experiments/2026-10-23-full-query/self-audit-20261024/transitive-substitute.cpp)。产品算法至今未改；必须修复并加回归，不能仅用网页通过作为该缺陷已解决的证据。

### 2026-10-23 第 3、4 点进度与远端恢复

**版本边界：前一 revision 的完整输入 Query 已通过真实 GPU 采集与小输出测试；最新 r3 已构建，三轮 no-thinking 探针确认输入全覆盖和生成采样，但没有实际 GC。**前一 revision 的输入通道收集本轮用户问题的全部 token、每 32 个形成一个分段、保留并评分全部分段；但生成侧慢/快窗口的捕获仍受 thinking armed/range 限制，不能据此证明答案阶段或 no-thinking 期间持续更新。服务端单独保存 GC 完整问题范围，避免被检索的默认 512-token 上限截断。后台最多每次特征读取计算 8 段，算完前不准排序，按问题和块版本缓存；不完整采集或超预算时停止该轮 redundancy 评分，不静默退回后缀。该 revision 的相关 CPU CTest **18/18** 通过；远端恢复后，在新目录 `/tmp/kvmem-full-query-20261023` 完成 CUDA adapter、服务和当时性能入口的编译，三个相关无模型测试通过。三轮 IQ3 模型探针的完整问题分别为 **1813/1813/1812 tokens、各 57 段**，头中尾共 9 项输出约束全部正确，后两轮前缀命中，未发生采集错误；观察到 4 次提交及 4 次 CPU payload 提前释放，最终槽位回收计数为 0。这些计数重叠，不能相加。证据、合成开销和局限详见 [完整 Query 验证记录](experiments/2026-10-23-full-query/README.md)。全位置覆盖仍使用段内均值，不等于语义无损；普通模型 attention 也能帮助完成小输出探针，因此不能用它证明 GC 质量。同一 revision 的八轮零修复游戏完成 3 轮后，在第 4 轮主动结束：真实 Chrome 已确认第 2 轮的 `physics.js` 斜向移动失败，而该轮之前尚无 GC 删除，后续任务也不会改此文件。因此零修复设置下已不能全部通过，无需继续消耗 GPU。第三轮已观察到 170 次 CPU payload 提前释放、最大候选池 53，GC 线程无故障；这不是完整 29 项验收。失败证据见 [单项浏览器报告](experiments/2026-10-23-full-query/game-r1-early-stop/report.json)。按先质量后性能的顺序，**正式 p95/吞吐测试未启动**。

最新版本为 **`full-query-all-decode-r3`**，25 项 [源码指纹](experiments/2026-10-23-full-query/source-manifest-all-decode-r3.json) 已保存；r2 只打包上传，没有构建或运行，已被 r3 替代。adapter 已移除生成 Query 捕获的 thinking armed/range 限制，**thinking 已 armed 时，回收候选仍限定 thinking 块**。未 armed 的非 thinking 模式保留旧策略；没有显式 candidate cap 时，不承诺 56 块上限。同时修复了答案每个 token 重复通知 closed、使 `flush_observation()` 高频执行的问题：`ThinkingSpanTracker::take_update()` 仅在边界或状态变化时发布，服务端 6 处路径统一使用；同边界 open→closed、空区域、新 thinking 和新问题重置均有覆盖。旧循环回归先失败，修复后 256 个答案 tokens 只增加 8 个慢窗口样本。

本地证据为事件去重前全回归 **18/18**、去重后受影响测试 **2/2**、Python **27/27（日志覆盖 5、长 Query probe 11、性能解析器 11）**，服务端单文件语法编译通过；详见 [r3 验证记录](experiments/2026-10-23-full-query/README.md#最新-r3-的测试归属)。r3 随后在远端完成全部 CUDA 目标构建和原生无模型测试 **3/3**，25 项源码 hash 一致、维护补丁 0014 已核对；新服务 hash 为 `824f31ba118e6baeae288b3a8e4ffe663246629958a6446cdf2646cccfd466b3`。隔离目录无 `.git`，上游以原仓库 pin 和 graph 文件 hash 核对。r3 三轮 no-thinking 探针已通过独立采集/输出检查，实际无 GC；详见下段。扩展到全部 decode 增加最多 4 层 Query D2H，两槽 harvest 忙时会等待，原捕获路径含同步，不能称为零阻塞；本地 `llama.cpp` 尚未应用 0014，CPU 测试不能替代 GPU 捕获验证。上述旧 r1 证据不能证明 r3 通过。

两轮 **off movement pilot r1 已结束且未完成任务**，使用旧二进制 `0799b3f6...`，不是 r3。第一轮 stop、生成 1,363 tokens；第二轮 length、耗尽 16,384 tokens、耗时 1,034.745 秒，最终 content 为空，只有 53,334 字符 reasoning，未生成 `physics.js`。四项提前移动检查因输入不完整未执行，报告为 **0/0，不是 0/4**，driver exit 1。原 fixture 不变，独立新 fixture 只澄清 turn 2 的原要求；此次 GC=off，不能归为 GC 质量失败、不能证明新 fixture 已奏效，也不能据此反复改 redundancy 算法。精简证据见 [pilot 审计摘要](experiments/2026-10-23-full-query/movement-baseline-pilot-r1/audit-summary.json)，原始 reasoning 留在 tmp。完整质量未通过，仍按先质量后正式性能执行。

r3 的三轮 **no-thinking** 完整 Query 探针已归档：三轮 stop、生成 52/52/51 tokens、无 reasoning，问题 1813/1813/1812 tokens 各 57 段、invalid 0，9 个头/中/尾输出字段全对，续轮前缀命中 1918/3796。未 armed thinking 时，生成慢采样累计 1/2/3、快采样 45/90/134，确认非 thinking 期间仍能采集；不是逐 decode 的时延或新鲜度证明。实际 GC 提议/提交/释放均为 0，通用 driver 因缺回收活动 exit 1，独立采集/输出 checker exit 0；这不是 GC 质量或性能验收。证据见 [no-thinking 审计摘要](experiments/2026-10-23-full-query/probe-no-thinking-r3/audit-summary.json)。

`full-query-path-gate-r3` 原生合成诊断也已通过：off/shadow/delete 各 1 warmup＋1 measured，共 **6 请求**，严格 full-query parser gate 通过。每次均 6144 输入 tokens、192 段、18,911,232 bytes 逻辑问题表示、invalid 0；256 decode 新增 slow 8/fast 249，最大候选池 56，worker/policy 无失败，off 无 worker，shadow 无释放。两个 delete 各提前释放 48 个 CPU payload、均发生于 3 个 step，首次 step 为 129/111，最终槽位回收仍为 0。全部输出 token 一致、GPU 窗口稳定；见 [路径检查报告](experiments/2026-10-23-full-query/path-gate-r3/path-gate-check.json)。**仅为功能诊断，质量与正式性能标志仍为 false**；单轮时延不证明 p95 达标，没有 RSS 采样，也不能把逻辑表示大小当成进程内存。本轮模型运行均已结束，下一步仍为完整原生 thinking 任务的 baseline/GC 质量对照，然后正式性能；旧非 thinking 29/29 只保留为历史对照。

长输入首次评分可能跨多个后台轮询，因此本地还补了排序前的最新生成 Query 刷新阶段：问题分数使用缓存，只重新计算近期生成观察的重要性，完成全部候选更新后才选代表块/排序。对应测试先复现了“长轮次使用旧生成关注点”的失败；不能只依赖采样计数上涨证明这条路径有效。

本次阶段性实现（基底 HEAD `660e46ae4a1a2079e771b8c43be5ceb03c4d75b5`）加入了 thinking-only 的有界候选池：32-token 块下默认 52 块核心池，允许最多 4 个已记住的替代参照。必须保留块计入核心池；较旧 thinking 块的滑动窗口会到达尾部才回绕。池内全对比较，不构造 N×N 矩阵。生成期间同时维护 32-token 的 Query 观察窗、8-token 快速窗和最近 4 个观察窗口的均值，与完整输入问题的分段评分一起取最大值；生成采样被非阻塞锁跳过后，各层窗口可在后续 token 重新同步。输入问题采集不允许因该锁竞争丢批。跨越 thinking 边界的块不参加 thinking-only 回收。候选池仍是有界近似，不能声称找到了全部历史块的全局最优副本；段内均值、层抽样和有限生成观察窗也不能保证语义无损或预测未来需求。

以下远端证据来自**本次全输入 Query 改动之前**：当时本地相关 CTest **17/17** 通过，隔离远端 CUDA 服务端完成编译，`historical_kv_redundancy_test` 与 `thinking_span_tracker_test` 通过。最后一次八轮验收使用源码目录 `/tmp/kvmem-thinking-v2-20261023`、二进制 `build-cuda-v2/bin/llama-kvmem-server`，输出目录 `/tmp/kvmem-thinking-v2-20261023/redundancy-gc/redundancy-thinking-final-r3`。采用本文件下文的 IQ3 GGUF、原生 thinking、32 tokens/block、36K/16K/1K/recent 0、q8_0 KV、CPU 容量 1024、Ready 352/384、每轮生成上限 16K、**0 自动修复**。

SSH/Tailscale 曾在第七轮期间断开，本地控制器当时只确认前 **6/8** 轮；2026-10-23 主机恢复后，已从远端保存的产物确认实际 **8/8** 轮均 HTTP 200、`finish_reason=stop`、无截断，所有续轮有前缀缓存命中，**0 自动修复**。最终累计日志为 `committed=1007`、`reclaimed=843`、`host_released_early=943`、`policy_failures=worker_failures=0`；这些是重叠的累计量，不能相加。`max_round_features=54`，在 52+4 上界内；`max_observation_samples=790`、`max_fast_samples=28499`。恢复后使用原浏览器检查器检查未修改的生成代码，结果为 **28/29**：唯一失败是 `diagonal movement normalized`（斜向移动未正确归一化）；六个应用文件完整，浏览器异常与非预期网络请求均为 0，初始参数保留 9/9。生成的 `physics.js` 虽归一化了方向，距离仍乘原始向量长度，导致斜向速度多乘 √2。此结果说明旧实验完整生成了应用，但未通过全部功能检查；仅凭这一失败不能归因于 GC。**这是旧输入 Query 尾窗源码的质量结果，不能覆盖新的 full-query 实现；新路径的真实模型质量和 p95/吞吐仍需独立验收。**

离线复查又修复了非 thinking 分支的同类预算漏洞：开启 `KVMEM_HISTORICAL_GC_MAX_CANDIDATES` 时，过多的 mandatory 参照原先会超过 N，并挤占可回收候选的位置。现在两种有界模式都把 mandatory 参照纳入核心预算，过多时按年龄分段采样并为可回收块保留名额。新增用例以 512 个块、128 个 mandatory 块和 N=64 复现旧失败，再验证只读取 64 个特征、执行 64×63 次比较且不回收 mandatory 块；本地相关 **17/17** 重跑通过。**恢复出的上述八轮结果属于此修复之前的远端快照，也不包含后续 full-query 改动**；它们不能替代当前 `/tmp/kvmem-full-query-20261023` 独立构建和验收证据。

旧实验恢复产物保存于本地新目录 `tmp/thinking-v2-20261023/redundancy-thinking-final-r3/recovered-20261023/`，未覆盖之前的断连记录。其中 `workload.json`、`browser-results.json`、`summary.json`、`checkpoint-0.json` 直接从远端复制；检查器输入 `results.json` 是 `browser-results.json` 的原样副本。已执行 `node scripts/check_browser_fps.mjs <检查目录> <Chrome 路径>`，输出 `browser-checks.json`、`app.png` 和提取的 `app/`；`recovery-summary.json` 保存来源、文件 SHA-256、版本边界和失败说明。远端原始 `summary.json` 的 `browser_all_passed=null` 表示断连前未收到浏览器反馈，保留原值；恢复后的 **28/29** 结论以本地 `browser-checks.json` 为准。不要将断开的本地控制器退出码当作模型生成失败，也不要把本次产物恢复当作旧 KV/GC 内存状态恢复或新源码重跑。

## 要测的对象与判定顺序

这里的 GC 回收 **CPU historical KV 块的数据/槽位**，不负责选择 GPU 工作集。管理器执行生命周期与安全校验，`lru` / `redundancy` 策略提出候选。`off` 不执行 GC，`shadow` 运行策略但不删除，`delete` 才尝试真实回收。三者回答不同问题，不能把 HTTP 成功、网页正确、CPU 释放和 decode 开销混为一项“通过”。

按顺序验收：①无模型单测和服务生命周期；②真实模型完成任务、前缀复用、实际 CPU 释放及网页功能；③固定工作量的 decode p95/吞吐；④若要称为长期稳定，再做多次回收后的长任务和内存上界。质量失败时先看参数、输出是否截断、模板与缓存命中、模型是否真的完成任务，再分析策略；不要先用性能分数代替正确性。

## 固定实验环境与配置

- 远程 Linux/CUDA 主机：`yifan@100.118.87.22`。当前完整 Query 实验使用 `/tmp/kvmem-full-query-20261023`，旧候选池实验使用 `/tmp/kvmem-candidate-budget-20261023`；二者不是同一源码快照。原用户仓库 `/home/yifan/kvmem-llama.cpp` 不一定与实验目录同一提交，不要直接覆盖。每次记录源码 manifest、二进制 hash、显卡型号、驱动及模型 hash；目录带 Git 元数据时另记 `git rev-parse HEAD` 和 `git status`，没有 `.git` 时不能冒用原仓库提交来证明实验目录的内容。
- 已使用的 GGUF：`/home/yifan/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`。文件名含 `mtp`；历史测试仍按仓库支持的常规推理路径运行，接手时以实际启动日志核对混合模型接入和所用配置。
- 导师配置预设 `mentor-27b`：原生模型聊天模板、`q8_0` KV、32 tokens/block、GPU KVMem budget 36,864 tokens、generation reserve 16,384、每轮最多生成 16,384、sink 1,024、recent 0、driver默认context 65,536（当前r5显式使用262,144）、GPU layers 99、seed 42。旧实验及 driver 默认 profile `greedy` 使用 temperature 0；当前 r4 原生 thinking 质量对照显式使用 `--sampling-profile server-defaults`，实测 temperature 1、top_p 0.95、top_k 20，不能混成同一配置。GC 组 CPU historical capacity 1,024 **块**，Ready 空槽 low/high 为 352/384；旧r4 off质量参考为4096块，当前r5为8192块；原因及性能比较限制见文末。low/high 是 CPU 空位水位，不是 GPU token 预算。
- 真实任务是 `scripts/fixtures/browser-fps-feedback-workload.json` 定义的八轮网页 FPS 代码生成，再用 Chromium 执行 `scripts/check_browser_fps.mjs` 的 29 项功能检查。测试客户端保留完整历史，每组延续**自己**生成的回复；这不是 DeepSWE/DeepSeekHarness，也没有配置 agent 工具调用或客户端上下文压缩。`--replay` 是固定 HTTP 请求的另一种实验，不能当作端到端独立生成的质量结果。
- 本地控制端需要 Python、Node、Playwright、Chromium、`ssh`/`scp`；远端需要对应源码构建的 CUDA 可执行文件和 GGUF。`KVMEM_PLAYWRIGHT_MODULE` 可指向现成 Playwright 包。避免同时跑多个占用同一 GPU 的实验；各组使用独立输出目录，保留失败运行。

## 第一层：无模型生命周期与策略测试

Windows 本地曾使用 MinGW/Ninja 的 CPU-only 构建。以下是**新目录**的复现命令（PowerShell）；若工具不在该路径，用本机实际 `cmake`/`ctest`/编译器替换：

```powershell
$env:PATH = 'C:\mingw\mingw64\bin;' + $env:PATH
& 'C:\mingw\mingw64\bin\cmake.exe' -S . -B build-gc-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_ENABLE_NVME=OFF
& 'C:\mingw\mingw64\bin\cmake.exe' --build build-gc-host -j 2
& 'C:\mingw\mingw64\bin\ctest.exe' --test-dir build-gc-host -R '^(historical_kv_|thinking_span_tracker_test|raw_kv_gc_release_test)' --output-on-failure
```

这个子集覆盖策略选块、必须保留块、回收命令的 handle/version 安全校验、CPU payload 释放、capture/续填、thinking 边界识别、仓库与原始存储等；具体测试名以 `kvmem/CMakeLists.txt` 为准。2026-10-23 首次 full-query revision 新增完整 Query 覆盖测试后相关 **18/18** 通过，其隔离远端另行通过 redundancy、query coverage、thinking tracker 三项无模型测试；这些是旧 revision 证据。最新 all-decode 的 18/18（事件去重前）及 r3 的 2/2（去重后）归属见上文，不把三次测试混写成最新版本的远端验收。这仍只是单元和集成层证据，源码再修改后必须重跑。此前更完整的模型无关回归，在 Windows 沙箱把 `TEMP`/`TMP` 指向工作区可写目录后 **26/27** 通过；剩余 `kvmem-session-cache-lifecycle-test` 的符号链接子项因 Windows 缺少创建链接权限而失败，与第 3、4 点的策略代码无关。该全量结果也不是新代码的全量验收。可在具备符号链接权限的主机上复测该项。

远端 CUDA 构建示例（用于新隔离副本，勿覆盖正在运行的源码或二进制；须根据实际 CUDA 路径/架构调整）：

从项目正常 checkout 构建时，先运行 `bash "$ROOT/scripts/apply-patches.sh"`，应用固定 upstream pin 对应的维护补丁。Linux 入口和 `scripts/windows/build.ps1` 都包含 `0014-historical-redundancy-capture.patch`；它使单 token decode 也能注册 Query 捕获。只编译本仓库 adapter、却没有重放该补丁，会漏掉生成采集。已有实验提取目录应先核对来源与补丁状态，不能仅凭目录名认定已应用。可以只读检查：

```sh
git -C "$ROOT/llama.cpp" apply --reverse --check \
  "$ROOT/patches/0014-historical-redundancy-capture.patch"
```

这项反向检查通过只证明该补丁存在，不代替全部源码指纹或模型路径测试。随后构建并运行三个相关无模型测试：

```sh
cmake -S "$ROOT" -B "$ROOT/build-cuda-explicit" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$ROOT/build-cuda-explicit" \
  --target historical_kv_redundancy_test historical_kv_query_coverage_test \
  thinking_span_tracker_test kvmem-historical-access-test llama-kvmem-server -j4
ctest --test-dir "$ROOT/build-cuda-explicit" \
  -R '^(historical_kv_redundancy_test|historical_kv_query_coverage_test|thinking_span_tracker_test)$' \
  --output-on-failure
```

`$ROOT` 是远端**隔离源码目录**，不是模型目录。另有 `scripts/test_server_historical_gc.py` 用真实模型启动私有 loopback 服务，检查无效配置早拒绝、前缀复用、reset/取消、容量压力等生命周期；用 `--server`、`--model`、`--output`、`--policy redundancy --mode delete`，针对 admission 另跑 `--admission-pressure` / `--admission-blocked`。其固定小容量和输出语法与导师的长任务配置不同，不能拿它证明网页质量或 p95。

## 第二层：真实模型质量和回收

最省事的完整入口是本地控制器 `scripts/run_browser_fps_feedback.py`。它通过 SSH 启动远端 `scripts/test_server_historical_workload.py`，待八轮生成完成，拉取输出并在本地 Chromium 做 29 项检查。设置 `--max-repairs 0` 表示**零自动修复**；即使检查失败也保存报告。例子（本机 PowerShell，`$Root`/`$Model`/`$Server` 为远端绝对路径）：

```powershell
$Root = '/tmp/kvmem-full-query-20261023'
$Model = '/home/yifan/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
$Server = "$Root/build-cuda-explicit/bin/llama-kvmem-server"
python scripts/run_browser_fps_feedback.py `
  --host yifan@100.118.87.22 --remote-root $Root --server-bin $Server `
  --model $Model --name redundancy-full-query-new-run --output tmp/redundancy-full-query-new-run `
  --chromium 'C:\Program Files\Google\Chrome\Application\chrome.exe' `
  --policy redundancy --mode delete --capacity 1024 --low 352 --high 384 `
  --max-repairs 0 --enable-thinking --sampling-profile server-defaults --context-size 262144
```

确认输出目录是**新路径**，且远端 `"$Root/redundancy-gc"` 父目录存在。脚本中的 `--dry-run` 只保存远端命令，不启动模型，不能算测试。`--server-bin`、`--enable-thinking` 是当前工作树脚本支持的参数；若接手时使用较旧提交，请检查 `python scripts/run_browser_fps_feedback.py --help`。上述原生 thinking 任务需确认 `summary.json` 的 `thinking_sampling_seen=true`、`thinking_pool_armed_seen=true`、慢/快 Query 样本非零；完整的第 4 点还要独立核验 no-thinking/答案阶段的采样，不能要求这些运行也 armed thinking。第 3 点再检查有实际 GC 时 `max_round_features<=56`。比较组至少区分 `lru/off`、`lru/delete`、`redundancy/off`、`redundancy/delete`，保持同一个模型、任务、模板、种子和预算；`off` 无需 low/high。每组使用不同 `--name` 与 `--output`，不要把一次运行的输出喂给另一组。

每组检查以下证据：

1. `remote/summary.json`：`completed_without_truncation=8`、`all_continuations_reused_history=true`、`repair_requests=0`、无 503/截断；后续请求 `prompt_cache_hit_tokens` 均非 0。
2. `inspection-0/browser-checks.json`：实际 29/29；还要确认八轮文件完整、浏览器无异常。HTTP 200 或生成了六个文件都不等于通过此项。
3. `remote/server.log` 和 `gc_finish_samples`：`policy_failures=worker_failures=0`，`delete` 有 `committed>0`，并且 `host_released_early>0` 或 `reclaimed>0` 才证明实际 CPU 资源释放。`shadow` 只需证明策略运行，不应出现真实删除；`off` 不要求 GC 活动。完成边界样本可能重复累计，**不要跨请求求和**；`committed`、提前释放与完整槽回收也不能直接相加。
4. 保存 `controller-command.json`、`driver.log`、`remote/experiment-config.json`、`remote/command.json`、请求/回复、浏览器报告与截图，确保别人能查出所用模型、配置和代码。输出内容与服务错误需要人工复核，不以脚本退出码作为唯一证据。

2026-10-22 的 `redundancy/delete` 352/384 两次复测均 **8/8、29/29、0 修复**，有前缀复用和 CPU 释放；但回收首次发生在**第 8 轮末**，不能证明多次回收后的持续质量。原始摘要见 `docs/experiments/2026-10-22-watermark-retest/quality-summary.json`。这些是旧代码/配置证据，不自动覆盖下述待验证改动。

## 第三层：decode 效率基准

在远端同一模型、同一构建的 `kvmem-historical-access-test` 中运行：

```sh
"$ROOT/build-cuda/bin/kvmem-historical-access-test" "$MODEL" \
  --gc-full-query-bench 5 > benchmark.log 2>&1
python3 scripts/summarize_historical_gc_bench.py benchmark.log --output analysis --require-full-query
```

这里 `$ROOT=/tmp/kvmem-full-query-20261023`，`$MODEL` 指上文 GGUF；统计脚本要用与当前构建对应的源码版本。一次命令在同一个模型/context 内对 `off`、`shadow`、`delete` 各预热一次，并各测 5 轮；每轮固定合成输入 6,144 tokens、**256 个单步 decode**，仍预约 16,384 输出配额。单步时间包含 greedy sampling、`llama_decode`、`llama_synchronize`；这不是 HTTP/SSE 客户端的 token 间隔，也不计入 GC-enabled prefill。`analysis/summary.json` 保存配对变化与方法限制，`analysis/runs.json` 保留每轮逐步数据。统计器要求输出序列一致、GPU 窗口稳定、worker/policy 无故障，并要求 `delete` 在 decode 期间真实回收。

当前 r3 入口只把合成输入 `[0,6144)` 标为 thinking，**256 个生成位置不标 thinking，off 还明确 disarm thinking policy**；shadow/delete 的候选仍只来自 thinking 输入块。所有模式均须验证完整输入 6,144 tokens/192 段，并要求慢/快生成 Query 计数在这 256 步后增加；这样才不会把“只在 thinking 范围内采样”误当成全部 decode 的观测证据。最多 56 块候选池和真实评分仍须验证，合成标记不证明模型真的输出了 thinking 或完成了真实任务。`off` 同样收集 Query，用于隔离后台评分/回收的增量成本。当前解析器本地 **11/11** 通过，r3 已构建但尚无正式 GPU 性能验收结果；r2 从未构建/运行，旧 r1 构建日志不能覆盖当前版本。必须看到 `full_query_validation.status=passed`，不能只看性能数字。旧 `--gc-mentor-redundancy-bench` 未接通这些新路径，保留用于历史复现，不能作为第 3、4 点的新验收；`--require-full-query` 会拒绝旧日志。

导师沟通过的**效率目标**是 `delete` 相对同轮 `off` 的 p95 延迟增加不超过 5%、吞吐下降不超过 3%。比较每轮配对变化及最差轮，不能只拿各模式中位数相除；实验仍要报告 p50/p95/p99、吞吐、真实 CPU 释放量。2026-10-22 有效的五轮结果：吞吐五轮均未超过 3% 下降，但 p95 有 **1/5 轮 +10.10%**，所以不能宣称效率目标稳定通过。见 `docs/experiments/2026-10-22-watermark-retest/native-acceptance.json`、`native-summary.json` 和 `native-runs.json`。`off` 仍采集历史/冗余特征，只隔离同一路径上开启 GC 的增量开销；不能把它叫作原生 llama.cpp 基线。

## 当前实验分支与必须重新验证的部分

- `docs/experiments/2026-10-23-candidate-budget/README.md` 是 **opt-in N=64 候选池**试验：当时用 `KVMEM_HISTORICAL_GC_MAX_CANDIDATES=64` 限制 redundancy 候选 CPU 块，在池内全对比较；不设置该变量作为当时的默认对照。它不改变 GPU/CPU 容量，也不等于论文的 token 级算法。复跑时可分别用浏览器控制器的 `--candidate-blocks 64` / 不传此项；原生 benchmark 则分别在启动进程前设置 / 不设置同名环境变量。每次另建输出目录。已做 16/16 本地单测、远端单测、各 1 轮的原生 benchmark 和一次 8/8、29/29 网页任务。`n64-summary.json` / `default-summary.json` 的绝对速度不可直接比较：两次运行连 `off` 基线都明显漂移；单轮不足以宣称 p95 改善。N=64 的 GC 也只在最后一轮首次生效。当前工作树还有后续策略改动，不能直接假定其默认行为等于当时的对照。
- 截至本文件写入时，`git status --short` 还显示策略、adapter、server 和控制脚本的**未提交修改**。历史模型证据必须按各自 manifest/hash 归属；当前 r3 已有构建、no-thinking 采集探针和原生合成功能诊断证据，但没有完整原生 thinking 任务质量与正式性能结论。接手智能体先确认最终代码与已测 r3 是否一致，有新改动则重新编译并运行相关测试，再做相同模型和任务的质量/效率对照。不要覆盖他人未提交修改，也不要把工作树实验误称为远端分支正式结果。
- 接下来的关键区分性负载应让历史超过 GPU 工作集，**在首次真实 CPU 回收之后继续多轮生成**，观察质量、前缀复用、回收进展、CPU 内存上界和长尾。对 N 做少量预先设定的取值，并把同一任务、同一预算下的多轮配对结果完整保留。若测试 thinking-only 策略，额外核对 thinking 标记/token 边界、块跨界保护、最新 Query 发布时机，以及这些机制是否给前台 decode 加入额外开销。
- DeepSWE/DeepSeekHarness 尚未配置，用户明确暂缓；不要写成已完成的真实 agent 验收。性能样本没有锁定 GPU 频率/功率，短样本不能证明长期 p95 保证。质量/效率报告中需分别写“已观测”“未观测”“不能由本实验推出”。

相关入口：`docs/background-gc-mentor-27b.md`、`docs/background-gc-parameter-retest.md`、`docs/background-gc-browser-fps.md`、`scripts/test_server_historical_workload.py`、`scripts/run_browser_fps_feedback.py`、`scripts/summarize_historical_gc_bench.py` 和 `tests/historical-access-test.cpp`。遇到命令与文件不一致，以**当前源码的 `--help`、构建配置和运行日志**为准，并在报告中解释差异。

## 新采样基线的版本边界

当前在 r3 产品二进制上核验 `--sampling-profile server-defaults`：旧 driver 强制 temperature=0，新可选 profile 省略采样覆盖项、保留 seed=42，实际 server trace 为 thinking=1/temperature=1/top_p=0.95/top_k=20。默认仍为 greedy，旧记录不变。两轮 off pilot `movement-native-sampling-pilot-r4` 使用原 fixture、原 KV 参数与 16K 上限，未设置 reasoning budget；两轮正常 stop（2433/9925 tokens），原四项移动检查 **4/4 通过**。这不是完整 29 项或 GC 质量结论。输入在 `/tmp/kvmem-full-query-20261023/quality-sampling-r4-inputs`，不能把本地两份已变动的测试脚本继续称为 r3 manifest 的原样快照。详见 full-query README 的最新小节。

`off-thinking-native-full-r4` 已结束：前6轮HTTP200/stop、无截断，续轮均命中前缀；第6轮累计52,555 tokens，第7轮实际HTTP400 `prompt + max_tokens exceeds n_ctx`，第8轮未执行。控制器和driver均exit1，模型已退出，原始目录和配置保留。原浏览器检查器在未修改的六轮网页上得到 **27/29**；两项失败是八轮未全部完成、未执行第八轮要求新增的`round`字段，其他功能检查均通过。此结果不能冒充完整任务或GC质量通过。详见[r4审计](experiments/2026-10-23-full-query/off-context-limit-r4/audit-summary.json)与[浏览器报告](experiments/2026-10-23-full-query/off-context-limit-r4/browser-checks.json)。原摘要只按特定503统计`refused_requests`，必须结合逐请求记录识别这次400。

`off-thinking-native-262k-r5` 使用逻辑上下文262144（模型训练上限）、CPU capacity8192（覆盖完整逻辑上下文），GC组使用1024/352/384；两组GPU budget36K、gen reserve/每轮最大输出16K、sink1K、recent0、32-token块及原任务/采样参数不变，没有增加reasoning budget。r3核心源码及服务hash、r4测试脚本hash已核对，只有运行配置改变。GPU工作池仍为53248 cells/1664 slots/1,853,882,368 bytes，与r4相同，见[off启动记录](experiments/2026-10-23-full-query/quality-context-r5-launch.json)。

r5 off已terminal，八轮均HTTP200/stop（2433/9925/15634/9380/9440/4659/13913/1371 tokens）。第8轮cache hit=0，reset后重算66670-token输入，prefill383.399秒，整轮512.001秒。driver/controller exit1且未进入自动浏览器反馈；随后独立使用原检查器检查原样网页，**29/29、零修复**。原summary的`browser_all_passed=null`没有修改。详见[off审计](experiments/2026-10-23-full-query/off-prefix-reset-r5/audit-summary.json)和[原检查报告](experiments/2026-10-23-full-query/off-prefix-reset-r5/browser-checks.json)。前缀失配根因尚未确证；原生模板会trim，重新分词也可能导致token前缀不同。不得据此宣称全程缓存连续或GC质量通过。

**`redundancy-thinking-native-262k-r5` 已结束，会话16266不再轮询。** 当时确认off模型进程退出、GPU空闲并核对r3二进制后启动；实际配置和GPU池见[GC启动记录](experiments/2026-10-23-full-query/quality-gc-r5-launch.json)。本地`tmp/full-query-20261023/redundancy-thinking-native-262k-r5/`，远端同名目录位于`/tmp/kvmem-full-query-20261023/redundancy-gc/`。旧off会话22708也已结束。重跑GC仍要求八轮正常stop、原29/29、零修复、实际CPU释放后持续多轮生成，以及全部续轮复用历史；若发生reset不得放宽检查冒充通过。质量通过后才启动正式性能测试，容量不同的off质量参考不能直接判定5%/3%的性能阈值。

GC组当时前四轮正常stop，2433/9925/15648/15244 tokens，续轮cache hit2888/12926/28729；第3轮开始实际回收，第4轮结束累计committed775、CPU payload提前释放650、最终槽位回收391（阶段计数不相加），worker/policy failure均0。完整问题100 tokens/4段、invalid0，观察累计1355/43226，候选池最大56。第5轮曾reset0/cache hit44084，继续沿用回收后的历史。第4轮结束ready=0，不能据此宣称容量验收通过。此段为[第4轮检查点](experiments/2026-10-23-full-query/quality-gc-r5-round4.json)；[第3轮检查点](experiments/2026-10-23-full-query/quality-gc-r5-round3.json)保留。最终第八轮HTTP503及27/29结果见文件开头的终态审计；完整质量及性能尚未验收。

等待期间的词表探针没有改产品代码：用同一原生模板重放八个请求，输入长度全匹配；第7轮API重建文本与canonical文本相同，但实际生成13913 tokens、重新分词13912，前缀总数66571/66570。原始采样IDs/解析前字节未记录，尚不能区分非canonical token切分与parser消费空白。详见[无推理探针审计](experiments/2026-10-23-full-query/off-prefix-reset-r5/prefix-vocab-probe-audit.json)。GC组若也失配，仍应失败并补原始证据，不能放宽检查。

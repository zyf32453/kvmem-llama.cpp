# Redundancy GC 参数复测（2026-10-22）

## 导师检查入口

本记录对应分支 `codex/background-gc-redundancy`，建议结合以下代码检查：

- 策略与回收分离：`kvmem/src/host/historical_kv_redundancy.cpp`、`historical_kv_policy_lru.cpp` 和 `historical_kv_gc.cpp`。
- 混合模型接入：`src/adapter/llama-memory-kvmem.cpp`、`llama-memory-kvmem-hybrid.cpp`，以及 `patches/0015`–`0017`；补丁由 `scripts/apply-patches.sh` 应用。
- 32-token 尾块续填：`kvmem/src/host/historical_kv_capture_session.cpp`、`historical_kv_raw_store.cpp`、`historical_kv_repository.cpp`，对应 `kvmem/tests/historical_kv_capture_session_test.cpp`。
- 功能与效率复现：`scripts/run_browser_fps_feedback.py`（本次指定 `--max-repairs 0`）、`scripts/check_browser_fps.mjs`、`tests/historical-access-test.cpp`。

本次上传附带以下原始 JSON 的副本，方便不访问实验主机也能检查数据：
[功能汇总](experiments/2026-10-22-watermark-retest/quality-summary.json)、
[第 1 次浏览器检查](experiments/2026-10-22-watermark-retest/browser-checks-r1.json)、
[第 2 次浏览器检查](experiments/2026-10-22-watermark-retest/browser-checks-r2.json)、
[五轮效率验收](experiments/2026-10-22-watermark-retest/native-acceptance.json)、
[基准汇总](experiments/2026-10-22-watermark-retest/native-summary.json)、
[逐步计时记录](experiments/2026-10-22-watermark-retest/native-runs.json)。
JSON 内的原实验目录字段保留不变；模型权重、构建产物及完整服务器日志不进入 Git。

上传前重新编译本地 CPU historical 测试子集，15/15 通过；三个 Python 实验入口、浏览器检查脚本和两个新 JSON fixture 通过语法检查。
这些检查与下文已完成的远程 CUDA / IQ3 测试分开记录，上传时未重新运行大模型。

## 顺序与通过条件

根据导师反馈，先调整参数验证网页任务正确性，再进行 p95 与吞吐验收。
不修改模型输出、任务要求、29 项检查或 redundancy 评分公式。

- 固定 Qwen3.8-27B GSQ-RCO IQ3_S、原生模板、q8_0 KV、32 tokens/block。
- 固定 GPU budget 36864、generation reserve / max_tokens 16384、sink 1024、recent 0、context 65536。
- 固定 seed 42、temperature 0、原八轮 browser-fps-v2 任务；不启用自动修复。
- 原主对照：CPU 容量 1024，空槽 low/high 384/448；redundancy 两次均为 28/29。
- 第一候选：保持容量 1024，将 low/high 降至 352/384，减轻并延后回收压力。
- 若第一候选仍失败，再逐项诊断；不把未发生回收的运行算作 GC 质量通过。
- 质量通过条件：同一配置至少两次完整 8/8 正常完成、历史前缀复用、29/29 检查通过，且观察到真实 CPU 回收、policy/worker failure 为 0。
- 通过上述条件后再做固定工作量的原生逐步计时，比较 off/shadow/delete，报告 p50/p95/p99 和吞吐。p95 增加不超过 5%、吞吐下降不超过 3% 按导师沟通作为效率目标；短样本不能证明长期保证。

该任务是脚本驱动的多轮代码生成，不是 DeepSeek Harness / DeepSWE。客户端不压缩历史。
容量和水位只控制 CPU historical 池，不改变 GPU 工作集预算。
每次运行保留独立目录，失败记录不覆盖；挑选出的配置需要复测，不能只报告最好的一次。

## 结果

第一候选第 1 次：8/8 正常结束，后续请求均复用前缀，29/29 功能通过，自动修复 0 次。
第 1–7 轮末回收提交均为 0；第 8 轮末提交 54 块、提前释放 CPU payload 54 块、完整槽位 reclaim 0。
因此属于末轮发生回收的轻压力验证；当前 GPU 许可尚存时 CPU 数据可以先释放，不能把它说成完全没有回收，
也不能把这次通过泛化为多次回收后的长期质量保证。policy/worker failure 均为 0。
原始记录：`artifacts/mentor-27b-config/iq3-browser-redundancy-watermark352-384-r1/`。
第 2 次同配置复测也为 8/8、29/29、自动修复 0 次，历史前缀复用且客户端历史完整保留。
同样在第 8 轮末首次观察到回收：提交 54 块，完整 reclaim 16 块，提前释放 CPU payload 38 块，
policy/worker failure 为 0。这两个释放计数是不同生命周期观测，不能把相加规则推广到其他运行。
原始记录：同目录名将 `r1` 换为 `r2`。
机器可读汇总：`artifacts/mentor-27b-config/parameter-retest-quality-summary.json`。

两次都通过了本次八轮轻压力任务的质量条件。回收发生较晚；不能推断长任务多次回收后仍然无损。
评分公式、任务与功能检查没有修改，后续不应将 352/384 宣称为所有工作负载的最佳参数。

## 原生 decoding 性能验收

旧基准仅接收非混合 adapter、f16 KV、小 GPU 预算和 32 个生成步，不能直接用于当前 IQ3 配置。
在同一测试程序中新增 `--gc-mentor-redundancy-bench 5`，仅修改测试源码，不修改产品 GC 库：

- 相同 IQ3 模型，GPU budget 36864、generation reserve 16384、sink 1024、recent 0、q8_0、32-token 块、context 65536。
- CPU 容量 1024、水位 352/384；固定合成输入 6144 tokens，计时生成 256 步，仍预约完整 16384 输出配额。
- 一个模型/context，各模式先预热一次，再按轮换顺序测五轮 off/shadow/delete；每模式 1280 个正式单步样本。
- 每步测量 greedy sampling、llama_decode、llama_synchronize 的总时间，并核对全部历史 GPU 驻留与输出序列一致。
- GC 在 prefill 后启动，delete 必须在生成期间提交回收并实际释放 CPU 数据。
- 统计原生单步 p50/p95/p99 和生成吞吐；该口径不是客户端 SSE TBT，也不包含 HTTP/网络或 GC-enabled prefill。
- off 也保留 redundancy 特征采集路径，因此衡量同一路径内开启 GC 的增量开销，不代表整个历史管理功能相对原始 llama.cpp 的成本。

第一次性能尝试在预热阶段发现较长配置 JSON 与 stderr BEGIN 行交错，主动终止并保留 `r1` 原始日志，
不得纳入验收结果。已在配置输出后增加 fflush，再独立启动 `r2`；产品二进制 SHA-256 在仅重建测试前后相同。
有效 `r2` 完成 3 次预热与 15 次正式样本，退出码 0；统计器也通过完整性、一致性、真实释放检查。
每次正式 delete 都在生成期间提交并提前释放 64 个 CPU 历史块，4 个生成步骤观察到释放进展。
完整槽 reclaim 为 0，因为测试故意保留当前请求的 GPU 执行许可；此时 CPU payload 已真实释放。
全部输出 token 序列相同，GPU 全历史驻留不变，policy/worker failure 为 0。

### 五轮配对结果

每轮分别计算 256 步的 p95，再与同轮 off 对照；正的延迟变化表示变慢，负的吞吐变化表示变慢。
下列全部正式样本均保留，没有剔除第 4 轮。

| 轮次 | off p95 / ms | delete p95 / ms | p95 变化 | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 69.673 | 63.283 | -9.17% | -0.59% |
| 2 | 65.785 | 67.022 | +1.88% | +0.75% |
| 3 | 63.014 | 64.713 | +2.70% | -0.03% |
| 4 | 65.720 | 72.361 | +10.10% | -0.13% |
| 5 | 65.945 | 67.694 | +2.65% | +0.26% |

- 配对变化中位数：p95 **+2.65%**，吞吐 **-0.035%**。
- 吞吐五轮全部在下降不超过 3% 的目标内，最差下降 **0.59%**。
- p95 有 **1/5 轮超过 +5%**，最差 **+10.10%**；不能仅用中位数宣布稳定通过。
- 各模式五次统计量的中位数：off 16.544 token/s、p95 65.785 ms；shadow 16.536 token/s、p95 63.946 ms；delete 16.522 token/s、p95 67.022 ms。
  这些是分别取中位数，不能拿它们的比值替代上面的逐轮配对变化中位数。
- 配对 p99 变化中位数 +2.47%，范围 -12.17% 至 +7.28%。每次仅 256 步，对罕见长尾仍不充分。
- shadow 第 1 轮较 off 快约 20.22%，其余四轮变化约 -1.68% 至 +0.99%；早期速度漂移存在，不能将负延迟变化解释为策略提速。
- 全过程监控：显存峰值 13756 MiB、GPU 最高 83°C、进程 RSS 峰值约 11935 MiB；进程整体平均约 1.02 个 CPU 核。
  CPU 数值包含加载、prefill、decode、清理，不能解读成 GC 线程自身的使用率。GPU 频率/功率未锁定。

本次结论：**八轮轻压力任务两次功能全通过；原生吞吐目标满足；p95 尚未稳定满足 +5% 目标。**
仍需定位第 4 轮的尾延迟波动及其与后台调度/硬件状态的关系，不能声称所有效率要求已经通过。

有效原始记录：`artifacts/mentor-27b-config/iq3-native-redundancy-mentor352-384-r2-20261022/`。
其中 `analysis/summary.json`、`analysis/runs.json` 保留汇总与逐步计时，`acceptance.json` 保留目标及五轮对照。
第一次日志无效尝试保存在远程同名 `r1` 目录，其 `invalidated.json` 明确记录终止原因，不计入有效样本。

### 复现入口

功能复测通过 `scripts/run_browser_fps_feedback.py` 使用 `--policy redundancy --mode delete --capacity 1024 --low 352 --high 384 --max-repairs 0`；
其余参数沿用 mentor-27b 预设，完整命令保存在两次运行目录的 `controller-command.json` 中。

性能测试与统计命令：

```sh
build-cuda/bin/kvmem-historical-access-test MODEL.gguf --gc-mentor-redundancy-bench 5 > benchmark.log 2>&1
python3 scripts/summarize_historical_gc_bench.py benchmark.log --output analysis
```

本次修改范围仅包含实验参数、上述测试入口、汇总器的通用限制说明与实验记录；未调整 redundancy 评分公式或产品 GC 调度。

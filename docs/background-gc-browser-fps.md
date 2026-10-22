# 多轮浏览器 FPS：CPU historical GC 质量案例

本文保留早期单 token / 旧模型的实验过程。当前已恢复 32 tokens/block，并使用 IQ3 27B；
最新配置与结论以 [导师配置记录](background-gc-mentor-27b.md) 和 [参数复测记录](background-gc-parameter-retest.md) 为准。

这是一个固定八轮的编程案例，不是 DeepSWE 或具备文件/终端工具的自主 agent。
目标是让模型完成离线浏览器射击游戏原型，再修改它，观察旧要求保留和功能正确性。
不要求复刻商业游戏；不使用商业素材，也不把“生成了网页”当成质量通过。

## 固定任务与判分

`scripts/fixtures/browser-fps-workload.json` 定义八轮任务：状态、移动/碰撞、武器、敌人、
第一人称画面、页面集成、武器回归修改、初始状态回归修改。早期规定弹匣 7 发、备用 21 发、
换弹 1.25 秒、伤害 20、速度 3.5，以及禁止自动换弹；后续不再逐项重复这些数值。

模型只输出预先指定的六个文件，最后两轮覆盖其中两个。测试程序按输出原样组装，
不人工修复、不补写功能。原始请求、流式事件、完整回复和截断/失败都保存。
每组独立延续**该组自己的回复**；不能把基线生成的正确代码喂回所有组后宣称端到端质量无损。
`--replay` 仍可用于另一种固定输入性能对照，但不属于这个闭环案例。

`scripts/check_browser_fps.mjs` 在 Chromium 中执行生成的网页，不向网页提供 Node 权限，
拦截网络请求，只提供六个测试文件。检查初始化和旧数值、状态独立性、移动与碰撞、
射击/墙体阻挡、手动换弹与计时、敌人攻击、胜负、重开和真实键盘/鼠标事件。
同时保存浏览器错误、检查结果和截图。它不等于完整游戏可玩性或通用编程能力评测。

缺少文件或某轮截断时报告未完成，不将旧文件代替失败修改后算作通过。
HTTP 成功、浏览器功能正确、GC 实际释放和缓存复用分别记录，不能互相替代。

## 对照与有效性

主要三组分别为 `lru/off`、`lru/delete`、`redundancy/delete`，另加 `redundancy/off`
区分新增特征路径与实际回收。保持模型、任务、随机种子、
输出上限、GPU 预算、CPU 块容量、水位和聊天格式相同。比较功能检查、旧要求保留、
实际 CPU 释放、输入/生成耗时和进程内存。相同水位不保证两种后台策略在有限时间内
释放相同数量的块；保留更多历史获得的质量变化，不能直接证明选块更好。
一次任务只是案例，不证明某策略普遍更好。

必须检查后续请求的 `prompt_cache_hit_tokens`：历史被重新计算会补回已删内容。
`--require-prefix-reuse` 在截断或某次延续没有命中历史时停止；无隐式重试/reset。
还要结合日志确认同一仓库、退休/释放发生在后续请求之前，且没有另一个 GPU 工作集
淘汰过程掩盖 CPU GC 的影响。`--trace` 保存这些证据。

现有实现的短生成片段会占用整块 GPU 槽。原始 32-token 块在这个长输出负载中可能很早
耗尽物理槽，即使有效 token 数远小于 GPU token 预算。`--block-tokens 1` 仅作为排除此
干扰的**实验配置**；结果不得外推到原来的 32-token 生产配置。它也使策略评分粒度变小。

当前 Qwen 模型的原生模板会在旧 assistant 回复前移除最初的空 thinking 前缀，破坏精确
前缀复用。`--stable-qwen-template` 使用保留该前缀的文本 ChatML 测试模板，首轮格式与
原生非 thinking 输入一致。它不是通用模型模板，不支持工具调用，也不改服务器默认模板。

## 运行

在专用测试目录和空闲 GPU 上串行执行。以下配置是实验配置，不修改默认产品设置：

```sh
python3 scripts/test_server_historical_workload.py \
  --server build-cuda/bin/llama-kvmem-server --model /path/to/model.gguf \
  --workload scripts/fixtures/browser-fps-workload.json --output /path/to/new-output \
  --mode off --policy lru --capacity 8192 --low 6144 --high 6400 \
  --budget 4096 --max-tokens 1024 --context-size 16384 --block-tokens 1 \
  --seed 42 --stable-qwen-template --require-prefix-reuse --trace
```

随后分别改为 `--mode delete --policy lru` 和 `--mode delete --policy redundancy`，
每组使用新的输出目录；额外以 `--mode off --policy redundancy` 运行特征路径对照。
low/high 是 **Ready 空位水位**：这里压力从可用空位少于 6144 开始，
目标回到 6400，并非保留 6144 个已用块。Mean-K/描述缓存的额外开销仍需单独计入内存解释。

安装/指定 Playwright 后在本地检查取回的结果；浏览器可使用已有 Chromium/Chrome：

```sh
node scripts/check_browser_fps.mjs /path/to/output /path/to/chromium
```

环境变量 `KVMEM_PLAYWRIGHT_MODULE` 可指向已有 Playwright 包的绝对路径。
检查器创建 `app/`、`app.png` 和 `browser-checks.json`，拒绝覆盖已有 `app/`，保留首次检查。
可先用 `--max-turns 2` 做部分预检，汇总会明确标记不是完整任务验收。

## 2026-10-22 首次案例记录

产品代码基于 `808fe62`，使用此前的 Qwen3-30B-A3B Q4_K_M 模型与 RTX 3080 Ti Laptop
16 GiB 主机，单 GPU、线程 4、batch 64、microbatch 16、f16 KV。此次只补充测试与文档，
没有修改 GC 策略实现或默认配置，也没有配置 DeepSWE、改变主机设置。

首先保留两组失败的预检：

- 原生模板、32-token 块：两轮均达到 1024 输出上限而截断；第二轮 cached=0、reset=1。
  日志显示 GPU 槽池只有 160 槽，短生成片段触发反复 `prefill_pressure`，不能作为纯 CPU GC 对照。
- 原生模板、1-token 块：两轮分别正常输出 223/277 token，但第二轮仍 cached=0、reset=1。
  核对本地模型模板发现历史 assistant 回复会丢掉最初的空 thinking 前缀。

之后固定 1-token 块及稳定 ChatML 模板，执行完整任务。主三组均完成 8/8 请求，后续七轮
全部命中缓存；各组仓库 ID 始终为 1，没有 reset、没有 `prefill_pressure`，两种 delete
组也没有 policy/worker failure。这里的成功只指推理和生命周期流程。

- **LRU/off**：2905 个生成 token，请求总耗时 191.98 秒。最终 state.js 保留了原始 8 项
  关键数值和本轮新增 round=1，但其他模块覆盖 `window.G`，网页报 `G.createGame is not a function`。
- **LRU/delete**：3327 个生成 token，188.84 秒。最后完成边界累计退休 2778 块、
  提前释放 2778 块 CPU payload、完整槽回收 2442 个。最终状态接口也变了：弹药写成 10，
  备用 30，换弹 1.5 秒，伤害 10；页面还引用了不存在的 game.js/updateBots.js。
- **redundancy/delete**：3321 个生成 token，226.85 秒。最后完成边界累计退休 736 块、
  提前释放 720 块 CPU payload、完整槽回收 672 个。单独执行最终 state.js 时，原始 8 项
  数值及 round=1 都正确；完整网页仍因其他模块覆盖 `window.G` 而启动失败。
- **redundancy/off**（补充控制）：3311 个生成 token，220.17 秒；8/8 完成、七次延续
  命中缓存、同一仓库、没有 reset 或 GPU pressure。最终状态同样保留上述数值，但页面
  报 `G is not defined`，也未形成可运行游戏。

这些累计计数不能跨请求相加，也不是退出后清理所有资源的总数。最后完成边界的 payload
观测值，LRU/delete 为 243,695,616 字节，redundancy/delete 为 896,729,088 字节；
它们还包含当前请求的预留，且 redundancy 多出 Mean-K 等资源，不能当成纯有效 KV 字节。
三组进程 RSS 峰值都约 17.68 GiB，受模型及其他分配主导，本案例没有展示明显 RSS 峰值下降。

**不能由这些结果断言 redundancy 的选块质量优于 LRU**：它删得更少、保留得更多，
实际内存和上下文覆盖没有对齐。相同水位不等于实际达到同一个保留量；此负载下 redundancy
的回收进度明显更慢，值得后续定位。总耗时同时包含不同长度输出、不同 attention 历史和
特征计算，也不能用作纯后台 GC 开销。单次顺序运行没有控制主机性能漂移。

另外，redundancy/delete 在第二轮尚未退休任何块时，输出已经与 LRU/off 不同，
因此增加 redundancy/off 对照，不能将所有输出差异归因于退休。浏览器检查的多数失败由
同一个初始化错误连带产生，不应将检查项数量解释为多道独立任务的正确率。
补充控制确认：redundancy/off 与 delete 的前三轮回复完全相同，第三轮 delete 已发生
退休，第四轮开始回复不同。LRU/off 与 delete 也在前三轮相同、第四轮分化。
因此测到了退休后的延续效果，但原始 LRU 路径与 redundancy 特征路径不能直接假设逐字一致。

当前结论是：这类复杂任务能暴露历史保留和接入问题，但本次**没有任何一组完成可运行游戏**，
尚不能给出端到端任务成功率收益。下一步应先用有限的实际报错反馈建立能启动的无 GC 基线，
再在相近实际保留预算下比较策略；不需要为此扩展一套大型 agent/生产 GC 框架。

原始证据保存在 `artifacts/redundancy-fps/`：每组有请求、回复、启动参数、服务器日志、
进程/GPU 采样、生成文件、浏览器检查与截图；`comparison.json` 汇总流程、回收和耗时。
两个不回收组的最终检查分别在 `fps-stable-block1-off-inspection/` 和
`fps-stable-block1-redundancy-off-inspection/`；初版检查保留，没有覆盖。
检查器在最后一组发现缺失全局对象时的保护判断问题，已修复并重新检查原始输出，
没有重新生成或修改模型代码。正式执行脚本副本保存在 `test-sources/executed-workload-v2.py`；
最终提交的 runner 再调整帮助说明、报错文字及 replay 的 seed 元数据；正式案例没有使用 replay。

证据归档为 `artifacts/redundancy-fps/evidence.tar.gz`，SHA-256：
`77d1bf9f9a200d3f2e5ee2e93b72e39a540e783d60ddb5e5ec4f6d338a5b796c`。
Python/JavaScript 语法检查和 `git diff --check` 通过；浏览器验收返回失败属于如实记录的
模型任务失败，不能将其改写成测试全通过。

## 有限浏览器反馈与回收进度诊断

第二阶段先建立能运行的无 GC 基线，保持产品评分公式、权重和后台配置不变。
`browser-fps-feedback-workload.json` 保留原来的八轮功能要求和验收检查，去掉每文件
1600 字符的软限制，明确只有 state.js 初始化 G，其他模块必须扩展已有 G。
这是明确接口约定，并未提供实现代码或把最终答案回填给模型。

`run_browser_fps_feedback.py` 启动远端现有服务器，本地 Chromium 验收完整八轮输出，
最多把三次真实失败反馈给同一个模型会话，每次只替换一个文件。按状态、移动、武器、
敌人、渲染、页面集成的固定顺序选择出错模块；缺失公共函数优先修复其所属模块。
每轮浏览器报告、反馈正文和模型输出分别保存。通过即停止；达到上限、截断、前缀复用
失败或服务拒绝都算失败，没有人工修补、隐式重试或服务器重启。
修复后的成功与初始生成成功分开报告，不能把反馈后修好的数值算作无提示记忆正确。

为容纳完整任务及有限修复，新基线设逻辑上下文 16384、GPU budget 12288、生成预留与
每轮输出上限 1024、CPU historical 槽容量 16384。仍使用单 token 块及稳定文本模板，
显式指定 sink=1、recent=0；线程 4、batch 64、microbatch 16、seed=42、temperature=0、
F16 KV、全部模型层加载 GPU。该基线不与上一版不同预算的耗时直接比较。
预检尝试减少 GPU 加载层数时，服务在第一轮推理前拒绝了请求：historical capture
要求所有注意力 KV 位于单张 CUDA 卡上。失败记录保留，后续恢复受支持配置。

运行示例（控制端须有 Python、SSH/scp、Node、Playwright、Chromium）：

```sh
python scripts/run_browser_fps_feedback.py \
  --host user@host --remote-root /path/to/existing-isolated-build \
  --model /path/to/model.gguf --name fps-feedback-baseline \
  --output /path/to/new-local-evidence --node /path/to/node \
  --chromium /path/to/chromium --capacity 1024 --max-repairs 0
```

上述是 mode=off 的基线调用。当前控制器已支持 off/shadow/delete，以及 lru/redundancy，
并统一使用 mentor-27b 预设；开启 GC 还需显式提供 `--low/--high`。
早期阶段只运行 off 的限制已解除；不同配置的结果仍应分开报告。

回收进度的排查已有两类相互吻合的证据：

- 原始模型运行中 redundancy 完成 46 轮评分、提交并成功退休 736 块，恰好每轮 16 块；
  missing=0、no_query=0、skipped_queries=0、retry=0，没有策略或 worker 失败。
  累计读特征 108824 次、做相似度比较 3480896 次，许多未回收块在后续轮次重复计算。
  736 块退休中，完成边界已有 720 块提前释放 CPU payload。因此主要进度缺口发生在
  提交回收之前，不能用完整槽暂时仍受 GPU 引用保护来解释全部差距。
- 独立诊断 `kvmem/tests/historical_kv_progress_probe.cpp` 使用实际 manager、policy 和
  repository，提供始终有效的合成特征，以及无引用阻碍、可立即释放的真实 CPU 小块。
  8192 槽中放入 2048 个 Live 块，目标释放 1024 块，推进 2000 次 manual_tick：
  LRU（scan=64/batch=16）退休并释放 192 块；现有 redundancy（16/16）释放 32 块。
  仅在诊断实例改为 scan=64/batch=16 时释放 160 块；改为 16/64 时释放 128 块。
  特征和评分公式没有变化，产品参数也没有修改。

原因是 adapter 把 redundancy 的 scan 设为 16，而这个预算既限制特征处理，也限制
manager 收集整个槽目录。容量 8192 时，单次收集就要 512 个 tick；worker 每个 tick
之后再睡 5 ms。收集、评分完成后只输出最多 16 个候选，下一轮重新收集及计算。
受控诊断首次提交需要 767 个 tick，对应仅轮询等待就至少 3.83 秒；LRU 为 159 个 tick、
0.79 秒。该程序没有实际 sleep，测得的短计算时间使用八维合成特征，不能当成真实模型
的计算耗时或前台延迟；等待下界和重复工作的结构性开销则来自现行控制流程。

独立编译诊断时须链接当前服务器使用的库，避免误用构建目录内遗留的旧静态库：

```sh
g++ -O2 -std=c++17 -pthread -I kvmem/include \
  kvmem/tests/historical_kv_progress_probe.cpp -L build-cuda/kvmem -lkvmem \
  -Wl,-rpath,"$PWD/build-cuda/kvmem" -o /tmp/historical-kv-progress-probe
/tmp/historical-kv-progress-probe
```

后续优化应先考虑把目录扫描预算与昂贵的特征计算预算分开，或在保证身份、内容版本、
压力复核和代表块保护的前提下复用一轮评分结果、分批输出。不能简单无限增大每次工作量，
因为仍需验证对前台 decode 延迟的影响。本阶段只诊断，不实施这些产品改动。

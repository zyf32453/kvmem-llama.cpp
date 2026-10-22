# 导师指定的 27B / 32-token 实验配置

2026-10-22 根据用户转达的导师要求调整。前半部分保留准备过程，后半部分记录
Qwen3.8-27B 上的实际接入和验收结果。此前 Qwen3-30B-A3B、单 token 块的结果仅作历史诊断。

2026-10-22 晚间按“先参数与质量、后效率”追加复测：CPU 水位 352/384 下，redundancy 两次网页检查
均为 29/29，并在末轮发生真实 CPU 释放；五轮原生性能对照吞吐均在 3% 目标内，但 p95 有一轮增加
10.10%，尚不能宣布稳定满足 5% 目标。完整配置、范围限制与证据见 [参数复测记录](background-gc-parameter-retest.md)。

## 固定参数

本实验统一按 1K = 1024 token 解释：

- 模型：Qwen3.8-27B，采用项目文档推荐的 ISTA DASLab GSQ-RCO IQ3_S GGUF，
  文件名 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`。
- `--kvmem-block-tokens 32`。
- `--kvmem-budget 36864`。
- `--kvmem-gen-reserve 16384`。
- 每个 HTTP 请求 `max_tokens=16384`；服务默认 `-n 16384`。
- `--kvmem-sink-tokens 1024`，对应 32 个 sink 块。
- `--kvmem-recent-tokens 0`。
- `--kv-dtype q8_0`：采用项目 IQ3 / 16 GiB 配方的主 KV 精度。模型权重 IQ3 与 KV 精度是不同设置。
  原来 30B-A3B 的 f16 KV 测试不能替代该配置的验收。

选择依据为本地 `README.md` 的“Recommended settings (16 GiB)”和“IQ3 27B”段落，
以及 `scripts/windows/README.md` 的主模型链接：
[ISTA DASLab 模型卡](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)。
下载通过 [hf-mirror](https://hf-mirror.com/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) 直连，
不使用代理或 VPN。`-mtp` 文件包含额外预测头，但本次历史 GC 验收继续关闭 MTP，也不加载视觉投影器。

实验脚本补充的逻辑上下文为 `-c 65536`，以覆盖 36864 + 16384 = 53248 的 GPU KV 槽池。
这不是导师额外指定的数值，也不是 65536 token 全部常驻 GPU。完整槽池应有 1664 个 32-token 槽。
CPU historical 容量及空槽水位不等于这些 GPU token 预算，必须另外明确指定。

原 Qwen3 的固定聊天模板不直接用于新模型。新预设使用模型原生模板；需要重新检查多轮前缀
复用，不能假定模板行为与旧模型相同。现有 historical GC 入口仍是 recency；本次参数调整
没有实现 GPU retrieval 接入，也没有修改 redundancy 的评分公式。

## 实验入口

`scripts/test_server_historical_workload.py --preset mentor-27b` 设置上述参数，
拒绝与预设冲突的块大小、预算、生成上限、sink、recent 和 KV 精度参数。
生成预留现在可用 `--gen-reserve` 单独表示；未使用预设时，保持原来默认等于每轮输出上限的行为。
脚本扩大了原来的参数范围限制，并检查逻辑上下文能容纳工作集预算与生成预留。

`scripts/run_browser_fps_feedback.py` 已切换为调用该预设，不再硬编码 block=1、sink=1 或
旧模型的固定聊天模板。必须提供 CPU `--capacity`；启用回收还必须提供 `--low/--high`。
不自动从旧实验照搬这些按“块”计数的值。

两个入口均支持 `--dry-run`，只生成配置与命令，不启动服务器，不连接远程主机进行推理。
以下仅演示预设展开；路径和 CPU 容量是预览占位，不能作为已验收配置运行：

```sh
python3 scripts/test_server_historical_workload.py \
  --server /pending/llama-kvmem-server --model /pending/Qwen3.8-27B.gguf \
  --output /tmp/new-mentor-config-preview --mode off --capacity 4096 \
  --preset mentor-27b --dry-run
```

实际执行时，还会读取模型加载日志，要求元数据名称匹配 Qwen3.8-27B，并检查生效槽池的
cells、slots、block_tokens、budget、gen_reserve、sink_blocks。错误模型或预算被缩减时
终止运行，不把降级结果标记为导师配置。仅 `--dry-run` 成功不代表模型加载、容量准入或推理成功。

## 准备阶段发现的问题（现已处理）

1. **原主机没有在已检查范围内找到目标权重，现已确定下载版本。** 初次检查远程
   `/mnt/d/ollama/manifests` 清单只发现 `qwen3/30b-a3b`，
   对应之前使用的权重哈希。对 `/home/yifan`、`/mnt/d` 最多五层的文件检查只找到
   `ggml-vocab-*` 测试词表，没有发现可运行的 27B GGUF 权重。存在无权访问的目录，
   也未排除更深目录、其他磁盘、其他后缀或未登记的 blob，因此不能宣称整台主机没有此模型。
   用户随后转达导师确认“先下载 GGUF，16G 使用 IQ3”，据文档选定上述文件。
   其下载版本、完整性结果及路径见后文。
2. **混合模型 historical 接入。** 准备时 `begin_historical_capture` 因 `recr_` 拒绝 qwen35/hybrid。
   现在只托管 attention KV，recurrent state 仍由原生 hybrid 路径管理；完整变更及限制见下文。
3. **32-token 尾块续填。** 准备时逐 token decode 会形成大量只有 1 个有效 token 的槽。
   现在同一请求可续填未封口的尾块，满 32 token 或请求结束时封口，预约量按物理块数计算。
4. **硬件适配。** 16 GiB RTX 3080 Ti Laptop 已按完整导师 GPU 预算加载并完成推理，
   生效槽池 53248 cells / 1664 slots；具体内存日志见下文。

原始搜索记录：`artifacts/mentor-27b-config/remote-model-inventory.json`。
配置预览：`artifacts/mentor-27b-config/preview/experiment-config.json` 和 `command.json`。
这些证据保存在本地忽略目录，没有作为新模型质量或性能测试结果。

## 本次准备工作的验证

两个 Python 入口通过语法检查，`git diff --check` 通过。13 项配置检查通过：
预设命令展开、8 种冲突参数拒绝、原通用默认值保持、输出上限与生成预留独立传递、
输出超过预留拒绝、上下文不足拒绝。记录在 `artifacts/mentor-27b-config/config-checks.json`。
这是首次参数准备时的检查记录。导师随后确认 IQ3 后的参数与模型加载结果见下节。

用户授权停止的旧 `fps-feedback-v2-off-b` 实验已结束；先核对其服务器进程的完整命令，
再只终止该进程，保留原始结果与明确的 superseded 标记。停止后确认服务器退出、
GPU 使用约 179 MiB、利用率 0%。记录与旧实验副本在
`artifacts/mentor-27b-config/old-experiment-stop.json` 和 `superseded-old-run/`。

## 导师确认 IQ3 后的模型选择

导师已确认先保留 llama.cpp，下载 GGUF，并注意 16 GiB 显存对应的 IQ3 量化。
按项目现有 IQ3 配方，选定 `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` 的
`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，而不是任意同名 IQ3 或 NInfer 工件。

- 固定仓库版本：`d562806dbafae37109975e970aae91b43e73b440`。
- 镜像文件头提供的完整大小：12,120,016,960 字节，约 12.12 GB / 11.29 GiB。
- 期望 SHA-256：`58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f`。
- 下载位置：`/home/yifan/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/`。
- 已读取文件头确认 `general.name=Qwen3.8-27B GSQ-RCO`、`general.architecture=qwen35`、
  存在 SSM 元数据；这是混合模型路径。文件名中的 IQ3_S 表示权重量化类别，实际是非均匀混合精度。

该文件带 MTP 权重，下载它不等于开启 MTP；本次使用普通 decode，不加载额外视觉投影器。
预设主 K/V 从旧测试的 f16 改为文档配套的 q8_0，原通用脚本默认仍为 f16。
更新后 14 项配置检查通过，新增一项拒绝把导师预设误改为 f16；记录为
`artifacts/mentor-27b-config/config-checks-iq3.json`。
这些检查仍只验证参数，未替代新模型和 q8_0 historical GC 的实际验收。

### 下载与短加载检查结果

已通过 hf-mirror 直连下载完整文件，大小及 SHA-256 均与上面的期望值一致。
远程完整路径为：

```text
/home/yifan/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
```

完整性记录保存在 `artifacts/mentor-27b-config/iq3-download-verified.json`。
使用现有隔离构建的服务器，在本机 RTX 3080 Ti Laptop 16 GiB 上完成短加载检查，
启用 KVMem recency、q8_0 KV、32-token 块，B=36864、R=16384、sink=1024、recent=0，
逻辑上下文 65536、每轮输出上限 16384、MTP/视觉/历史 GC 均关闭。
日志确认生效槽池为 53248 cells / 1664 slots，没有缩减 B/R。

日志分配量分别为 CUDA 模型 10827.36 MiB、KV 1768.00 MiB、recurrent state 149.62 MiB、
CUDA compute buffer 212.39 MiB；这些是各缓冲区日志值，不是整卡峰值显存。
构建所用 nvcc 为 13.2.86，与本地 README 给出的 IQ3 正确性验证基线一致。

请求计算 12+7 与 7*8，输出 JSON 数值 19 和 56 正确，生成 14 tokens 后正常 stop。
检查程序退出码 0，自己的服务器随后退出，显卡恢复空闲。命令、模型加载日志、请求、回复和
结果保存在 `artifacts/mentor-27b-config/iq3-loading-preflight/`。

该早期检查只证明模型文件、基本推理及所列 GPU 分配能在当前主机运行；它未启用 historical
capture。下面是之后完成的混合模型 GC 和尾块续填验证。

## 混合模型与 32-token 尾块接入

历史捕获现在可识别 Qwen3.8 的 hybrid memory owner。attention 层的 KV 走原有历史目录、
回收策略和后台 worker；recurrent state 仍由 llama.cpp 的原生 hybrid context 准备和清理，
不作为可逐块回收的 historical KV。文本的四轴 M-RoPE 位置会检查为相同的逻辑位置；
本次不支持视觉输入、MTP 或需要 recurrent 回滚快照的 `n_rs_seq > 0` 配置。
对应的上游子模块改动保存在 `patches/0015`、`0016`、`0017`，并加入常用补丁脚本。

historical capture 在一轮 decode 中保留一个可增长的尾块，同一块接收后续 append。
未封口期间，它不能发布为可读历史，也不能被 GC 回收；满 32 token 或请求结束时才封口。
纯 RAM 路径开启此功能，pinned/NVMe 路径继续使用原有规则。单测验证了连续 31 次单 token
输入、一次输出填满首块、下一次输出新建短尾、请求结束封口，以及 token 数、字节顺序和
封口前不可读/不可回收。远程真实模型两轮 smoke 观察到 53 token 占 2 块、接续 32 token
占 1 块，多轮前缀复用正常。

本地 CPU 构建与四项历史块单测通过；远程 CUDA 服务器重建成功，Linux 上完整
`historical_kv_` 子集 15/15 项通过。第一次扩大测试范围时有 6 个旧测试二进制因
Repository 接口更新尚未重新链接而缺失符号；重链后全数通过。
远程另一个 `kvmem-historical-access-test` 在 CTest 中因缺少其专用输入而被跳过，不能计作通过。
三个子模块补丁在隔离远程 checkout 中均通过反向应用检查，确认处于已应用状态。
最终的 `scripts/apply-patches.sh` 又在同一隔离 checkout 执行一次，全部报告
`already applied` 并以退出码 0 结束，验证补丁重放不会重复修改源码。

## 八轮真实负载：LRU、redundancy 与 GC 关闭基线

远程隔离目录为 `/tmp/kvmem-gc-99bef27.3CD9MU`。三组使用同一 IQ3_S GGUF、seed 42、
原生聊天模板、q8_0 KV、32 tokens/block、GPU budget 36864、generation reserve 16384、
每轮 `max_tokens=16384`、sink 1024、recent 0、context 65536。GC 组另设 CPU 历史容量
1024 块，空槽低/高水位 800/820 块；这三个 CPU 数值是本次试验为制造回收压力所选，
不是导师指定的 GPU 参数。GC 关闭组仍保留相同的 CPU 容量。所有八轮都正常 `stop`，
没有截断或拒绝，后续轮次均命中历史前缀。

**这组 800/820 水位是过强压力诊断，不宜作为推荐配置。** 单轮 16K 生成预约约占
512 个 32-token CPU 槽，在 1024 槽容量下，即使尚无历史块，空槽也至多约 512，
达不到 820 的高水位。GC 因此长期处于无法彻底解除的压力状态。

- GC 关闭：生成 4669 token，加权解码 21.23 token/s，首字中位延迟 0.959 s；
  Chromium 功能检查 29/29，通过，未报告浏览器异常。
- LRU：生成 4826 token，加权解码 21.59 token/s，首字中位延迟 0.977 s；
  截止末轮累计提交回收 160 块、完成 reclaim 148 块、提前释放 CPU payload 153 块，
  policy/worker failure 均为 0；Chromium 检查 23/29。
- Redundancy：生成 4729 token，加权解码 21.66 token/s，首字中位延迟 0.993 s；
  截止末轮累计提交回收 143 块、完成 reclaim 139 块、提前释放 CPU payload 137 块，
  policy/worker failure 均为 0；Chromium 检查 23/29。

两种 GC 策略共同失败于射击/换弹状态、最后击杀后的动作、W 键移动及无浏览器异常检查，
浏览器报 `canvas.getContext is not a function`。三组前两轮输出字节相同，第三轮开始分叉；
LRU 在第二轮结束时已经提交 5 块回收。这表明当前压力配置下的输出质量有明显风险，
**不能把 23/29 宣称为质量验收通过**。浏览器检查属于有限的确定性功能检查，也不等于
完整的可玩性评价。

高压力 LRU 再跑一次八轮，仍全部正常结束并通过 23/29 项，说明该配置下的网页失败可复现。

为区分删除内容与运行波动，另在 CPU 容量 2048、低/高水位 800/820 下做了两次三轮
“GC 开启但无实际回收”对照，两次 `committed/reclaimed` 始终为 0：一次第三轮与关闭 GC
的输出不同，另一次三轮全部逐字相同。相同容量的 GC 关闭对照也与原关闭基线三轮逐字相同。
这说明第三轮分叉并非 CPU 容量造成，也不能仅凭一次分叉归因于被删除的历史块；
至少有运行波动或 GC 线程时序影响，需要重复及固定输入试验。无回收对照的运行器以
“要求观察到 GC 活动”作成功条件，所以进程最终报预期的不满足条件；三轮请求均正常完成，
该退出码不是模型或服务器崩溃。

尝试以 GC 关闭组保存的八个 HTTP 请求作固定输入重放：前三轮请求哈希相同且前缀命中；
第三轮生成一旦分叉，第四轮原请求包含基线的第三轮回复，已非当前服务器 KV 的接续，
缓存命中为 0，运行器按 `--require-prefix-reuse` 停止。这个四轮重放不能代替完整的
八轮连续历史 GC 实验；前三轮请求可直接比较，完整八轮必须承认模型自身回复接续造成的
上下文分叉。

三组为各一次运行，且采用模型自身回复接续对话，从第三轮起上下文内容随策略分叉。
因此 21.23/21.59/21.66 token/s 的差异不能被解释为 GC 提速，也不足以精确量化
前台开销；本次未观察到明显减速。各轮虽然把最大生成量设为 16K，实际单轮最长约 1.2K，
不能据此宣称完成 16K 连续生成压力测试。GC 统计是同一个 domain 的累计快照，不可跨轮求和；
`host_released_early` 指 CPU payload 实际提前释放，不是 GPU 工作集回收。

本地原始结果位于 `artifacts/mentor-27b-config/iq3-browser-off-full/`、
`iq3-browser-lru-full/`、`iq3-browser-redundancy-full/`（工作区忽略目录），
各有 `summary.json` 和 `inspection-0/browser-checks.json`。远程原始日志位于上述隔离目录下的
`redundancy-gc/iq3-browser-*-full-20261022/`。

## 可达到的 CPU 空槽水位：384 / 448

将同一 1024 块 CPU 容量的低/高水位改为 384/448；模型、全部导师 GPU 参数、8 轮任务、
seed、浏览器检查器均不变。因为 16K 输出仍预约约 512 块，此时高水位可达到，GC 不再从
任务开头就持续追逐不可能达到的 820 个空槽。以下是每种策略的首轮结果：

- LRU：8/8 正常 `stop` 并复用前缀；累计提交 67、reclaim 67、提前释放 CPU payload 19 块；
  Chromium 25/29，无浏览器异常。失败 4 项均与射击、换弹及最后击杀后的状态有关。
  生成 4488 token，加权解码 19.51 token/s。
- Redundancy：8/8 正常 `stop` 并复用前缀；累计提交 68、reclaim 68、提前释放 CPU
  payload 68 块；Chromium 28/29，无浏览器异常，仅最后击杀后的移动/射击检查失败。
  生成 4574 token，加权解码 20.04 token/s。最终日志记录 485 个特征块、15360 次相似度配对，
  证实该策略实际执行；复测对应为 480 个特征块、15200 次配对。

这表明合理水位下两种策略都能回收，且这轮 redundancy 通过了更多网页功能检查；
但单轮分数、不同续写内容与运行顺序不足以证明稳定优势。后半程的速度比最初 GC 关闭组低，
连续运行导致的 GPU 时钟/温度变化尚未排除，不能把差值全部归因于 GC。

同配置各复测一次：LRU 再次 8/8 正常完成、累计提交/reclaim 67 块，Chromium 23/29，
加权解码 19.61 token/s；redundancy 再次 8/8 正常完成、累计提交/reclaim 67 块，
Chromium 28/29，加权解码 19.61 token/s。两次 redundancy 都只失败于最后击杀后的
移动/射击检查；两次 LRU 的射击状态检查均失败，第二次还出现网页交互异常。
因此这项任务的两次观测均支持 redundancy 的网页功能表现优于 LRU，但样本只有每策略两次，
且都未达到关闭 GC 的 29/29，尚不足以作普遍质量结论。

为核对速度漂移，在上述试验后补跑 GC 关闭八轮：再次 29/29，加权解码
21.26 token/s（初次 21.23），第 6–8 轮约 20.87/20.32/20.62 token/s。
合理水位的两次 LRU 整体约 19.51/19.61，redundancy 约 20.04/19.61 token/s；
后段多个 GC 轮次只有 16–19 token/s。运行中曾采到 GPU 约 81°C、907 MHz，
但关闭组在这段仍维持约 20 token/s；单点温度/时钟和未交错随机化的运行顺序不足以分解
硬件状态与 GC 开销。现在只能判定**有可见的后段减速信号，尚未满足“前台影响很小”的
严格验收**；应另做对齐 token、交错顺序、多次重复的时延试验，并分析后台线程锁竞争。
结果位于 `artifacts/mentor-27b-config/iq3-browser-lru-watermark384-448/` 与
`iq3-browser-redundancy-watermark384-448/`，复测目录同名后加 `-repeat`；远程原始目录名分别以
`iq3-browser-lru-watermark384-448-20261022`、
`iq3-browser-redundancy-watermark384-448-20261022` 结尾。

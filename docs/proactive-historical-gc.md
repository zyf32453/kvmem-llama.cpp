# NInfer 主动历史 KV 回收

本功能位于 `codex/proactive-history-gc`，基于已有 NInfer GC 适配增加独立的 statistical 策略。沿用当前原生 **64 tokens/block、至少一块 recent、单请求、RAM Host 仓库**，不改变历史召回算法，也不覆盖原有 LRU/redundancy 压力回收策略。

设计依据是老师仓库的 [statistical GC](https://github.com/kvmem/kvmem-vllm/blob/a0240f7c0087789edc5bd24b4a71e7e40e4f09ca/docs/statistical-history-gc.md) 和 [verified artifact GC](https://github.com/kvmem/kvmem-vllm/blob/a0240f7c0087789edc5bd24b4a71e7e40e4f09ca/docs/verified-artifact-gc.md)。迁移的是完成步骤、真实观测和保留结果共同授权主动淘汰的机制；没有把“出现 summary”视为整段 thinking 可删除的证明。

当前 thinking profile 使用 `--kvmem-gc-step-redundancy`，复用正常 Terminus `analysis`，在线不再单独生成摘要或调用大模型复核。受信工具结束回调登记步骤，CPU 线程准备完整 thinking 页与保留见证，原生 Mean-K/importance 门槛提出建议，再由 decode 安全边界固定不可选择标记。实现、限制和验收进展见 [thinking-summary-step-gc.md](thinking-summary-step-gc.md)。该 profile 是 statistical 的固定子配置；当前开发版内置独立计数的容量 LRU 兜底，命令行仍不能与其他 GC 主模式同时启用。受保护页和未退出的读者仍能限制可用空间，不承诺无限生成。下文默认阈值、文件等值 producer 和 r6 记录描述原有 statistical/file profile，不能替代新 thinking profile 的验收。

最新审阅记录包含小预算开关对照、官方评分、回收覆盖和对应的源码入口。[R14 性能审阅](terminal-bench-r14-pressure-gc-20261112.md)

## 行为和触发时机

1. 调用方在步骤结束后提交源内容、保留结果以及观测证据；源和结果对应当前请求中的确切 token 区间。
2. 服务端检查会话、实际输入 token 的 SHA256、范围、观测编号和生命周期。默认需要至少 3 次不同观测、连续 3 次 cold、最新 importance ≤ 0.1，且步骤已完成。
3. 满足条件即可产生删除建议，**不需要触及低水位**。策略不以 LRU 有损删除补足主动回收数量。没有合格证据时不删，容量压力仍可能导致正常的容量拒绝；此策略不承诺无限生成。
4. 证据在本次请求的历史空间预约和候选重选之前发布；Query 和 Pressure 的候选入口都检查删除资格，避免等待 worker 下一次轮询期间再次选入。若源块仍在 GPU 工作集中，在完整 prefill 的安全边界，使用原有 fenced spill/restore 切换工作集。历史总量小于或超过 K 都适用，不要求一定是新的 user Query。保留 sink/recent/media 和当前必须使用的块，工作集切换后废弃旧 MTP 草稿。主模型和 MTP 的 KV frontier 不一致时暂缓该切换。
5. worker 产生建议，前台在现有安全边界重新检查 generation、内容版本、证据 revision 和保护条件，然后提交不可召回标记。CPU 存储由原有引用释放机制处理；有读者时延后释放。结果 witness 保留，失效的可选 checkpoint 撤销。

长历史超过 K 时沿用普通 Query/Pressure 的排序和转移流程，并过滤符合条件的源块；不会为主动删除再扫描、回载整个 CPU 历史。该版本没有增加一个不受原生安全边界约束的强制 GPU 删除通道。重选和工作集转移会产生实际开销，功能测量和效率对照分开记录，不能承诺零开销或每个相邻 decode 都完成一轮 GC。时序约束是下一次允许的候选选择必须看见已发布的资格标记，物理释放可以稍后完成。

## 启用方式

沿用现有 NInfer 启动参数，GC 选择改为：

```text
--kvmem-gc-statistical
```

LRU、redundancy、statistical 三个主模式互斥；step-redundancy 是 statistical 的固定子配置。普通 statistical 的默认阈值为：

```text
--kvmem-gc-min-observations 3
--kvmem-gc-min-cold-observations 3
--kvmem-gc-max-importance 0.1
```

普通 Agent 请求如果不提供证据，不会自动获得主动删除权限。普通 statistical 的观测来源和完成判断由调用方负责。新 thinking profile 只接收受控范围，由原生特征门槛选择页；相似度仍是近似信号，不是语义覆盖证明。此前文件 profile 的结果不能替代新 profile 的真实任务验收。

老师的文件落盘例子是独立的实验 profile：文件正文与实际文件逐字节一致、工具成功结果配对后，可以使用 **1 次文件等值观测 / 不要求 cold / 允许 importance 未知为 1**：

```text
--kvmem-gc-statistical --kvmem-gc-min-observations 1
--kvmem-gc-min-cold-observations 0 --kvmem-gc-max-importance 1
```

这不是伪造了一次 attention-cold 观测。文件 profile 应与经过文件核验的 producer 配套；不能只改成这个阈值就允许任意 thinking 内容回收。Thinking 路径应使用专用 step-redundancy 开关及配套控制器。

## 请求接口和证据生成

目前支持 text-only `/v1/chat/completions`，需要显式 `prompt_cache_key`，不支持 media/graft。请求增加 `kvmem_gc`：

```json
{
  "prefix_tokens": 512,
  "prefix_sha256": "对实际 prefix token IDs 按 little-endian uint32 编码后计算的 SHA256",
  "steps": [{
    "id": "step-1",
    "kind": "thinking",
    "source_begin": 64,
    "source_end": 256,
    "result_begin": 320,
    "result_end": 448,
    "observation_id": 1,
    "importance": 0.05,
    "cold": true,
    "complete": true
  }]
}
```

这里展示的是 `kvmem_gc` 字段内部，不是完整聊天请求。范围采用半开区间，token 坐标从完整渲染输入的开头计算。种类支持 `thinking`、`completed_step`、`tool_detail`、`superseded`，种类本身不构成删除证明。

每批最多 64 个步骤，会话同时跟踪最多 64 个步骤；这不是整个会话一生只能执行 64 步。源只取严格在区间内部的完整块，不删除混合边界块、sink 或未封口尾块。结果必须包含已接受的完整 witness 块，且不得与仍在跟踪的任何源范围重叠。一次 batch 全部校验通过后才发布；同一仍在跟踪的 id 的范围和种类不可修改，不能撤销 complete。新观测使用递增编号，原样重发同一编号幂等，改写同一编号会失败。

一个步骤的源块全部完成逻辑退休后，清理其证据记录，解除该步骤对 witness 的额外保护；仍被其他步骤引用的 witness 继续保护。已退休的位置保留有界标记，原样重发旧证据不会复活源块。checkpoint 真正回退时清除后缀证据和后缀标记，重新绑定 KV generation，不能把旧坐标套到改写后的输入。

普通 statistical 目录已满时淘汰最旧的**证据记录**，不因此删除任何 KV，也不令正常推理请求因生命周期总步数而失败。被遗忘证据对应的旧删除建议因 revision 失效；其源内容保留，后续可重新观测。新 step-redundancy profile 不能丢弃已提交的不可选择标记；目录全被已提交项占用时保留新源。日志记录 `tracked`、`completed`、`evicted_evidence` 和 `ignored_retired_updates`，便于区分真正完成回收和放弃管理证据。

Native 维护边界使用批量结果区间标记：一次锁定证据目录，仅遍历已有 witness 范围，并把保护位合并到原有 sink/recent/active 保护图。证据为空时不会为每个历史页分别加锁扫描目录。这降低了长历史下该路径的结构性开销，尚不代表已测得 decoding 的改善幅度。

`scripts/ninfer_gc_step.py` 提供可复用的 `attach_evidence`、`build_evidence` 和 `verified_write_step`。工具通过服务端 `/apply-template` 的 `return_tokens: true` 获取实际准备好的 token IDs 与逐 token 字节，检查完整字节对应关系并生成摘要；服务端再次核对真正准备执行的 tokens。**不再将渲染文本交给 `/tokenize` 重新分词**：消息正文中可能出现字面 `<think>` 等拼写，二次分词会丢失模板对结构标记与普通文本的区分。该接口不执行模型推理，还减少一次分词与 HTTP 往返。服务端缺少此能力时拒绝生成证据，自动桥接保留原请求正常推理，不使用不可靠的旧路径兜底。无法唯一定位、文件已改变、工具失败、配对不唯一或 witness 不完整时拒绝生成证据。

应用提供步骤描述 JSON 后，生成待发送请求：

```text
python3.11 scripts/ninfer_gc_step.py --base-url http://127.0.0.1:18081 \
  --request next-query.json --steps approved-steps.json --output evidence-request.json
```

`approved-steps.json` 是数组，包含上述 step 的 id、kind、observation_id、importance、cold、complete；用 `source_text`、`result_text` 代替四个 token 坐标。文本必须在完整渲染中唯一出现。短结果可以向外扩到已存在的完整块作保守保护，不会为了凑块而制造输入 padding。

DSH `write` 成功结果格式与老师示例一致时，可使用：

```text
python3.11 scripts/ninfer_gc_step.py --base-url http://127.0.0.1:18081 \
  --request next-query.json --verified-write-call CALL_ID --workspace WORKSPACE \
  --output evidence-request.json
```

实际文件必须在 workspace 内，最大 1 MiB，内容与 `write` 的 `content` 参数完全相同；tool 结果必须与该 call id 对应且符合成功格式。此 producer 只生成正文证据，随后把生成的请求正常发送到聊天接口，不会自行执行工具。

## Agent 自动接入

`scripts/ninfer_gc_agent_bridge.py` 在已有 OpenAI 请求前增加证据，Agent 的提示词、采样、输出预算和工具执行方式保持原请求的值。默认不增加 logprobs；使用同一转发器原有的逐请求日志记录，分别保存原请求、带证据请求、准备结果和模型响应。桥接准备失败时原请求继续转发，失败单独记录，不伪造一次成功回收。

桥接会读取服务端 `/props` 的实际 `kvmem_gc` 配置；仅当它确认为 statistical、64 tokens/block、RAM-only 和文件 profile **1/0/1** 时生成文件等值证据。默认 3/3/0.1 统计模式不能直接接收这种只有一次文件核验的 producer。每请求最多核验 4 个候选，每会话保留 64 个待核验候选、最多 16 个会话；新完成项优先，已知项在预算内轮转重试。重试仍是同一个文件观测编号，不累计 cold。

DSH 的结构化 `write` 调用共享真实 workspace 时：

```text
python3.11 scripts/ninfer_gc_agent_bridge.py --workspace /actual/agent/workspace \
  --upstream http://127.0.0.1:18081 --port 18083 --out /actual/gc-audit
```

将 Agent 的 API base 指向 `http://127.0.0.1:18083/v1`。桥接进程对应**一个受控任务尝试**；客户端未传 `prompt_cache_key` 时，使用本进程新建的稳定标识，也可用 `--session-key` 显式指定。已有客户端标识不会被覆盖。不同任务不得复用一个隐式会话；切换任务应重启桥接或由控制器提供新的显式标识。手工提供 `kvmem_gc` 的请求保持原样，其会话由该调用方负责。

Terminus-2 的 JSON `commands[].keystrokes` 路径已增加独立 producer。它识别 `cat > PATH <<'END'` 或 `cat <<'END' > PATH` 的单条 literal heredoc 写入，要求紧随其后的消息确为 Terminus 终端反馈，并核对实际文件全部字节。回收范围只包含模型输出 JSON 中对应文件正文的完整块，分析、计划、命令头/路径、分隔符、thinking 和工具反馈保留。原生模板对指定消息做对照渲染来确认位置；同文示例不会获得源正文的身份。

对于实际共享的文件目录，增加 `--terminus --terminal-root /app`。目录映射必须由实验控制器确认。当前 producer 只接受绝对文件路径；终端可能已经执行过 `cd`，不会根据回显或容器默认目录猜测相对路径。对 Docker 隔离任务，使用控制器明确拥有的单个容器：

```text
python3.11 scripts/ninfer_gc_agent_bridge.py --container OWNED_TRIAL_CONTAINER \
  --terminal-root /app --upstream http://127.0.0.1:18081 \
  --port 18083 --out /actual/gc-audit
```

容器 ID 由启动者提供，不从模型文字中推断。核验使用固定参数的只读文件查询，不执行模型给出的 shell 命令。任务容器需有 `python3`。动态展开、追加、管道、混合命令、未落盘脚本及无法核验的文件均不产生证据。文件匹配不等于任务通过，也不证明 thinking 已被 summary 覆盖。新的 thinking producer 使用独立工具反馈边界、正常 analysis 与原生特征门槛；旧大模型摘要/复核器只保留作离线参考，详见专门文档。

已对提供的 `dna-assembly` 轨迹第 21 轮完整输入做候选发现检查：41 条消息中，此 literal-file producer 找到 **0 个候选**。该轨迹主要通过 `python3` stdin 计算并生成输出，不能把输入的 Python 程序等同于最终文件正文。因此该任务可继续用于阶段 token 占比统计，但目前不能用它证明这个 producer 已触发主动 GC。真实主动回收验收必须记录实际合格证据、提交/释放数，不能仅凭服务返回 200 或 Agent 有输出判定通过。

## 本地验证与待验收项

2026-11-03，Windows / MinGW GCC 16.2，Host 构建中 25 项测试均通过。新增生命周期测试覆盖无压力主动删除、未完成步骤拒绝、缺失 witness 拒绝、source generation 变化拒绝、证据 revision 变化拒绝、边界块保留以及读者延迟物理释放。连续 120 步使用 8 个仓库槽位完成 120 次提交和物理释放，完成步骤不会永久积累元数据或 witness 保护；目录淘汰、旧建议失效、事务回滚和真正回退后的重用也有测试。

另有实际后台线程测试连续执行 48 步，前台不调用 `manual_tick()` 推动扫描，只发布证据、提交标记和移交存储所有权；后台自行选择并释放。每三步保留一个真实 Host 读者，验证逻辑退休先于物理释放、读者结束后槽位可复用，48 次边界提交均完成，worker/policy 失败数均为零。这仍是 CPU Host 集成测试，不能代表完整 GPU decode 时序或其 p95。

证据生成工具、自动桥接与验收审计在 Python 3.11.16 下共 15 项测试通过，包含实际本地 HTTP 转发测试；原有 token 分阶段统计的 5 项回归测试通过。新增终端测试覆盖 escaped JSON、多字节正文、重复文本来源、文件改变、工作区越界和固定容器查询参数。HTTP 测试验证了客户端缺失会话时的控制器绑定、核验异常时继续转发、非法请求保留原生 400，以及查询参数保留。这里的文件和 HTTP 模型为测试夹具，不能视为真实模型或真实 Docker 容器验收。

独立服务端证据 schema 测试已构建运行。完整原生 CUDA 工具链现已恢复；启动参数、OpenAI 聊天 schema、Qwen 前端和 Host GC 四组测试在 r4、r5 及相应 r6 二进制上均运行通过。不能把本地编译单元检查与这些实际运行结果混为同一项检查。

远程环境已于 2026-11-03 通过命令恢复：Docker Desktop 引擎重启后，补齐其 Ubuntu 集成代理连接；没有重启承载 SSH 的 Ubuntu，也没有改动 NVIDIA 调试注册表。该代理是本次运行期间恢复的进程，不承诺跨主机重启自动恢复。源码包 `e163b2013c3d94ef87a2c1e229718310f292f189ecc5e001286300f18fbc25ca` 经授权传至 `/tmp/kvmem-ninfer-gc-20261028/proactive-20261103`，远端 SHA256 一致。

该源码包在远程 Ubuntu/GCC 11.4 下完成 21 项 Host/证据格式测试，全部通过，包括 120 步生命周期与实际后台线程的 48 步测试。21 项由 20 项共享 KVMem 测试和独立证据 schema 测试组成，与 Windows 主工程的 25 项集合不同。另在独立 Docker 容器实测了文件字节一致、终端写入证据、改写拒绝、目录越界拒绝、符号链接越界拒绝和文件上限，共 6 项检查通过；这 6 项检查未进行模型推理。

完整 CUDA 构建使用保留的 `kvmem-ninfer-gc-builder:20261029` 镜像、GCC 13.3、CUDA 13.2.86 和 sm_86。原镜像缺少 Python，已在同一隔离目录补齐 Python 3.11.14，不改系统解释器。r4 功能版本对应源码包 SHA256 `f639c085f2ff817fda1f1091a3dca46a0ec5edac4ea42018fc1e3a09ee9977d3`、原生 prepared tree `288c67ad83c00940ad0e5920338251722b81f360`，构建标识为 `proactive-f639c085-288c67ad`。准确二进制 SHA256 与四组测试命令见远程 `native-r4-tests.json`。原生代码和 Python 审计工具分别留存版本；后续审计工具修正不代表原生二进制重新构建。

首次真实模型暴露了一个实际接入缺陷：Chat 的 `prompt_cache_key` 之前只启用缓存边界，没有传入引擎会话，证据被 HTTP 400 拒绝。r4 已保留并传递显式会话标识，且不会覆盖 Responses 已提供的会话；启发式 `--derive-session-keys` 仍不是此功能的依赖。r3 的 400 和一次验收容器遗漏驱动库的启动错误分别保留，均不是模型任务的 0 分。

两组 r4 使用同一 IQ3_S 模型、96K active、16K reserve/output、1K sink、64 recent/block、256K logical ctx、4760 MiB RAM Host、rk4v4 KV、Mean-Q cosine/max、MTP4 和原生默认 ngram，单路执行。文件任务采样为 temperature=0.6、seed=1000、top_p=0.95、top_k=20、min_p=0，思考上限 4K；两组只改变 GC 及配套证据。它是相同配置的功能对照，不是老师 Benchmark 成绩的复现。

`model-off-k96-r4` 和 `model-stat-k96-r4` 各三轮真实请求全部 HTTP 200，模型实际生成并读取了 64 行文件，字节、行数和 SHA256 均正确。statistical 在 ready 高于高水位时提交并物理释放 **15 个 64-token 块（960 tokens）**，逐块与已核验源范围关联，无未核验删除、无 LRU 兜底。证明的是已落盘文件正文的主动退休，不是 thinking/summary 通用语义覆盖。

`model-http-bridge-k96-r4` 从上述真实 Agent 请求重放到实际 HTTP 桥接与原生模型：客户端未传会话标识或证据，桥接自动核验并绑定会话，两次请求均 HTTP 200、返回正确 JSON，实际提交/释放 15 块。下一请求恢复到 frontier=4357，命中 4357/4390 tokens（约 99.2%），日志的 `ignored_retired_updates=2` 表明重发证据没有复活退休源。这是桥接与连续请求集成检查，不是一个新的 Terminal-Bench 尝试。

上述 r4 功能运行启用了测试 hook、transfer 校验及逐块审计，**不能把请求均值当作 token p95，不能用这些功能结果宣布达到 p95 增幅 ≤5%、吞吐下降 ≤3% 的效率门槛**。后续同日志级别的 r6 效率结果单独见文末；正式 Agent 质量测试也须确认实际触发了这个 producer，而不是只检查 HTTP 成功。

`scripts/ninfer_gc_proactive_probe.py` 提供真实模型文件任务的独立验收入口：模型通过受限 `write`/`read` 工具实际生成并读取文件，检查精确字节、行数和 SHA256，并保留每轮请求、证据和响应。任务通过与 GC 通过分开记录。开启审计后，statistical 提案记录原始 token 范围、slot/generation/round 和当时的 ready/low/high；验收将其与实际 committed ack、已批准源范围及逐块 `reclaim ready=1` 关联，按事件身份去重。请求汇总计数会随所有者延续或重建而重复或重置，不将其直接相加作为总释放量。仅有提案、删除发生在压力水位内、没有物理释放或文件任务失败，均不能通过主动回收模型门槛。该小任务不是 Terminal-Bench 官方成绩，不能替代正式任务与效率测试。

远程原始结果均在 `/tmp/kvmem-ninfer-gc-20261028/proactive-20261103`，保留每轮请求、转发证据、实际响应、工作区快照、完整服务日志和准确命令。`model-*-k96-r3` 为旧故障及其基线，r4 起修复会话问题；文件 profile 的参考基线是下述 r6，新 thinking profile 见本文顶部入口。继续验收时使用新目录，不覆盖旧实验，也不混用不同二进制的对照结果。

## r6 文件 profile 连续生命周期验收（历史基线）

r6 源码包 SHA256 为 `fb99101a09520367c5a04c2ad0351cee16c2aa0cdc7354d67bf849ede7a6d2bf`，原生 prepared tree 为 `f92ea0f4c720e9597c4b395d018539ae6a95a2ff`。实际服务二进制 SHA256 为 `4371b7328e788d39e0d5b35b33588e24e472e76efb76f63b478eba5650f6a1c7`，构建标识 `proactive-fb99101a-f92ea0f4`。该版本关闭 Query 测试 hook；功能诊断仍可单独启用 transfer 校验和审计，效率测试两组同时关闭这些额外日志/校验。构建、四组原生测试及命令见 `native-r6-tests.json`。

r6 包含两项由真实模型发现的后续修复。第一项是上述 prepared token 接口，避免普通正文中的特殊标记拼写在二次分词时改变 token 身份。r4b 中的摘要不匹配 400 保留供溯源；服务端摘要校验没有放宽。第二项是 statistical 回收器的生命周期：保留、恢复到同一个已接受 frontier 时，让稳定的回收器随历史状态一起移动，保留绑定和排队释放任务；真正回退前缀时仍停止旧回收器并撤销后缀绑定。LRU/redundancy 的既有生命周期不改。

`model-off-k96-r6` 和 `model-stat-k96-r6` 在上述同一 96K 功能配置下，各三轮均 HTTP 200，文件字节、实际读取及行数/SHA256 正确；statistical 实际提交并释放 15 块。它们是 r6 文件 profile 二进制的功能基线。

`model-chain-k2-r6-natural` 是单独的 **2K active 生命周期诊断**：保持 16K reserve、4760 MiB Host、64-token 块、1K sink、64 recent、MTP4/原生 ngram，以真实模型自然结束的响应延续同一会话。它重放受控的真实文件写入与核验，输出整数作连续请求检查，不是官方 Agent 任务或 96K 质量成绩。66 轮全部 HTTP 200，65 个步骤完成元数据清理，**578 个源块（36,992 tokens）全部提交并到达物理 ready**。逐块提案、提交和释放事件一一关联到已批准的源范围，全部发生在 ready 高于高水位时，无未核验删除。事件只有一个 GC incarnation，说明同一正常延续历史没有每请求重建回收器。

最后一次请求结束的汇总记录 569 个 ready；前台已经完成 578 个逻辑退休。保留的 worker 在随后空闲期间使余下 9 个块到达 ready，最终逐块审计为 578/578。验收等待 2.5 秒观察后台完成，不把这段等待计入生成耗时。这验证了物理回收能够晚于 decode 完成，而逻辑退休无需等待物理释放。普通 decode 边界只移交所有权；原有 prefill 路径仍允许提前释放 Host owner，不能把所有 ready 事件都解释为同一种释放路径。

旧 `model-chain-k2-r5-natural` 的 66 轮同样 HTTP 200，65 个源范围均已逻辑退休，但只能核验 329/578 个 ready 事件：请求结束时旧回收器被销毁，不能据此证明其余块由后台释放，也不能仅凭缺失事件判断内存泄漏。r6 在同一自然响应诊断下补齐了这一生命周期验证。更早强制 `ignore_eos` 的诊断因结束后的角色/控制类输出破坏后续前缀复用而主动停止，该夹具不用于 GC 效率或质量结论。

通用 thinking/summary 语义覆盖、完整 Terminal-Bench 成绩与多并发仍不属于这些窄场景结果已经证明的能力。

## r6 同配置效率初验

结果在 `latency-k96-r6`。普通 decode 两组各重复 2 次，MTP4/原生默认 ngram 两组各重复 3 次，顺序交错；每次重置模型及 Host 状态，以不保留 Host 前缀的 256-token 请求预热。所有测量使用同一 r6 生产构建、96K active、4760 MiB Host、64-token 块等配置，并同时关闭 Query 测试 hook、transfer 验证和逐块审计。每次开启 statistical 均实际释放 15 块，五次关闭 GC 均没有回收。

效率输入是同一受控文件轨迹加整数输出指令，temperature=0、seed=1000、thinking 关闭；每次完整输出 2048 tokens。单请求使用 `ignore_eos` 固定计时长度，不把这种强制续写的输出当作下一轮历史回放。连续生命周期使用自然结束的响应，两种用途分开。10 次均 HTTP 200、完整 2048 tokens、无 SSE 错误，检查到的前 20 个整数正确。该检查不是完整整数任务或官方 verifier 成绩。

使用服务端累计 timing 差分，普通 decode 每次具有完整的 2047 个单-token 间隔；MTP 接受多个 token 时按实际验证轮次统计，不把轮次 p95 当作单 token p95。合并同组间隔使用 nearest-rank p95，吞吐为合并 token 数 / 合并 decode 时长：

- **普通 decode**：p95 从关闭 GC 的 **52.341 ms** 到 statistical 的 **52.072 ms**，变化 **-0.51%**；吞吐 **23.478 → 23.369 tokens/s**，下降 **0.46%**。
- **MTP4 + 默认 ngram**：decode 轮次 p95 **91.105 → 89.631 ms**，变化 **-1.62%**；吞吐 **47.308 → 47.364 tokens/s**，变化约 **+0.12%**。

该负载下两组都满足脚本对应的“p95 增幅不超过 5%、吞吐下降不超过 3%”门槛。样本较少、负载规则，数值变化接近运行噪声，不将负变化解释为 GC 已被证明能加速任意任务，也不外推到长历史、更多会话或正式 Agent 准确率。

首 token 和证据准备单列：普通 decode 的请求首 token p95 **7.105 → 7.066 s**；MTP **6.891 → 6.978 s**。statistical 证据准备 p95 分别为 **25.6 ms / 90.7 ms**，在发送模型请求之前测量，不计入上述请求首 token 时间或 decode p95。它们表明该机制仍有前置 CPU 和工作集转移成本；本结论是 decode 开销初验，不承诺整个请求零开销。

## r6 HTTP 自动接入与真正前缀改写

`model-http-bridge-k96-r6` 对同一个普通客户端连续测试三次请求：首次核验并回收已落盘正文、沿同一前缀继续提问、再回退并改写文件正文及相应用户要求/工具读取结果。三次均 HTTP 200；两个版本各自的实际文件行数和不同 SHA256 均回答正确。客户端不必手工提供 GC 字段或会话标识，桥接自动处理且不添加 logprobs，不改变原采样。

前两次延续同一历史，重发的证据不复活退休块；第三次真正改写前缀后重新绑定 KV，旧 source 标记不套用到新内容。逐块审计共 30 次提交和 30 次 ready，分别来自原始版本和改写版本的 15 块；中间的幂等续问没有重复删除。这样既验证了正常延续保留回收器，也验证了真正回退仍撤销旧绑定。

`acceptance-final-r6-state.json` 记录四阶段完成：原生测试/同配置功能对照、66 轮连续生命周期、10 次效率测量、三次 HTTP/前缀回归。测试服务均已停止，原始结果保留。以上为主动 GC 基础机制和可核验文件 profile 的初步验收；尚不能宣称任意 Agent 负载、通用 thinking/summary 覆盖或完整 Terminal-Bench 准确率已经通过。

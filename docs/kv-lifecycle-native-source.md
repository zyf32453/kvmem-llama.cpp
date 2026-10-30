# C2：NInfer 已接纳源记录接线

日期：2026-10-31。父提交 `763889597c022a52ac42f8b4ad5c0301cdfad1d7`。本批在 C1 的 SourceJournalStore 上接通原生记录，不提供 KV 重建、持久恢复或系统级永久遗忘；阶段 C 及完整六阶段目标仍未完成。

## 接口与身份

`ninfer-serve --kvmem-source-mib N` 显式启用 S 配额，默认关闭。N 必须为正且换算为 bytes 不溢出，要求 KVMem、单 GPU、单运行请求、RAM，以及普通/MTP 执行；首批模型验收固定普通与 MTP4、64-token page、rk4v4、Mean-Q/max。多 GPU、SSD 和多 lane 在解析/Engine normalize 均拒绝；文本适配器拒绝媒体、graft 和自定义位置行。没有把 parser 接受的其他 MTP 宽度宣传为本次已验证配置。

Frontend 从实际 tokenizer JSON/config、当前实际模板内容（包括文件覆盖）生成 SHA-256；聊天模式再绑定模板参数、continuation/thinking 选项，raw-token 模式使用不同域标记。已知消息边界按实际 token 区间记录 System/Developer/User/Assistant/ToolResult；只有有证据的生成头标为 Control，未知范围保持未标注，原始 token 输入为 RawInput。事件 ID 是本份输入中的消息序号，不是完整外部 tool_call_id/任务版本目录；后者仍属于后续任务语义接线。

Program 将 frontend digest 与模型 artifact identity、后端 schema、KV 编码、RoPE、prefill/投机和召回等执行配置合成源配置身份。它证明源的解释环境，不能凭这个 digest 授权旧 KV 恢复；精确 checkpoint/退休/计算谱系仍必须各自校验。

## 接受边界与清理

首次 prefill 单元前，保存整个已经接纳并归 Program 所有的 prepared input。这里“源已接受”不表示输入已经全部算过：prefill 中断时，源 accepted frontier 可等于完整输入长度，而 computed frontier 只到当前完成的逻辑边界。这样没有把源文本与物理 KV 覆盖混为一谈。

输出在完整 Program `commit` 成功的末尾写入；强制插入的控制 token 在整个 `append_forced_tokens` 成功后写入。不能接在 prefix-identity helper 上：该 helper 在强制插入时可能提前、重复执行。未接纳的 pending row、被丢弃的 MTP draft 和因源配额失败而回滚的候选不进入源记录。最后采样 token 可已接受、尚无对应 KV，两个 frontier 分开报告。

每个 Program 独立拥有源目录，request/window 只保存当前 SourceRef；物理窗口/冷历史驱逐不会销毁已保存的源记录。源配额不足继续使用第 26 补丁的两种清理：有效 sequence 有序 abort，已消费 membership 只结清其资源记录。失败保留已保存输入和之前输出，不隐式遗忘、不仅因额度不足锁存整个 Engine，等待 FIFO 仍可继续接纳有空间的请求。

## 用量、成本及范围

`/props` 配置和 usage 报告 `source_budget_bytes`，usage 还报告当前/峰值/拒绝、记录数、最后源 ID/revision、accepted/computed frontier、事件数和诊断摘要。S 用量是该 Program 内所有源记录总和，last 字段只描述最近记录，不是某个 KV window 的累计值。

目前固定最多 256 份记录，每份 token/event 数量也按上下文限制。S 包含固定目录、真实 token/event buffer 容量及 old+new 扩容预约。它不包含有界的 allocation/control 对象、allocator 开销、prepared-request 的源事件暂存、模型/其他运行区、checkpoint 或 RSS。H、M、S 是独立口径。

当前每次已接纳请求都会保存独立输入，包括后续执行失败的输入；未做跨请求输入去重、源过期/归档或公开遗忘接口。空间/目录满时会拒绝后续捕获或追加，即使 KV 还能回收。分支复制、几何扩容和保留旧 buffer 都有真实成本；没有性能不变或完整进程内存上界的结论。首版默认值、源保留策略及总体资源上界仍需后续阶段验收。

诊断 `source_last_tokens_fnv1a64` 按实际 journal 的 little-endian int32 token 计算 FNV-1a；不是恢复许可或密码学完整性依据。启用 `--kvmem-verify-transfers` 时，每段新增 journal token 还直接与实际已接受的 Program token 逐项比较，原始输入也逐项校验。

## 验收入口

[ninfer_source_acceptance.py](../scripts/ninfer_source_acceptance.py) 固定服务/模型/脚本身份，创建并清理自己的隔离容器，保存每项 argv、源配额、响应、日志及失败。功能配置为 B=2048、R=512、H=96 MiB、context=262144、prefill=512、rk4v4、Mean-Q/max，审计及逐 token/传输检查开启；不能当作配对性能档位。

- `capture`：公开 tokenize 得到原始输入 IDs，completion 返回实际接受输出 IDs；核对数量、最后采样的未计算边界和源实际内容摘要，再发另一请求确认前一份源没有随换窗丢失。
- `quota`：S=1 MiB，有限次原始请求制造真实压力；核对额度拒绝、失败候选未写入，然后让一个已输出的流触及 S 上限，验证 typed terminal error 和此前已排队请求继续回答 599。
- `cancel-prefill` / `cancel-decode`：先证明真实执行阶段，再断开客户端。prefill 用公开模板/分词结果核对完整已接纳输入、尚未完成的 computed 边界和无 provisional output；decode 检查保留的接受边界、owner/预约归零及后续请求成功。

当前候选指纹及主机契约固定在 [native-source-candidate-v1](contracts/kv-lifecycle-native-source-candidate-v1.json)。最终各项结果、版本归属和未通过尝试以本批执行摘要为准；不能将较早候选或主机检查当作最终二进制的 GPU 证明。

## 本批结果与未完成项

最终原生 tree 为 `3a7573b8f77e5e0d266e67e2a78b252e78e96ec2`，第 27 补丁逐项重放核对完成。服务 SHA-256 为 `919b18eb350ef5cbd9d67d5d7736f04a3d4305be145c694413fb0859dc47c874`。21 项 Windows Debug 契约、7 项原生检查通过，包含选项/默认配置、实际 frontend 来源与模板配置，以及源记录组件的 12 个分配失败点。原生 Release 组件使用 CHECK，未以被 NDEBUG 关闭的 assert 冒充通过。

最终服务的真实模型结果见 [执行摘要](experiments/2026-10-31-lifecycle-stage-c-native-source/summary.json)：

- 四组 LRU/redundancy × 普通/MTP4 内容检查通过。每组原始输入 6943 tokens、实际接受输出 64 tokens，源 accepted=7007、computed=7006；原生逐 token 对照与公开 IDs 的内容摘要都相符。S=16 MiB 时源峰值 127092 bytes，下一请求没有丢掉前一份源记录。它是源记录检查，不是模型删除后恢复或任务质量验收。
- 两组 LRU × 普通/MTP4 的 S=1 MiB 检查通过。各 14 次填充成功后触及明确拒绝，另外验证失败候选未入源、已输出流再触及源上限，以及此前真实排队的请求继续回答 599。终态源计费 1046080、峰值 1047576 bytes，均未超出 1048576 bytes；各观察到 3 次源拒绝。error 前分别已有 844 / 248 个 SSE 帧。
- 四组取消检查通过：LRU/普通和 redundancy/MTP4 各验证 prefill/decode。prefill 取消保留完整输入 13897 tokens，computed=1024，没有把未执行部分或 provisional 输出写成已计算；decode 取消的接受/计算边界分别为 3547/3546、3552/3551。owner/预约归零，后续请求均成功。

另一个较早二进制 `0668e397…` 的四组内容和两组额度检查分开保存，未归属给最终版本。保留三个已修正的开发/驱动问题：首次构建驱动多复制了不存在的 `/old/tests`；首次编译暴露模型头消费者缺少直接库依赖，已补齐相关目标；嵌入核心未在根 CTest 登记，因此第一次调用未找到测试，之后直接执行实际构建的组件，原始 exit 8 保留，最终驱动也采用直接执行。

源记录尚未按会话去重或过期；预热和后来失败的请求同样会占源空间/目录。保留旧 buffer 扩容会增加临时峰值。这些成本没有用“GPU 正常运行”掩盖。本批未测配对性能、整个进程 RSS 上界或新 Agent 质量；原来偶发 CUDA 与 TSan 环境限制仍未关闭。

下一步是恢复 planner 和逻辑目录：复核 source/checkpoint/退休/版本条件，先预约恢复空间，执行受控新域重建并在成功后发布。RAM 记录、source digest 或旧 checkpoint 文件存在都不等于可以恢复；系统级遗忘、归档 fence、持久恢复、任务语义和多层/多会话仍需后续完成。

本地原始包为 `artifacts/lifecycle-stage-c-native-source/evidence.zip`，SHA-256 `18451c2336af94fb4e3e9f1a8408db0a8b57fb337cb78915d7e91dc06d8ec6b7`，包含最终与中间日志、编译失败、驱动修正、原生输入及主机结果。Git 保存补丁、工具、契约和摘要，不加入模型或大体积编译产物。

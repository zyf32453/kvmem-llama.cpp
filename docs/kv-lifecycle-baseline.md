# KV 生命周期第一阶段基线与契约

日期：2026-10-31。第一阶段以源码、支持边界和可执行契约为交付物，不把它称为完整生产验收。

## 主后端决定

主服务接入采用 NInfer，冻结源提交 `a50789573e4ba43b137076c15b552d0e3a98e9ca`，原生后端基底 `63428ac249da8a6c5d891b508d7e483d8c7def6c`，24 个有序补丁后的 tree 为 `09122c9e434b4861a62eec19f25a9b94c6f5bdd5`。补丁逐项 SHA-256 来自 `backends/versions.json`；机器可读清单同时固定公共核心和相关测试源码，换行统一 LF 后检查。

选择依据是该路径已有原生 Host 容量、checkpoint/尾页修复、MTP 和实际 GC 后继续完成 Agent 任务的记录。不是基于两后端速度排名；没有同硬件严格对照可据此宣称 NInfer 更快。较新的公共 `memory_contract.hpp`、`HostKvStorage`、content/history 版本和 checkpoint 目录直接复用，不再从零建设另一套同义目录或存储账户。

llama.cpp 保留参考基线 `be21a18e6141dbd4a6ff26d214e01903ffb947c3`。共享行为探针以相同源码分别链接两份核心，检查可迁移的生命周期不变量；NInfer 原生 page 保护和准入差异单独声明。不会把历史 llama.cpp 的 32-token/q8_0/GGUF 运行算作本 NInfer 64-token/rk4v4 的验收。

开发在独立 `codex/kv-lifecycle-baseline` 分支开展，保留原工作树的用户修改。

## 模型与实验身份

固定 `Qwen3.8-27B-GSQ-RCO-IQ3_S NInfer v3`，模型文件为 15,017,456,896 bytes，SHA-256：

`29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`

本次在 `yifan@100.118.87.22` 只读计算完整文件哈希，前后大小和修改时间相同；未复制模型、启动推理或修改主机设置。位置记录在 JSON 清单。既有最终 CUDA 服务二进制哈希为 `4a66fc88d2acf941875399e35f6be77e9f4260d25362f9a21f4dde63ea37cb1f`，只标识旧验收，不假定新构建逐字节相同。

新 GPU 运行仍须记录实际模型/服务哈希、补丁 tree、GPU/驱动、采样、有效预算、任务/镜像/依赖身份；本阶段的主机测试不能替代它。

## 首批支持范围

首批是单 GPU、单运行请求、文本、RAM historical KV、64-token 原生页、`rk4v4` KV、Mean-Q/Mean-K cosine Query scoring、max 聚合，关闭 LRFU 和 graph 混合。普通解码与 MTP4 各有独立配置和计时口径。

原生 GC 开关为不传 GC 标志的 off、`--kvmem-gc-lru`、`--kvmem-gc-redundancy`，后两者互斥。公共 manager 的 shadow 类型存在，原生 CLI 没有声明 shadow 接入，不能把旧 llama.cpp CLI 的 off/shadow/delete 参数照搬过来。

冻结四个复现档位，完整值与既有脚本入口见 JSON：

- `native-token-micro`：普通解码；selected 2048、reserve 512、Host 320 MiB、context 16384、固定输出 2048、thinking 128、520 records；seed 42、temperature 0、ignore EOS。报告完整单 token 覆盖。
- `native-round-mtp4`：同预算、MTP4；延迟单位是验证/解码轮，不能冒充逐 token p95。
- `agent-pressure`：selected 8192、reserve/output 16384、Host 1190 MiB、context 262144、thinking 4096、sink 1024、recent 64，MTP4、seed 1000、temperature 0.6。诊断容量压力，不是推荐生产默认值。
- `agent-teacher`：selected 98304、Host 4760 MiB，其余沿用上述 Agent 档位。用于老师预算的集成/质量对照；如果未触发 GC，就不能支持 GC 后质量结论。

selected 历史与 generation reserve 是不同预算；8192/16384 的已用配置合法，不能假设 reserve 必须小于 selected。微型计时脚本沿用原生 sink/recent 默认值，后续 GPU 运行须记录实际生效值；没有把未知值写成 0。任务/Harness/镜像身份按每次运行锁定，未将整套基准集任意版本纳入首批支持。

以下列为**未纳入第一版支持**，其中某些能力在项目其他路径已存在：SSD+GC、多运行请求或多 GPU GC、多模态 GC、其他 KV 编码、任意 active/Writing 页退休、NInfer 整请求严格预约、在线换策略、通用持久恢复与永久遗忘。开发检查器拒绝这些档位；并不声称原生服务已经对所有未验收组合实现拒绝。这些运行时 capability gate 属于后续接线。

## 四种动作契约及当前覆盖

### 释放副本 EvictReplica

逻辑历史仍可用。只有合法替代副本已提交，或明确可执行的恢复计划存在时，才归还该物理副本；租约持有期间仍计费。公共 `HostKvStorage` 分层组件有相关能力和测试，原生 GC+SSD 组合未验收。本阶段只验证组件契约，不宣传原生组合支持。

### 退休历史 RetireHistory

当前原生已实现的操作。后台准备意图，在前台安全边界复核并提交 tombstone，随后新 Query/压力选择/checkpoint 不得再次取得该身份。既有操作按真实许可完成。CPU 释放与完整 Ready 分开，退休本身不等于释放完成。

NInfer 只注册完整 sealed 页，并复核 active、sink/recent、media 和在途依赖的保护。checkpoint/Query 临时保护在需要结束后解除；不能把临时保护变成永久保留。repository 的广义退休能力与这个原生支持档位分别说明。

### 作废派生版本 InvalidateDerived

现有 content/history 版本、partial-tail descriptor 失效和 checkpoint 依赖校验继续使用。truncate 到 96 tokens、page=64 时，页 0 描述保留，页 1 旧描述必须失效。任意 timeline 编辑及其完整后缀重建不是本阶段新增的已支持 API。

### 永久遗忘 ForgetSource

独立于内存压力，未来必须撤销源记录、归档别名和恢复资格。当前不提供通用实现，不能通过把清理缓存命名为 ForgetSource 来宣称完成。

## Active 与完成边界

1. 主 owner 在下一次新选块前提交已准备批次；`progress_historical_gc()` 到原生选择器的接线由固定补丁及旧 GPU 记录支持，主机测试只覆盖 bridge/manager 行为。
2. 提交后新访问关闭，已许可操作继续，不能换掉已提交的 GPU 输入。
3. 物理 owner detach 与真实内存释放分开；reader/DMA 还持有真实依赖时不得退款。
4. GPU-only 许可可以继续存在而 CPU 数据已释放；完整槽位复用还要等 metadata/GPU 依赖退出。
5. `generation` 不匹配的旧完成、旧删除和旧读取不能作用于新分配。

当前边界通过 `pending_mu_` 交接，且调用策略 `permits()`；因此不保证前台不等待 worker 或不执行策略回调。这是第二阶段的首要改造项，并非本阶段已修复。

## 准入与失败结果

NInfer 首批采用**分块准入**：prefill 边界可最多等两秒，并可立即归还已经 detach 的 Host owner；有 reader 时仍保持其额度。decode 常规物理销毁在 worker 推进。不将这条路径描述为所有释放均在后台或已接纳整请求无需 GC 补给。

后续目标的严格整请求档位仍需真实预约主/草稿、尾页碎片、在途和 rollback 资源，不能通过加大水位替代。已预约资源不能当作公共 Ready，pending credit 只避免过量决策。

现有公共错误 `InvalidPlan / Unsupported / BudgetExceeded / Stale / Pending / TransferFailed / Disconnected / InvalidSession` 保持原义：错误计划、能力不支持、真实配额不足、过期身份、尚未完成、真实传输失败、断开与会话无效。容量错误不能当成磁盘错误；Pending 不能当成恢复成功。新增 RecoveryUnavailable 等恢复错误在第三阶段明确设计，本阶段不发明未接线的 HTTP 状态映射。

## 资源归属及第二阶段缺口

- Host payload/预约：`HostKvStorage` 账户，包含未封口页和仍被 reader 持有的分配；按实际最后释放归还。
- 磁盘 extent/副本：`SpillFile` 与 record/source/lease；过渡两份副本分别计费。这里只覆盖组件，GC+SSD 未验收。
- 原生 archive/host_versions/retired_pages/特征表：sequence/page bridge owner，按 logical context 上界，不等同于 Host 物理页容量；应核算最坏 bytes。
- Query/Mean-K/候选及邻接缓存：feature/policy owner；已有候选比较预算，不等于所有缓存都有完整字节账本。
- pending/retry/观测批次：manager；第二阶段核对容量和交接，不能依赖无限事件队列。
- thinking/回复范围及访问历史：语义来源 owner；需核对合并、truncate 与总条目预算，不能仅因 token frontier 有限就默认所有事件容器有界。
- checkpoint/state：原生 catalog/目录有已有 count/bytes 限制，真实活跃引用另存；进一步追踪跨域别名和占用。
- 源记录、持久恢复 journal、任务锚点：尚无完整统一实现，属于第三/四阶段；不计为本阶段预算已闭环。
- staging、GPU 状态、驱动和模型映射：backend owner；系统自有预算与进程 RSS 分开报告，不把重写推理框架所有分配作为前置任务。

## 可执行检查与证据等级

机器可读契约：[kv-lifecycle-baseline-v1.json](contracts/kv-lifecycle-baseline-v1.json)。检查器：[check_kv_lifecycle_baseline.py](../scripts/check_kv_lifecycle_baseline.py)。

检查器验证源码/24 补丁指纹、模型固定身份声明和配置范围；传入本地模型路径时才重新计算模型文件哈希。指定 build-dir 时必须是本源码的独立 Debug、CPU-only 构建，并重新构建清单中的 11 个目标、运行 CTest、核验每项真实完成，记录测试程序指纹。零测试、缺测、skip 或失败均不得通过。未提供 build-dir 时只报告 source_verified_only。

11 项包含 page bridge、公共 memory contract、native payload、真实存储预算、IO、分层组件、residency、内容版本与三项 checkpoint 测试。契约清单区分 host_test、固定 native patch/旧 GPU 证据和 future_not_supported。补丁哈希核对不等于重放补丁 tree，更不等于新 GPU 运行通过。

共同探针：[kv-lifecycle-compatibility-probe.cpp](../tests/kv-lifecycle-compatibility-probe.cpp)，不依赖引擎类型，分别链接两份冻结核心，验证退休关门、旧 reader 保留、跨代复用、只释放一次和 CPU 早释时保留 GPU 许可。它不测试各原生 adapter 的全部入口。

示例（独立配置好 Debug 构建后）：

```sh
python scripts/check_kv_lifecycle_baseline.py --profile native-token-micro \
  --build-dir build-lifecycle-host --output artifacts/lifecycle-stage-a/run-001
python -m unittest discover -s tests -p test_kv_lifecycle_baseline.py
```

输出目录必须是新的目录，旧证据不覆盖。Windows 可以显式传入 `--cmake`、`--ctest` 的安装路径。

普通性能记录完整单 token 延迟；MTP 按 decode round 另报。原有 p95 +5%/吞吐 -3% 作为预先声明目标，报告配对、最差轮、p99/max；不能只拿平均值判稳定通过。质量使用真实 Agent 闭环，off/LRU/redundancy 同环境同预算；无 GC 的任务仅支持集成结论。

## 第一阶段完成与下一步

第一阶段完成意味着：主后端与版本、模型身份、首批范围、四种动作、Active/准入/错误口径、资源 owner 和可运行契约已明确。所有未来能力和未解决故障仍单列；不能因此宣布生产 GC 完成。

第二阶段先改 manager 的边界 mailbox，保留已有原生动作与保护规则，消除 worker mutex 等待及任意策略回调；随后结构化必要证据并补资源预算/故障路径。第三阶段复用现有 `MemoryStamp / BlockDescriptor / CheckpointIdentity` 增补恢复能力，不创建平行类型体系。


## 本轮执行结果

独立 Windows/GCC 16.2.0 Debug 构建完成，11/11 项 CTest 通过。检查器 9/9 项单测通过，覆盖零测试、缺测、skip/failure/error、未固定模型和超范围配置不能误报通过。另用真实 CLI 完成三项拒绝反例：未知 SSD/并发档位、篡改源码指纹、错误模型字节，均以预期原因退出 1。

同一份共同探针分别链接 NInfer 和 llama.cpp 冻结核心，两个执行均退出 0；同时核对参考核心的实际源码与 be21a18 相符。验证范围是退休关门、reader 生命周期、代际复用、只释放一次，以及 CPU 早释/GPU 许可/Ready 的区分。不是原生两后端整机差异验收。

固定清单含 78 项源码/测试/验证器指纹与 24 项原生补丁校验。本轮未重放原生补丁 tree、未重新构建 CUDA 服务、未运行 GPU/Agent；模型完整文件哈希是本次单独只读测得，主机 gate 报告 local_model_verified=false 表示本地未再次读取远程权重，二者不矛盾。

结构化结果见 [第一阶段执行摘要](experiments/2026-10-31-lifecycle-stage-a/summary.json)。完整日志、JUnit、负面控制和输入源码归档为 `artifacts/lifecycle-stage-a/evidence.zip`，SHA-256 为 `8cf2f9152f4d326c06be6ffeade489220c656433bb9a6d02180c32f74db25057`。归档只含主机验证记录与脚本，不含模型或旧 GPU 运行。

第一阶段状态：**完成基线与可执行主机契约梳理；生产验收未完成。** 后续改动若改变冻结核心，应明确建立候选版本/新清单，不绕过当前指纹检查，也不把此报告套用到新版本。

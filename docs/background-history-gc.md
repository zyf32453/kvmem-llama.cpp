# CPU 历史 KV 后台管理

本分支 `codex/background-gc-vllm` 按用户 2026-10-21 的新指示，对齐
[Di-Chai/kvmem-vllm 的 8560331](https://github.com/Di-Chai/kvmem-vllm/tree/8560331de4d9a889a1e884fe238b91e957a5958d)。
参考其 `docs/background-history-gc.md`、`csrc/historical_kv_policy.hpp` 和
`csrc/historical_manager.cpp` 的行为边界，复用本项目现有 ownership 实现。
本文件取代旧迁移记录中的“完整生产化”待办作为本分支首批交付范围；旧记录保留原始测试证据。

## 范围

首批入口为单 GPU、单 lane、串行文本、recency、flash attention、f16、RAM 历史，
每个请求有有限输入/输出额度。后台 GC 管理 CPU 历史及相关目录容量，不替换 GPU allocator，
不强制中断或换掉已经准入的 GPU 窗口。GPU 容量驱动的正常重选属于新的选用：退休块不能
重新取得 Host 搬运或执行许可；稳定窗口内已有的许可保持到原操作结束。

NVMe、独立 historical pinned 副本、MTP、多 lane、通用 checkpoint 和 redundancy 策略
不作为此次基本闭环的验收前置条件。已有组件保留，但不据此声明这些组合已经对齐参考实现。
首批容量指标覆盖历史 payload、目录和既有固定工作空间，不承诺整进程 RSS 上界或整个推理
框架零分配。DeepSWE 按用户要求继续暂缓，不修改 Windows 调试接口或驱动设置。

参考实现还包含 vLLM 的 P/M checkpoint、多 scope 协调、GDN/MTP 和 eager serving。
本项目的首版串行 RAM 路径没有这些等价调用入口；这不是代码逐文件移植，也不声明后端能力完全相同。

## 行为

1. LRU 在空位低于低水位时开始工作，目标为高水位；默认约为仓库容量的 5%/10%，最小 1/2。
   等待准入的连续请求可以提出超过普通水位的空位需求。已经退休、尚未释放的槽只作策略规划抵扣，
   不作为前台可分配容量。
2. 默认 LRU 只挑 Live、已封口且不属于 sink/recent 保留窗口的块。窗口使用原始 token 位置，
   避免回收和短尾片段改变块编号后错保留。recent 随本域已登记的追加位置单调前进；新域重新开始。
   在原始位置空间中的短片段全部计入窗口，不能将其误当作固定数量的紧凑物理块。
3. 策略只提出意图。manager 提交前重新校验同代身份和策略条件；显式退休和自定义策略仍能删除
   Writing、active 或保留窗口中的块。默认 LRU 的保护规则不成为机制的通用删除禁区。
4. 退休立即关闭新访问；已有 CPU reader/writer/DMA 依赖结束后由后台释放对应资源。
   GPU/metadata 引用不会无条件拖住 CPU payload；完整槽位复用仍需全部依赖结清。
   代际校验和完成回执防止旧操作误用复用后的槽位。
5. `off` 保留 ownership 与退出清理，`shadow` 只统计意图，`delete` 真正退休与释放。
   策略无候选或异常不阻止已退休资源继续回收。

实现上的差异：沿用本项目每域一个 worker，adapter 轮询 5 ms、观察切片最多 64 块、
每批最多 16 个删除意图；没有照搬参考版的多 scope 共享调度器。LRU 基于观察快照，
不是并发访问下的精确全局 LRU。提交前的压力刷新会读取固定槽表的原子状态，
因此观察切片上限不等于整个 tick 严格只访问 64 个槽。

## 容量不足

已准入请求使用既有整请求预约，不等待 GC。精确前缀连续请求在第一次预约失败后，
可在 HTTP/SSE 响应开始前等待实际空位，默认最多 5000 ms，每次重试仍执行完整预约。
`--kvmem-historical-admission-wait-ms 0` 可禁用等候，允许范围 0..60000。
超时返回 HTTP 503，客户端断开则撤销等候；退出等候时撤销后台容量需求。

只有 delete 模式会为这种等候提出删除需求；off/shadow 不能靠假设的释放继续接纳。
超出仓库容量、非法请求和新域 reset 仍直接拒绝。全部历史都受保护时也可能无法接纳，
不能为满足请求而绕过策略保护。相较参考版通用 scheduler，本分支的等候仅覆盖串行精确前缀延续，
并且有明确超时；不是跨会话的无限队列。

`/props` 发布历史容量、水位、模式和 admission_wait_ms。
`KVMEM_HISTORICAL_ADMISSION` 记录准入重试、等待毫秒数、成功和取消状态。
`KVMEM_HISTORICAL_GC finish` 保留退休、实际释放、延期及错误计数。
归还 allocator/槽位不等于 RSS 立即下降；删除历史可能改变后续回答内容。

## 验证

验收覆盖：策略与机制单测、真实模型生命周期回归、HTTP 取消/reset/前缀复用，以及
off/shadow/delete 下的容量压力对照。使用原有真实模型测试，不扩大成通用调度器改造。
新增加的压力回归命令如下（其强制输出 Z 只验证流程，不是回答质量评测）：

```sh
python3 scripts/test_server_historical_gc.py --server build-cuda/bin/llama-kvmem-server \
  --model MODEL.gguf --output artifacts/gc-admission-delete --mode delete --admission-pressure
```

同一命令分别使用 off、shadow 时，应在实际容量不足时拒绝，不能消费虚构空位。
`--admission-blocked` 单独验证全部历史被保护时的超时和连接取消；不与
`--admission-pressure` 同用。普通模式检查 HTTP 生命周期和实际 CPU 释放，压力模式
另行检查完整槽位的回收复用；二者不能合并为一个“reclaimed > 0”的断言。
性能与模型质量只按实测配置和负载报告；尚未定位的 CUDA 故障保留为已知问题，不能因本次
缩小范围或重跑通过而宣布已修复。

### 2026-10-21 验收记录

- 本地 GCC 16.2 Debug：14 项跟踪源码中的历史 KV/回收测试通过，另有用户 demo smoke
  通过（demo 不在本次提交中）。新增的容量需求/取消用例补跑所在的 2 个目标，通过。
- 远程隔离目录 `/tmp/kvmem-gc-99bef27.3CD9MU`：5 项相关 CTest 通过；
  新增用例所在的 2 项再次通过。模型使用用户提供的 `e9183b5c…a07ac` blob。
- 真实模型：43/43 生命周期场景通过，并通过真实 reset、旧域 payload/metadata
  容量拒绝后恢复、36 次输入请求及 48 次含生成请求的循环复用等检查。
- HTTP 容量对照：delete 在同一仓库完成 40 次连续请求，确实发生等待后台回收后再准入；
  6 次等待均成功，观察到 54 个退休、52 个完整槽位回收。off/shadow 均在容量耗尽时拒绝，
  随后 reset 能继续服务。delete 的 127 项检查通过，但启动 shell 未正确记录其退出码；
  此项以 `results.json`、响应及服务端日志核验，不将缺失退出码记作 0。
- 全保护窗口：1000 ms 等候到期返回 503，客户端断开约 113 ms 结束等待；
  两种情形都保留旧缓存供小请求延续，受保护历史提交删除次数为 0。

普通 HTTP 回归首次在末尾“实际释放”断言失败：日志已有 `host_released_early=12`，
但 `reclaimed=0`。原因是旧断言混淆 CPU 释放和完整槽位复用；GPU/metadata 引用仍在时，
CPU 提前释放是预期行为。修正断言分别验证两者，保留首次失败证据并重跑：33/33 项通过，
退出码 0，仍观察到 12 次 CPU 提前释放。模型测试程序、server、libllama、libkvmem
的哈希在两次验收之间保持一致，没有修改产品代码以迎合断言。

本地完整证据在 `artifacts/background-gc-vllm/background-gc-vllm-evidence.tar.gz`，
SHA-256 为 `a2bcadef55deaa8c94f3558406d71f6c4af109b98b911410ea5a5468c047632e`；
同目录有 `background-gc-vllm-summary.json` 和源码指纹。`artifacts/` 按仓库规则不提交，
远程隔离目录也保留原始证据；本文件随代码提交记录结论与复现命令。

强制输出 Z 不是任务质量评测。本轮没有宣称解决此前偶发 CUDA 故障、完成 DeepSWE，
或取得参考仓库全部后端能力及性能结果。

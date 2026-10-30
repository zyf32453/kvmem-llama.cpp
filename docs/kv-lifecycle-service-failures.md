# B：服务故障、停止与持续旧域轮换

日期：2026-10-31。父提交 `012b9f3e323bf105ed9c118b41718fc5bec3f3d1`。本批继续使用阶段 A 冻结的 NInfer 主后端及单 GPU、单运行请求、文本、RAM 历史档位。普通解码和 MTP4 分别验收；已有冷历史轮换不等于已实现 E 阶段多运行会话。

## 发现并修复的问题

第 25 补丁已有准备阶段的预算不足处理，但没有覆盖生成过程中第一次换窗所触发的辅助分配。旧二进制在 M=8 MiB、H=96 MiB 下，先输出约 2250 个 SSE 帧，再在 decode 内准备窗口时触及辅助额度。日志记录 `worker crash: historical KV auxiliary budget exceeded`，流返回未分类的 `internal_error`，Engine 进入共享故障路径。这不是正常的请求级容量拒绝。驱动随后停止进程，最终退出码 0，不能把该退出码写成“服务自行崩溃退出”。

第 26 补丁 `historical-gc-request-errors` 修复的是请求及执行权利的清理路径：

- 统一将显式 `BudgetExceeded` / `ServiceUnavailable` 转为请求的 Unavailable。尚未发送响应头时可返回 503；已经开始的流发送终态错误，不能改写之前的 HTTP 200。
- 区分两类失败：准备阶段仍持有有效 sequence，通过已有有序 abort 清理；Program 的 decode/control/批量 prefill 执行失败已经消费 membership，只结清对应 ResourceManager 记录，不再次 abort 旧 capability。
- 受影响请求退出后保留原等待 FIFO，不因一个请求额度不足锁存整个 Engine。批量 prefill 后续的 `advance_prefill` 仍采用第一类清理，不能把它误认为已消费的批量执行。
- 先完成可能分配内存的错误分类，再移除请求槽位。移除后若状态发布失败，也向已暂存的请求消费者发送终态错误，避免请求丢失其 owner 后永久等待。
- 若仍有未结清的 context transaction，继续升级到共享故障处理，不伪造“已安全释放”。本批没有承诺所有设备或事务错误都可请求级恢复。

没有改动模型数学内核、召回算法或 GPU 工作集大小。变更位于 Engine 错误边界、ResourceManager 清理接口和对应维护说明。

## 验收入口与版本归属

公开驱动为 [ninfer_lifecycle_service_acceptance.py](../scripts/ninfer_lifecycle_service_acceptance.py)，协议负例为 [test_ninfer_lifecycle_service_acceptance.py](../tests/test_ninfer_lifecycle_service_acceptance.py)。驱动只操作显式给出的隔离目录和它创建的容器，固定模型及服务校验值，保存命令、响应、服务日志、额度计数和资源采样。模型目录只读，未改远程原项目或原有服务。

当前候选的原生补丁树为 `1ad96bf9efbcd99f87bf63a36cb4ecf72f85c082`，服务 SHA-256 为 `5ddea137408974d54e13fcbfea9e5b0e59ab33cfc565dee4760f50d8e0537cb0`。第 26 补丁及全部 26 项补丁顺序已经重放校验。可执行契约清单 [service-candidate-v1](contracts/kv-lifecycle-service-candidate-v1.json) 固定 94 个源码文件；它补充原阶段 A 基线，没有把旧二进制证据重新归属给当前版本。

真实模型检查使用已固定 SHA-256 的 Qwen3.8-27B NInfer v3、RTX 3080 Ti Laptop、CUDA 13.2 / sm_86；B=2048、R=512、H=96 MiB，64-token page，rk4v4、Mean-Q cosine/max，传输校验与 GC 审计开启。它是功能压力配置，不能代替阶段 A 的微型性能档位。

独立运行方式如下。`--root` 必须包含已经完成构建的 `native-logs`、对应服务和辅助脚本，输出目录必须不存在：

```sh
python3 scripts/ninfer_lifecycle_service_acceptance.py \
  --root "$built_workspace" --model "$model" --model-sha256 "$model_sha256" \
  --image "$verified_builder_image" --out "$new_output_directory" \
  --cases shutdown-prefill shutdown-decode
```

额度不足使用 `--cases stream-quota --configs redundancy-none redundancy-mtp4`；连续轮换使用 `--cases churn --iterations 12 --records 256 --aux-mib 64`。这些是有限功能检查，没有长时间或性能合格含义。

## 当前结果

最新二进制完成原生构建和 5 项原生测试，Windows Debug 20 项主机契约通过。流协议 6 项测试在 Windows Python 3.14 和 Linux Python 3.11 分别通过，覆盖无成功 `[DONE]` 的合法终态错误、响应头前 503、截断 HTTP、成功不能冒充失败、错误后继续内容及错误码不匹配。

当前真实模型验收全部结束并通过，详见 [执行摘要](experiments/2026-10-31-lifecycle-stage-b-service/summary.json)：

- 活动停止 8 项：LRU/redundancy × 普通/MTP4 × prefill/decode。先观察实际执行阶段，再证明第二个请求已排队；运行及排队请求均收到明确 Unavailable，服务退出 0、未被 OOM kill。不是用空闲服务停止代替活动停止。
- 生成途中额度不足 2 项：redundancy 的普通/MTP4。分别在 2250 / 592 个先行 SSE 帧后出现一次 typed terminal error；此前已排队请求和后续新请求都回答 599。M 峰值分别 8,388,468 / 8,382,888 字节，未超过 8,388,608 字节；请求结束后 active owner 和 payload 预约归零。
- 小额度轮换 12 次：redundancy 的普通/MTP4 各 6 次独立会话，每次 64 条记录、M=10 MiB。全部回答 599，各发生 4 次旧历史驱逐，没有额度拒绝；M 峰值分别 10,414,364 / 10,080,368 字节，低于 10,485,760 字节。H 与 retained count 也在各自额度内。

小额度轮换的进程 RSS 从约 3.15→3.77 GiB、4.10→4.74 GiB，checkpoint 用量各约 1.15 GiB，线程首末均为 30。它表明两份冷域/checkpoint 仍会占用 M/H 外的真实内存，不能声称“总内存只有 H+M”，也不能以这 6 次样本证明长期没有增长。checkpoint/源记录的独立额度是 C/E 阶段的明确工作项。

另一个中间二进制 `32c9c7ed894da728eb1244fc388e5617d7fdbaead74959c6995a7b8f47339b5e` 已完成 LRU/redundancy × 普通/MTP4，各 12 次、共 48 次真实请求。每次独立 session key、256 条记录，均回答 599；各组发生 11 次旧历史驱逐，未使用辅助额度拒绝。M=64 MiB 下，LRU 峰值约 0.26 MiB，redundancy 峰值约 12.9 MiB。它证明对应版本的连续轮换路径，不能写成当前二进制跑了 48 次。

该中间版本四组首末请求的 RSS 分别约 3.05→3.07、3.07→3.08、4.01→4.05、4.03→4.06 GiB，观察到的 RSS 高水位约 3.25–4.24 GiB，线程首末均为 30。RSS 未回到初始值，有限样本不足以证明无长期增长。checkpoint、模型/状态区、框架和驱动保留不由 M/H 覆盖；例如该轮 checkpoint 用量为数百 MiB，不能把它算进 64 MiB 辅助上界。

## 未通过记录与限制

第一次停止总检查错误地要求错误流也带成功的 `[DONE]`，因此判为失败。真实协议以终态 error 和完成的 HTTP chunked body 结束；已修正驱动并加入截断负例，保留最初失败。旧 decode 额度耗尽则是真实产品缺陷，与这个驱动问题分开记录。

Windows 默认执行环境最初拒绝本机回环连接，产生 WinError 10013；允许执行本机协议测试后 6 项通过。未修改主机网络、注册表或驱动。GCC 13 的最小 TSan 探针构建成功，但运行在进入应用前失败，退出 66、`unexpected memory mapping`。这是工具不可用，不能写成 TSan 通过，也不能据此判断应用存在或不存在竞态。

本批未完成长时间、固定工作量性能、Agent 质量或恢复后的任务验证，也没有关闭此前偶发 CUDA 问题。用户“暂不修改主机调试设置”和“DeepSWE 暂缓”继续有效。恢复源记录、checkpoint/源配额、版本/遗忘 fence 和恢复事务在 C 阶段实现；SSD 和多个运行会话仍由 E 阶段单独交付。

原始证据分为三个不可变归档，分别保留旧缺陷及 48 次中间版本轮换、前一候选、当前候选。路径和 SHA-256 逐项列在执行摘要中；当前归档为 `artifacts/lifecycle-stage-b-service/current-evidence.tar.gz`，SHA-256 `e6412468e51668f0a407cb8a3c88bc52f1a2a76c8ea501a8eeed75ff3b490cbb`。归档不含模型或编译二进制。之前 B4 的完整源码/构建/ASan 证据仍保留，公共 GC 核心本批未改动，没有把先前 ASan 写成新 Engine 分支的 sanitizer 证明。

本地汇总包 `artifacts/lifecycle-stage-b-service/evidence.zip` 包含这三个归档、20 项主机原始结果、协议结果、输入与核对脚本，SHA-256 为 `d7ff7ef7b1f2628b8e550cf463e5d50e8025bae6f61f9cf9f4c57c8d5681e1ff`。Git 保存源码、契约、摘要及说明；完整原始包保留在本地 artifacts，不把模型或大体积实验文件加入仓库。

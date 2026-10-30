# 原生服务故障验收

本目录是第 26 补丁的结果索引。问题、修复和限制见 [服务故障报告](../../kv-lifecycle-service-failures.md)，逐项数据见 [summary.json](summary.json)。完整六阶段状态以 [进度记录](../../kv-lifecycle-progress.md) 为准。

当前服务 `5ddea137…` 的结果为：20 项主机契约、5 项原生测试、两环境各 6 项协议测试、8 项活动/排队停止、2 项生成中额度不足，以及 12 次 M=10 MiB 连续轮换全部通过。中间服务 `32c9c7ed…` 的 48 次轮换单独归属；旧服务 `a2a5f55b…` 的真实 Engine 失败和最初驱动误判均保留。

复现主机契约：在本源码配置独立 Debug/CPU-only 构建，按 [基线检查器](../../../scripts/check_kv_lifecycle_baseline.py) 的参数说明传入 `docs/contracts/kv-lifecycle-service-candidate-v1.json` 和 build-dir。协议测试入口：

```sh
python3 -m unittest discover -s tests -p test_ninfer_lifecycle_service_acceptance.py
```

真实服务检查通过 [公开驱动](../../../scripts/ninfer_lifecycle_service_acceptance.py) 指向隔离的已构建目录，固定模型和工具镜像；具体命令和结果文件保存在原始归档。驱动记录完整 argv，不从本摘要反推默认配置。测试输出目录必须新建，不能覆盖一次失败重跑。

本地 `artifacts/lifecycle-stage-b-service/evidence.zip` 中的 `analyze.py` 只读取归档，验证源码指纹、构建完成、模型/二进制身份、每项实际结果、正确答案、排队条件和资源边界，重新生成本摘要。它没有将工具不可用的 TSan 或未运行的生产/性能/Agent 检查计为通过。包内 `verification-inputs.json` 保存输入文件校验值。

M/H 是分资源额度。当前小额度轮换的 checkpoint 约 1.15 GiB，RSS 明显高于 M+H。有限运行不能证明长期内存稳定；未关闭的 CUDA 问题、TSan 环境限制、C–F 阶段工作仍在进度记录中跟踪。

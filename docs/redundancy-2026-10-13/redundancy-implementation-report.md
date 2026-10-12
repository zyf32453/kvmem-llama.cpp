# KV-Mem Redundancy 计算链路修复报告

**日期：** 2026-10-13
**分支/基线：** `redundancy` / `da036fbb15b25f97a3ebf47cb3cf8ac673ac5d71`

## 摘要

本次检查确认，生产推理中的 redundancy 评分链已经存在：`preview_retrieval()` 在候选选择前调用 `score_retrieval()`，将评分写入 `KvMemStore`，再执行 `preview_reselect()`。此前根据 `redundancy.hpp` 辅助函数调用点作出的“生产链路缺失”判断不完整；旧实现曾把 descriptor 投影和 cosine 计算内联在 `score_retrieval()` 中。

历史实验记录只有 `eligible_blocks=50` 和 `rejected_candidates=0`，未记录有效 descriptor 数、比较次数或分数分布，因此不能据此断定评分没有运行，也无法解释没有拒绝候选的具体原因。

## 本次修复

1. **校验 descriptor 输入。** 对非有限分量、溢出范数和零范数分别处理；无效 descriptor 不参与 cosine 比较。
2. **校验 block 完整性。** 只使用完整 block，并要求每个活动 KV 层都有覆盖该 block 的 mean-K 数据；数据缺失或不完整时不生成有效分数。
3. **区分无比较与真实 cosine -1。** 没有可比较的后续 block 时保持 `-2`（未评估）；真实反向 descriptor 的 `-1` 仍是有效分数。
4. **保持分数与 block 生命周期一致。** append、truncate、mean-K 恢复或重新捕获后以 O(1) 标记当前分数失效；选择器忽略失效分数。只改变 GPU/Host tier 而未改变 descriptor 的恢复仍保留有效分数。
5. **增加诊断。** 调试日志可报告 descriptor 尝试/有效数量、缺失层数、比较数、cosine 分布、block 分数 min/max/mean、达到阈值数量和消费端实际拒绝数。诊断由 `KVMEM_REDUNDANCY_DIAG=1` 启用；常规汇总也可由 `KVMEM_TRACE=1` 观察。

## 调用链

真实 Key 数据由 `harvest_from_host()` / `harvest_capture()` 写入 `RawKvStore`；decode 路径由 `decode_mean_flush()` 保存每层 mean-K。检索阶段执行 `harvest_flush()` 后，`score_retrieval()` 从活动 KV 层读取完整 block mean-K，利用现有 64 维投影 helper 生成归一化 descriptor，对后续候选计算 cosine 最大值，并调用 `set_redundancy_scores()`。随后 `preview_reselect()` 才进行候选过滤与排序。

评分仍只比较后续 block（局部窗口加最新窗口），并保持原策略：高相似旧 block 可从历史检索工作集中排除或降低排名；sink、recent 和 mandatory block 继续按原规则保护。没有更改阈值，也没有把 redundancy 逻辑扩展到独立的 prefill pressure offload 决策。

## 修改文件

- `kvmem/include/kvmem/redundancy.hpp`：有限值处理、cosine 校验和无后续比较的 sentinel 语义。
- `kvmem/include/kvmem/raw_kv_store.hpp`、`kvmem/src/host/raw_kv_store.cpp`：增加 per-layer mean-K 覆盖 token 数查询。
- `kvmem/include/kvmem/kvmem_store.hpp`、`kvmem/src/host/kvmem_store.cpp`：分数 freshness、非法 score 清理、生命周期失效和拒绝数诊断。
- `src/adapter/llama-memory-kvmem.cpp`：完整 mean-K 校验、helper 接入、score 刷新与失效挂钩、诊断统计。
- `kvmem/tests/kvmem_store_test.cpp`、`kvmem/tests/raw_kv_store_test.cpp`：descriptor/cosine、比较边界、生命周期、覆盖 token 数和 selector 开关对照测试。

## 验证结果

以下 host 测试程序运行成功，退出码均为 0：

- `kvmem_store_test.exe`（输出 `OK`）
- `raw_kv_store_test.exe`
- `backend_rebind_test.exe`
- `pinned_kv_tier_test.exe`（输出 `OK`）
- `git diff --check`

本地 CMake 配置为 `KVMEM_BUILD_LLAMA=OFF`；CUDA adapter、`llama-kvmem-server` 和真实模型推理未在本机编译或运行。当前本机缺少 `nvcc`，用户提供的服务器日志还显示 `cmake` 命令不可用。因此，本报告不声称端到端 CUDA 验证通过。历史实验日志不是本次运行产生的完整日志。

## 性能与限制

每个 block 的 64 维 float descriptor 约占 256 字节；每个 eligible block 最多与 128 个后续 block 做 64 维 cosine 运算。descriptor 使用现有 host mean-K，不新增 scorer 专属 GPU→CPU 拷贝或显存分配。诊断开启时会额外保存比较样本并输出统计；尚未在目标 CUDA 模型上测量实际推理延迟。

下一步需将本次改动构建到 CUDA server，在相同模型和请求下启用 `KVMEM_REDUNDANCY_DIAG=1`，采集完整 scorer 与 selector 日志，以确认实际 descriptor 完整率、分数分布和拒绝结果。

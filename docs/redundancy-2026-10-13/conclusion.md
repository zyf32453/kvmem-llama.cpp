# 结论与下一阶段

## 已证实

- 提交 `a39d07aa` 的检测路径在导师机上执行过：1 个长度足够的 thinking span、50 个 32-token 合格块、`query_replay`、prompt 1687、预算 1024。
- 阈值 0.90 和 0.50 下都没有块的冗余分数达到阈值。hard reject 因此没有删除块。
- 选择列表 `0, 22..52` 正好是 32 个块，等于 1024/32。缺的 1..21 是预算竞争的结果。
- 相同 mean-K 的 64 维投影余弦为 1。这个性质有主机测试，不依赖模型。
- 对 53 块的历史，比较窗口覆盖每一个更晚的块。窗口漏配不能解释这次的 0，除非描述符无效、比较根本没发生。
- 低于阈值的已评估分数仍可能通过 novelty 改变 ON/OFF 的选择。这一点有单元测试，没有这次请求的 OFF 对照。

## 未证实

- 50 个块的最大余弦具体是多少。
- 比较次数是 1225 还是 0。这是 H1 与 H2 的分界。
- 合格块里的 token 是否就是实验者以为的重复句。
- Redundancy 是否改变了这次 GPU 工作集。没有 `KVMEM_REDUNDANCY=0` 的 selected 列表。
- 性能、显存和回答质量。对应验收等级 L6，本阶段没有做。

## 验收等级

| 等级 | 状态 |
|---|---|
| L1 服务与 API | 此前已完成，本机未复跑 |
| L2 检测执行 | 此前已完成 |
| L3 可解释的分数分布 | 未完成。诊断代码已写入工作区，尚未在 GPU 二进制中运行 |
| L4 超过阈值的候选 | 未出现。现有证据下这是分数低于 0.50，或比较次数为 0 |
| L5 工作集因此改变 | 未证实 |
| L6 性能与质量 | 未开展 |

## 不要做的事

不要降低默认阈值，不要改 64 维投影、`importance_weight` 或 sink/mandatory/recent 保护来制造非零拒绝数。现有日志已经说明选择器没有收到达到 0.50 的分数。

## 下一阶段只做一次对照

在导师机的 `/home/yifan/kvmem-llama.cpp` 上先确认 `git rev-parse HEAD` 和工作区差异，再把本机这批诊断改动同步过去。只重编译 `llama-kvmem-server`。不要同时改 prompt、预算、block size。请求体继续用 `/tmp/kvmem_redundancy_probe.json`。三次实验顺序执行，避免两份模型抢 16 GiB 显存。

```bash
export KVMEM_TRACE=1
export KVMEM_REDUNDANCY_DIAG=1
export KVMEM_REDUNDANCY_DIAG_FILE=/tmp/redundancy_on_090.log
# 第一次不要设置 KVMEM_REDUNDANCY，阈值保持默认 0.90
# 第二次只增加 KVMEM_REDUNDANCY_THRESHOLD=0.50，并换一个日志文件
# 第三次只设置 KVMEM_REDUNDANCY=0，并换一个日志文件
```

启动参数保持报告中的那一组：`-c 4096 -ngl 8 --kvmem-budget 1024 --kvmem-gen-reserve 1536 --kvmem-block-tokens 32 --kvmem-method retrieval --kvmem-trace`。stderr 同时重定向到同一个日志文件。

判定：

- `reason=no_similarity_comparisons`：描述符或 mean-K 没有形成比较。先查 `zero_norm`、`missing_mean`、`non_finite_input`，不要改相似度公式。
- `comparisons=1225` 且 `reason=max_block_score_below_threshold`：H1 成立。重复文本的 pre-RoPE mean-K 没有接近到 0.50。这是特征限制，不是执行错误。
- `reason=candidates_meet_threshold`：再比较 ON/OFF 的 `KVMEM_TRACE selected`，并用 `hard_rejected` 与 `budget_excluded` 分开 hard reject 和 novelty 重排。

只有第三种结果出现后，才进入 L5。L6 放在 L5 之后。

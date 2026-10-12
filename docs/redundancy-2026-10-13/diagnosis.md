# Redundancy 诊断

日期：2026-10-13  
核对提交：`a39d07aa6db741fe6db447644d2ef06d526ed8d3`  
分支：`redundancy`，跟踪 `origin/redundancy`

## 结论

已有实验日志证明：redundancy 检测执行了，50 个合格块的冗余分数都低于 0.50，因此 hard reject 没有排除任何块。

它还不能证明描述符算错，也不能证明比较窗口漏掉了重复块。对这次 1687-token 的请求，块数是 53，比较窗口是后 64 块；只要描述符有效，合格块之间就是向前的全配对。相同的 mean-K 投影余弦是 1。所以 `rejected_candidates=0` 只剩下两类尚未用运行日志分开的原因：

1. 这些块的 pre-RoPE mean-K 投影余弦确实低于 0.50。
2. 描述符无效，比较次数为 0，分数停留在哨兵值 `-2`，同样不会达到 0.50。

这不是选择器 bug。阈值 0.50 的实验没有给选择器任何可拒绝的分数。不要为了制造非零 `rejected_candidates` 去改投影或默认阈值。

## 版本

| 项目 | 结果 |
|---|---|
| HEAD | `a39d07aa6db741fe6db447644d2ef06d526ed8d3`，2026-10-11，`feat(kvmem): skip redundant blocks in long thinking traces` |
| 跟踪文件相对该提交 | 有未提交诊断改动，见 `changes.diff` |
| 破坏性操作 | 未执行 `reset`、`clean`、未重新编译 CUDA server |
| 导师机二进制 | 本机只有 `docker-desktop` WSL，没有 `/home/yifan/kvmem-llama.cpp`，无法核对正在运行的可执行文件 |

报告中的这一行与提交里的格式一致，而与本次新增的 `KVMEM_REDUNDANCY_DIAG` 无关：

`KVMEM_REDUNDANCY enabled=1 eligible_blocks=50 thinking_spans=1 rejected_candidates=0 threshold=0.900 lambda=0.100 dims=64`

因此那次进程跑的是提交中的计分逻辑，不是本工作区里尚未编译进去的诊断代码。

## 这次日志已经能确定的事

配置是 `-c 4096`、`--kvmem-budget 1024`、`--kvmem-block-tokens 32`、retrieval、redundancy 开启。prompt 1687 tokens，大于 1024 的选择预算。`query_replay` 说明没有走 `all_resident` 或 `keep_selected`。

1687 / 32 = 52 个完整块加 1 个 23-token 尾块，共 53 块，编号 0..52。预算 1024 / 32 = 32 块。记录中的选择是：

`0 22 23 ... 52`

这是 32 个块。被留下的是 sink 块 0 和 22..52；没有进入工作集的是 1..21。

`rejected_candidates` 的定义是 `redundancy_score >= threshold` 的块数。阈值 0.90 和 0.50 都是 0，所以每个块的分数都小于 0.50。`redundancy_rejected()` 要求分数同时 `>= -1` 和 `>= threshold`。`-2` 和小于 0.50 的分数都不会被 hard reject。块 1..21 是 top-k 名额竞争中没被选中，不是因为超过 0.50 被拒绝。

50 个合格块占 1600 tokens。整段 prompt 只有 1687 tokens，所以重复内容如果确实构成了这次长 prompt，它就在这个 thinking span 里。span 之外最多约两个完整块加那个 23-token 尾块。

## 假设

### H1：余弦确实低于 0.50

这是当前最符合代码的解释，但还缺分布日志。

冗余特征不是文本，也不是 RoPE 之后的 K。捕获日志把这份 K 标成 pre-RoPE（`KVMEM_DUMP k_prerope`，`src/adapter/llama-memory-kvmem.cpp` 中 harvest 路径）。pre-RoPE K 仍由因果隐藏状态决定。同一句话出现在不同位置时，mean-K 可以差很多。

投影是线性哈希。两份完全相同的 mean-K 得到余弦 1，主机测试 `test_redundancy_descriptor_math` 已通过。因此只要有一对合格块的 mean-K 相同，0.50 阈值就不可能得到 `rejected_candidates=0`。现有结果说明合格块的 mean-K 并不相同，或者描述符根本没建成。

### H2：描述符无效

若 mean-K 缺失或范数不超过 `1e-8`，该块保持 `-2`，不参与点积。`-2 >= 0.50` 为假，`rejected_candidates` 同样是 0。旧日志没有 `valid`、`zero_norm`、`comparisons`，所以 H2 不能排除。

NaN 不会把一个真实的高余弦盖掉：`best` 初值是 `-1`，`std::max(-1, NaN)` 在 `!(a < b)` 的实现下仍返回 `-1`。Inf 在原实现里会通过 `norm > 1e-8f` 并做除法。这都不能单独解释“有高相似对但计数仍为 0”。

### H3：比较窗口没覆盖重复区域

对这次请求不成立，前提是描述符有效。

`redundancy_visit_partners()` 只看后 64 个块，以及落在这个窗口外的最新 64 个块。53 小于 65，所以块 0 的伙伴是 1..52，也就是全部更晚的块。`test_redundancy_partner_window` 固定了这个几何关系。50 个有效描述符应产生 `50 * 49 / 2 = 1225` 次比较。只有描述符无效时，比较次数才会是 0。

更长的历史才会漏掉中间块。例如 200 块时，块 0 不与块 100 比较。那不是这次 53 块实验的情况。

### H4：模板后的 token 与预期文本不一致

不能完全关闭。旧日志没有逐块 token 范围。不过 1600 个合格 token 已经占满 1687 里的绝大部分，模板不太可能把重复正文裁掉后仍留下 50 个合格块。下一步诊断会打印每个 span 的 `[begin,end)` 和每个合格块的 token 区间。

### H5：分数改变了排序，但没有 hard reject

仍然开放。分数在 `[-1, 0.50)` 的块不会被拒绝，但会按 `importance_weight=0.10` 做 novelty 重排。`test_redundancy_novelty_ranking_without_hard_reject` 证明 0.80 的分数在 0.90 阈值下不拒绝，却能让 ON/OFF 选出不同的块。报告里没有 `KVMEM_REDUNDANCY=0` 的 selected 列表，所以不能用 `0 22..52` 证明或否定这种重排。

## 代码位置

- 投影、归一化、余弦、伙伴窗口：`kvmem/include/kvmem/redundancy.hpp`
- 资格、打分、诊断：`llama_memory_kvmem::score_retrieval()`，`src/adapter/llama-memory-kvmem.cpp`
- hard reject 与 novelty 排序：`redundancy_rejected()`、`redundancy_rank()`、`pick_topk_ungrouped()`，`kvmem/src/host/kvmem_store.cpp`
- 预算足够时直接返回全部块，不应用 reject：`pick_topk_ungrouped()` 中 `n <= budget` 的分支。这次 53 > 32，没有走该分支。

诊断默认关闭。设置 `KVMEM_REDUNDANCY_DIAG=1` 后写入 stderr；`KVMEM_REDUNDANCY_DIAG_FILE` 再追加同一份文本。未设置时选择结果与原来相同。

下一次运行用这一行区分 H1 和 H2：

`KVMEM_REDUNDANCY_DIAG stage=reject_reason ... reason=...`

| `reason` | 含义 |
|---|---|
| `no_similarity_comparisons` | H2 或没有可配对的有效描述符。同时看 `zero_norm`、`missing_mean`、`invalid_non_finite` |
| `max_block_score_below_threshold` 且 `comparisons=1225` | H1。余弦算过，最大值低于阈值 |
| `candidates_meet_threshold` | 已经有可拒绝块。接着看 `KVMEM_REDUNDANCY_SELECT` 里的 `hard_rejected` 和 `budget_excluded` |

`comparisons` 明显小于 1225 且不为 0，表示只有一部分描述符有效。

## 测试

`build-local-test/kvmem_store_test.exe` 已用 MinGW g++ 16.2.0 编译并输出 `OK`。覆盖：

- 相同、正交、相反、零、NaN、Inf 描述符
- 相同 mean-K 投影余弦为 1
- 53 块窗口全覆盖，200 块时中间块不在窗口内
- 0.49 不 hard reject，0.50 hard reject
- 提交中的原用例：高 retrieval、冗余 0.95 的块被排除，sink 与 recent 保留
- mandatory 保护、预算足够时不丢块、ON/OFF 差异、低于阈值的 novelty 重排

这些测试不加载模型，也不证明真实 mean-K 的分布。

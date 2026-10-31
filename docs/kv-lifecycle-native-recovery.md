# NInfer 源记录的新域重建

日期：2026-10-31。C3 的规范前缀新域重建已通过四组原生模型功能检查；完整阶段状态见 [六阶段进度](kv-lifecycle-progress.md)。这不代表完整生产 GC、持久恢复或任意旧 KV 的精确恢复。

## 使用与资源边界

在已支持的单 GPU、单运行请求、文本/RAM KVMem 上，同时配置正数 `--kvmem-source-mib` 和 `--kvmem-recovery-mib`。两项默认关闭。恢复准备额度按 bytes 记账，独立于源 S、历史 H、GC 辅助 M，以及按 tokens 计算的 generation reserve。

公开的 in-tree C++ 接口包括：

- `GenerationResult::accepted_source`：本次请求实际记录的源版本；取消时也可能指向已接纳但只部分计算的输入。它不是 KV 的复用许可。
- `Engine::historical_sources()`：有界源目录及其明确分支父来源、前缀长度、accepted/computed 边界和当前保留窗口观测。
- `Engine::prepare_source_rebuild(receipt, prefix_tokens, output_reserve_tokens)`：读取源的 `[0,prefix_tokens)`，准备普通 raw-token 请求；随后通过正常的 `submit` / `generate` 预约并执行模型。

源编号从每个 RequestRecord 的稳定提交事实传到该请求的结果，不能从全局 `source_last_id` 猜测，避免排队请求接错来源。回执仅限本进程/模型实例；尚未接入 HTTP 接口或跨进程恢复。

规范路径要求完整前缀及独立向上取整的生成/草稿空间能装入预算。分块 prefill 不会绕过完整容量检查。超出完整窗口的长源明确拒绝，当前没有稀疏源重放的隐式回退。

## 所有权与提交

沿用 [源规划/准备组件](kv-lifecycle-source-recovery.md)、现有 Window 身份分配器和 Program 的实际 main/draft/state 资源规划，不建立另一套 KV 目录、allocator 或执行线程。

源可以比旧 KV 窗口活得更久。因此，从源创建新域时不要求旧窗口仍然存在；只有替换既有域的计划才需要检查其当前 MemoryStamp。SourceRef 的 store/revision/configuration 和局部遗忘校验始终有效。NInfer 的源存储属于不可变模型 Program，其他实例的回执首先被 incarnation 拒绝。

准备在 Engine 单元之间串行进行，先预约完整源副本空间，再复制。Frontend 使用实际接受的 IDs 构造线性位置，不重新套 chat template。普通 prepared-input 向量属于另计的请求工作区；进入 prefill 后，源分支还会单独占 S。这些开销未合并成进程 RSS 上界。

准入强制空 root，关闭旧 native prefix/historical checkpoint 的引用；实际模型资源仍由既有 ResourceManager/Program 预约。首次 prefill 前建立独立源分支和新域身份；每个 prefill 单元重新校验源，已有父源不被消费或修改。

成功的完整 Program commit 才完成恢复准备并归还其额度。恢复未完成时，包括部分 prefill 取消，不能把私有中间 checkpoint 放进可复用历史；清理继续使用原有有序 abort / consumed-membership 机制。新域之后的保留遵循既有缓存规则，旧退休身份不重新取得资格。

## 验收内容

机器可读候选为 [native-recovery-candidate-v1](contracts/kv-lifecycle-native-recovery-candidate-v1.json)，真实模型入口为 [ninfer_source_recovery_acceptance.py](../scripts/ninfer_source_recovery_acceptance.py)。受影响原生测试包含新的独立 C++ 模型测试和公共头边界检查。

真实模型检查要求：实际释放的旧 CPU 页位于重建前缀内；重建与独立原始 IDs 请求对照；真实 prefill 取消后不保留半成品；准备额度耗尽/超容量/请求 reserve 不匹配明确拒绝；首 token 停止也正确结清；旧窗口确实被换出后仍可仅从源重建。MTP 配置必须实际执行投机轮，不能只靠配置名称宣称覆盖。

最终候选为原生 tree `d508d9b98aff4606f415777344d2f6ad12182df8`，28 个有序补丁，冻结 104 项输入。独立模型测试二进制 SHA-256 为 `4b4167ce3ffad6dd110e8ee650665290c09027bb65857dc2cc6f1d6a7ce4f46c`；同次构建的服务二进制为 `d446ffdb2ef5e2b474fc8210442e4141b11424402d2d4b2ff6ee3f8bcce21094`，但本次没有以服务 HTTP 请求替代 C++ Engine 检查。完整机器可读结果见 [执行摘要](experiments/2026-10-31-lifecycle-stage-c-native-recovery/summary.json)。

- Windows/GCC 16.2 Debug 的 22 项主机契约通过。该次运行后仅修改一个原生 CUDA options 头的直接依赖；已逐项核对全部冻结核心/测试源码未变，复用其有效覆盖，不声称重新运行了 22 项。
- 最终源码的 9 项原生检查全部通过，包括 frontend、CLI、serving options、源记录/准备与公共头边界。Linux/GCC 13.3 的三项核心测试在实际链接的 ASan/UBSan/LeakSanitizer 下全部通过；不包含 TSan 或 GPU 内存检查器结论。
- 同一 Qwen3.8-27B v3 模型、RTX 3080 Ti Laptop 16 GiB / 驱动 596.08，LRU／redundancy × 普通解码／MTP4 共四组全部 exit 0，未 OOM。每组输入为 32,768 个实际 token，重建其中前 1,024 个；B=2,048 tokens、生成预留 512、H=96 MiB、S=16 MiB、R=8,192 bytes、prefill=512、64-token 页、rk4v4、greedy seed 42、context=65,536。
- 每组在首次重建前，都审计到重建前缀内 **15 个 64-token CPU 页的真实释放**。普通解码初始退休 476 页，MTP 初始退休 480 页；这两个数是退休观察，不等同于整批物理释放量。
- 每组部分取消都实际完成了 512 个 prefill tokens 后退出；R 归零，取消回执保留 accepted=1,920 / computed=512 的独立源分支，未完成域未进入历史缓存。两份 1,024-token 准备副本实际占满 8,192-byte R，第三份拒绝；全部销毁后退款。旧 revision、4,096-token 超容量、准入输出 reserve 不匹配也明确拒绝。
- 四组均比较规范重建、禁用前缀复用的独立原始 IDs 请求、旧 native window 被实际换出后的源重建，三者 8 个 greedy 输出 IDs 完全相同。两组 MTP 的原请求、重建和源重建各实际执行 2 轮投机。每组还覆盖首 token 停止、稳定的请求级源回执和父分支关系；四次新域开始仅三次完成，部分取消没有完成发布。这个短且重复的输入证明功能路径，不代表长任务质量。

早期尝试保留并分别归属版本：首次构建遗漏容器 CUDA toolkit 挂载；首轮 C++ 测试漏关与 KVMem 不兼容的 native context cache；停止断言误把 raw 输出当作隐藏停止 token 的 decoded 输出。分别修正运行工具/测试配置和断言，没有修改产品数学或停止规则。实际公共头检查另暴露继承的 umbrella include 依赖，改为定义头后最终九项通过。加入请求级回执前的单组 LRU 成功与中间二进制结果均未记为最终候选通过。

## 剩余范围

源仍按请求和显式分支保存，包含后来失败/取消的输入；没有跨请求去重或自动过期。后续已提供 [C4 的显式进程内家族遗忘](kv-lifecycle-native-forget.md)，仍不构成跨进程/归档的系统级遗忘。目录和 S 满时会明确拒绝，内存压力 GC 不隐式遗忘源。

当前重建是新域的规范 raw-token 继续计算，不承诺重现旧稀疏/recurrent 执行轨迹。HTTP 接线、长源稀疏重放、checkpoint/归档的退休和遗忘 fence、持久恢复及完整发布/回滚故障资格仍需完成。配对性能、长时间、新 Agent 质量和已有 CUDA/TSan 未关闭项继续单列；不能因此关闭阶段 C 或完整六阶段目标。

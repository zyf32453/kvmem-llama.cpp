# Historical KV 后台 GC：导师演示版

这是 2026-10-18 的独立机制演示记录，下面的阶段计划仅描述当时状态。
当前混合模型、32-token 块与 redundancy 验收进展见 [参数复测记录](background-gc-parameter-retest.md)。

更新日期：2026-10-18。当前阶段目标调整为一个可运行、可解释的 CPU GC 原型。
原来的“整套生产 GC”长期目标继续暂停；本次不以完成生产验收为交付条件。

## 本次交付

复用项目现有的 `HistoricalKvLruPolicy`、`HistoricalKvGc`、
`HistoricalKvRepository` 和 `HistoricalKvRawStore`，提供一个独立的小程序。
使用人工构造的 K/V 字节，运行真实的 CPU 存储分配与后台 GC 线程；无需 GPU、SSH 或语言模型文件。
这是 GC 机制原型，并非小参数语言模型的推理演示。

简化的是演示范围：只用 LRU、一层、三个等大小 CPU 历史块。
策略只提出删除建议；回收模块负责关闭新使用、等待已有读取完成、释放内存和复用位置。
已有的生产扩展代码保留，本次没有重新实现一套替代 GC。

## 演示过程

1. 存入 A、B、C，占满三个位置。最近使用序号分别为 30、20、10，所以 C 最久未用，B 次之。C 仍有一个尚未完成的读取。此时尝试放入第四块会立即失败。
2. 后台 LRU 在空闲位置少于 1 时触发，目标是恢复到 2 个空闲位置，选择 C、B。
   GC 关闭这两块的新使用权限，因此下一轮候选只剩 A。
3. B 没有人读取，后台立即释放其 CPU 数据。C 虽已退出候选，旧读取仍能读到原来的 K/V；不会提前释放。
4. C 的旧读取结束，后台释放 C。新块 D 可以使用回收的位置，并获得新的代次编号，旧编号不能访问新数据。
5. 演示结束时显式关闭剩余块，后台完成清理。

为了能逐步展示，小程序在阶段之间暂停、再启动 GC 线程；所有存储释放回调都检查必须发生在后台线程。
本次实测的 CPU payload 字节数依次为 **24 → 16 → 8 → 16 → 0**。
这是存储对象实际持有的 K/V 分配量，不是整个进程的 RSS；释放是归还给内存分配器。
“下一轮候选”在这里通过仓库准入接口验证，没有运行模型的 active selection 或 GPU attention。

## 怎么运行

在本项目根目录打开 PowerShell：

```powershell
.\scripts\run_historical_gc_demo.ps1
```

脚本复用已有的 `build-local-active`，只构建演示目标，然后运行；本机已有所需编译器。
成功时最后一行是：

```text
PASS: LRU -> retire -> drain existing reader -> free CPU payload -> reuse slot.
```

其他已配置 C++17 编译器和 CMake 的环境可以使用：

```sh
cmake -S . -B build-gc-demo -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_ENABLE_NVME=OFF
cmake --build build-gc-demo --target historical_kv_gc_demo
ctest --test-dir build-gc-demo -R '^historical_kv_gc_demo$' --output-on-failure -V
```

## 已完成的验证与边界

2026-10-18 在本机 Windows / MinGW 下编译并运行：演示、已有 GC 测试、已有策略测试共 **3/3 通过**。
演示包含可执行断言：LRU 选对旧块；新使用被拒绝；旧读取的数据完整；CPU 数据延后释放；位置安全复用；释放发生在后台；退出时数据占用归零。
只要有一项不满足，程序就返回失败。

这一阶段完成到上述演示与测试为止。暂不继续扩展 pinned memory、NVMe、多 GPU、多并发、MTP 或 redundancy 策略，也不启动大模型长测。
本结果不等于生产完成，不证明真实模型的输出质量、吞吐或长期 RSS 上界。
导师确认核心流程后，下一阶段优先接一个明确的实际推理配置，验证效果，再按需求扩展。

## 可直接向导师说明

> 我们先做了一个最小的 CPU 历史 KV 后台 GC 原型，复用项目里的策略和回收模块。
> LRU 只负责决定删除哪些块；GC 先使这些块退出后续候选，再等已有读取结束后释放 CPU 存储。
> 演示已经验证了安全延迟释放、位置复用和后台执行。当前先确认机制设计，后续再做真实推理与生产扩展。

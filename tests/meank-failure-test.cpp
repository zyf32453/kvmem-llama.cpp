#include "llama-kvmem-stagein.h"
#include "llama-kvmem-hooks.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Fn> static void rejected(Fn fn, const char * message) {
    bool threw = false;
    try { fn(); } catch (const std::exception &) { threw = true; }
    require(threw, message);
}
using Point = KvmemMeanKPoint;
struct Fault {
    Point point;
    uint32_t layer = 0;
    bool fired = false, deny_drain = false;
    unsigned drain_calls = 0;
    static bool inject(void * context, Point point, uint32_t layer) noexcept {
        auto & f = *static_cast<Fault *>(context);
        if (point == Point::Drain) { ++f.drain_calls; return f.deny_drain; }
        if (!f.fired && point == f.point && layer == f.layer) { f.fired = true; return true; }
        return false;
    }
};
struct Fixture {
    static constexpr uint32_t dim = 64, tokens = 8;
    llama_kvmem_execution_state * domain = llama_kvmem_execution_create();
    llama_kvmem_execution_state * previous = llama_kvmem_execution_exchange(domain);
    float * source = nullptr;
    std::vector<float> data = std::vector<float>(dim * tokens);
    Fixture() {
        require(cudaMalloc(reinterpret_cast<void **>(&source), data.size() * sizeof(float)) == cudaSuccess,
                "Mean-K source allocation failed");
        fill(1);
    }
    ~Fixture() {
        kvmem_meank_test_fault(nullptr, nullptr);
        try { kvmem_meank_free(); } catch (...) { std::abort(); }
        if (source) cudaFree(source);
        llama_kvmem_execution_exchange(previous);
        llama_kvmem_execution_free(domain);
    }
    void fill(float base) {
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t d = 0; d < dim; ++d) data[t * dim + d] = base + t + float(d % 8) / 8;
        require(cudaMemcpy(source, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
                "Mean-K source upload failed");
    }
    bool add(uint32_t layer, uint32_t begin = 0, uint32_t count = 3) {
        return kvmem_meank_add(layer, GGML_TYPE_F32, source, begin, count, dim, dim,
                              sizeof(float), dim * sizeof(float), dim * sizeof(float));
    }
    void fresh() {
        kvmem_meank_test_fault(nullptr, nullptr);
        kvmem_meank_free();
        require(kvmem_meank_ready(2, dim), "fresh Mean-K allocation failed");
        std::vector<float> out(dim, -1);
        require(kvmem_meank_d2h(0, out.data(), dim), "fresh Mean-K read failed");
        require(std::all_of(out.begin(), out.end(), [](float v) { return v == 0; }), "fresh sum retained failed data");
    }
};

static void normal_accepted_prefix() {
    Fixture f;
    require(kvmem_meank_ready(2, Fixture::dim), "Mean-K initialization failed");
    require(f.add(0, 1, 3) && f.add(1, 1, 3), "Mean-K prefix accumulation failed");
    kvmem_meank_sync();
    f.fill(20); // Source may be reused only after the accepted kernels finish.
    std::vector<float> out(Fixture::dim, -1);
    require(kvmem_meank_d2h(0, out.data(), Fixture::dim), "Mean-K prefix read failed");
    for (uint32_t d = 0; d < Fixture::dim; ++d)
        require(out[d] == 9 + 3 * float(d % 8) / 8, "Mean-K included rejected tokens or reused source data");
    kvmem_meank_zero(0);
    require(f.add(0, 0, 2), "Mean-K addition after zero failed");
    require(kvmem_meank_d2h(0, out.data(), Fixture::dim), "Mean-K replacement read failed");
    for (uint32_t d = 0; d < Fixture::dim; ++d)
        require(out[d] == 41 + 2 * float(d % 8) / 8, "Mean-K zero did not replace the old sum");
    require(kvmem_meank_d2h(1, out.data(), Fixture::dim), "independent Mean-K layer read failed");
    for (uint32_t d = 0; d < Fixture::dim; ++d)
        require(out[d] == 9 + 3 * float(d % 8) / 8, "zero changed another layer's sum");
    std::puts("PASS Mean-K accepted slices: exact sums, source reuse after completion, layer-local zero");
}

static void operation_failure(Point point, bool initial = false, bool deny_drain = false) {
    Fixture f;
    if (!initial) {
        require(kvmem_meank_ready(2, Fixture::dim), "Mean-K fault fixture initialization failed");
        require(f.add(0), "Mean-K fault fixture sum failed");
        kvmem_meank_sync();
    }
    Fault fault{point};
    fault.deny_drain = deny_drain;
    kvmem_meank_test_fault(Fault::inject, &fault);
    std::vector<float> out(Fixture::dim, -999);
    rejected([&] {
        if (initial) kvmem_meank_ready(2, Fixture::dim);
        else if (point == Point::Zero) kvmem_meank_zero(0);
        else if (point == Point::AddResult) f.add(0);
        else kvmem_meank_d2h(0, out.data(), Fixture::dim);
    }, "CUDA Mean-K failure allowed success/CPU replay");
    require(fault.fired, "Mean-K fault point not reached");
    rejected([&] { kvmem_meank_ready(2, Fixture::dim); }, "failed sum reused without cancellation");
    rejected([&] { kvmem_meank_ready(3, Fixture::dim); }, "failed sum resized before cancellation");
    rejected([&] { f.add(1); }, "failed sum admitted new kernel");
    rejected([&] { kvmem_meank_zero(0); }, "failed sum silently zeroed");
    rejected([&] { kvmem_meank_d2h(0, out.data(), Fixture::dim); }, "failed sum was exposed to host");
    if (deny_drain) {
        rejected([&] { kvmem_meank_cancel(); }, "unconfirmed Mean-K task was cancelled");
        rejected([&] { kvmem_meank_free(); }, "unconfirmed Mean-K storage was freed");
        rejected([&] { kvmem_stagein_gpu_free(); }, "staging free bypassed Mean-K ownership");
        require(fault.drain_calls == 3, "Mean-K cleanup bypassed checked drain");
    }
    kvmem_meank_test_fault(nullptr, nullptr);
    kvmem_meank_cancel();
    rejected([&] { f.add(0); }, "cancellation reopened the failed sum");
    require(std::all_of(out.begin(), out.end(), [](float v) { return v == -999; }),
            "unconfirmed DMA wrote into caller-owned host memory");
    f.fresh();
    std::printf("PASS Mean-K failure point=%u initial=%d undrained=%d: closed, caller unchanged, owned DMA retained, fresh allocation clean\n",
                unsigned(point), int(initial), int(deny_drain));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        require(cudaSetDevice(0) == cudaSuccess, "Mean-K device selection failed");
        normal_accepted_prefix();
        operation_failure(Point::Zero, true);
        operation_failure(Point::Zero);
        operation_failure(Point::AddResult);
        operation_failure(Point::ReadCopy);
        operation_failure(Point::Wait);
        operation_failure(Point::Wait, false, true);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL Mean-K errors: %s\n", e.what());
        return 1;
    }
}

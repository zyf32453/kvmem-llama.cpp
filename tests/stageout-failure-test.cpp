#include "llama-kvmem-stagein.h"
#include "llama-kvmem-hooks.h"
#include <cuda_runtime_api.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
using Point = KvmemStageoutPoint;
struct Fault {
    Point point = Point::EventRecord;
    size_t index = 0;
    bool direct = false, undrained = false, fired = false;
    std::array<unsigned, 7> calls{};
    static bool inject(void * context, Point point, size_t index) noexcept {
        auto & f = *static_cast<Fault *>(context);
        ++f.calls[size_t(point)];
        if (point == Point::Drain) return f.undrained && f.fired;
        if (!f.fired && point == f.point && index == f.index) { f.fired = true; return true; }
        return f.direct && point == Point::OpsCopy;
    }
};
struct Fixture {
    llama_kvmem_execution_state * domain = llama_kvmem_execution_create();
    llama_kvmem_execution_state * previous = llama_kvmem_execution_exchange(domain);
    uint8_t * gpu = nullptr;
    std::vector<uint8_t> bytes = std::vector<uint8_t>(4096);
    Fixture() {
        require(kvmem_stagein_gpu_ready(1024, bytes.size()), "staging allocation failed");
        require(cudaMalloc(reinterpret_cast<void **>(&gpu), bytes.size()) == cudaSuccess, "source allocation failed");
        fill(7);
    }
    ~Fixture() {
        kvmem_stageout_test_fault(nullptr, nullptr);
        // Clear the injected failure and prove completion before GPU free,
        // including while unwinding a failed assertion.
        try { kvmem_stageout_clear(); } catch (...) { std::abort(); }
        if (gpu) cudaFree(gpu);
        llama_kvmem_execution_exchange(previous);
        llama_kvmem_execution_free(domain);
    }
    void fill(unsigned seed) {
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * 13 + seed);
        require(cudaMemcpy(gpu, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess,
                "source upload failed");
        // Pageable H2D may return before the default-stream device copy ends.
        // Staging uses a nonblocking stream, so establish the producer boundary
        // explicitly instead of relying on allocator/timing side effects.
        require(cudaStreamSynchronize(nullptr) == cudaSuccess, "source upload completion failed");
    }
    void enqueue(size_t offset = 0) {
        require(kvmem_stageout_enqueue(gpu + offset, 512) &&
                kvmem_stageout_enqueue(gpu + offset + 512, 512), "batch enqueue failed");
    }
    void verify(int slot, size_t offset = 0) {
        require(slot >= 0 && kvmem_stageout_slot_base(slot) == nullptr,
                "unconfirmed output was exposed before wait");
        require(kvmem_stageout_wait(slot, nullptr), "batch wait failed");
        const auto * data = kvmem_stageout_slot_base(slot);
        require(data && std::memcmp(data, bytes.data() + offset, 1024) == 0, "stale or incomplete stageout data");
        require(kvmem_stageout_used() == 0, "submitted raw pointers remained queued");
    }
    void next_batch() {
        kvmem_stageout_test_fault(nullptr, nullptr);
        fill(99);
        enqueue(2048);
        verify(kvmem_stageout_submit(nullptr), 2048);
    }
};

static void recoverable_submit_errors() {
    for (auto point : {Point::OpsCopy, Point::GatherLaunch, Point::PackedCopy, Point::ItemCopy, Point::EventRecord}) {
        Fixture f;
        Fault fault;
        fault.point = point;
        fault.direct = point == Point::ItemCopy;
        fault.index = point == Point::ItemCopy ? 1 : 0;
        kvmem_stageout_test_fault(Fault::inject, &fault);
        f.enqueue();
        const int slot = kvmem_stageout_submit(nullptr);
        require(fault.fired && fault.calls[size_t(Point::Drain)] > 0, "failed operation bypassed checked drain");
        if (point == Point::ItemCopy || point == Point::EventRecord) {
            require(slot == -1 && kvmem_stageout_used() == 0 && kvmem_stageout_submit(nullptr) == -1,
                    "failed submit retained raw pointers for a later batch");
        } else {
            f.verify(slot);
        }
        f.next_batch();
    }
    std::puts("PASS stageout submit errors: descriptor, kernel, packed copy, second direct copy, event record; checked drain and fresh batch reuse");
}

static void wait_errors(bool unconfirmed) {
    Fixture f;
    f.enqueue();
    const int slot = kvmem_stageout_submit(nullptr);
    require(slot >= 0, "wait fixture submit failed");
    f.enqueue(2048); // queued next batch must survive failure of the old event
    Fault fault;
    fault.point = Point::EventWait;
    fault.undrained = unconfirmed;
    kvmem_stageout_test_fault(Fault::inject, &fault);
    if (unconfirmed) {
        rejected([&] { kvmem_stageout_wait(slot, nullptr); }, "unconfirmed wait granted fallback permission");
        require(kvmem_stageout_slot_base(slot) == nullptr, "unconfirmed wait exposed output");
        kvmem_stageout_test_fault(nullptr, nullptr);
    }
    require(!kvmem_stageout_wait(slot, nullptr) && kvmem_stageout_used() == 1024,
            "failed old wait consumed the next unsent batch");
    require(fault.fired && fault.calls[size_t(Point::Drain)] > 0, "event failure did not try checked drain");
    kvmem_stageout_test_fault(nullptr, nullptr);
    f.verify(kvmem_stageout_submit(nullptr), 2048);
    std::printf("PASS stageout event wait error: unconfirmed=%d, next queue preserved\n", int(unconfirmed));
}

static void unconfirmed_submit() {
    Fixture f;
    Fault fault;
    fault.point = Point::EventRecord;
    fault.undrained = true;
    kvmem_stageout_test_fault(Fault::inject, &fault);
    f.enqueue();
    rejected([&] { kvmem_stageout_submit(nullptr); }, "unconfirmed submit granted fallback permission");
    require(fault.fired && kvmem_stageout_used() == 1024, "uncertain queue lost original sources");
    rejected([&] { kvmem_stageout_enqueue(f.gpu, 512); }, "uncertain domain admitted another source");
    rejected([&] { kvmem_stageout_clear(); }, "uncertain clear released source ownership");
    rejected([&] { kvmem_stagein_gpu_free(); }, "uncertain free released DMA destinations");
    rejected([&] { kvmem_stagein_gpu_ready(2048, 4096); }, "uncertain resize reused DMA destinations");
    kvmem_stageout_test_fault(nullptr, nullptr);
    require(kvmem_stageout_submit(nullptr) == -1 && kvmem_stageout_used() == 0,
            "settling a failed submission resubmitted its raw pointers");
    f.next_batch();
    std::puts("PASS unconfirmed stageout: admission/clear/free/resize refused, original queue retained until successful drain");
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        require(cudaSetDevice(0) == cudaSuccess, "device selection failed");
        recoverable_submit_errors();
        wait_errors(false);
        wait_errors(true);
        unconfirmed_submit();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL stageout errors: %s\n", e.what());
        return 1;
    }
}

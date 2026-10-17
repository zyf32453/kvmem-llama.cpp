#include "llama-kvmem-stagein.h"
#include "llama-kvmem-hooks.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

static thread_local bool deny_alloc = false;
void * operator new(size_t n) {
    if (deny_alloc) throw std::bad_alloc();
    if (auto * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }
struct Deny { Deny() { deny_alloc = true; } ~Deny() { deny_alloc = false; } };
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static bool empty() {
    const auto s = kvmem_capture_workspace_status();
    return !s.reserved && !s.job_limit && !s.device_bytes && !s.pinned_bytes && !s.events && !s.host_bytes;
}
static bool equal(KvmemCaptureWorkspace a, KvmemCaptureWorkspace b) {
    return a.reserved == b.reserved && a.job_limit == b.job_limit &&
        a.device_bytes == b.device_bytes && a.pinned_bytes == b.pinned_bytes &&
        a.events == b.events && a.host_bytes == b.host_bytes;
}
using Point = KvmemCaptureReservePoint;
struct Fault {
    Point target;
    bool fired = false;
    static bool inject(void * context, Point point) noexcept {
        auto & f = *static_cast<Fault *>(context);
        if (point != f.target) return false;
        f.fired = true;
        return true;
    }
};
struct Domain {
    llama_kvmem_execution_state * state = llama_kvmem_execution_create();
    llama_kvmem_execution_state * previous = llama_kvmem_execution_exchange(state);
    ~Domain() {
        kvmem_capture_reserve_test_fault(nullptr, nullptr);
        llama_kvmem_execution_exchange(previous);
        llama_kvmem_execution_free(state);
    }
};
static void failures() {
    Domain d;
    for (auto point : {Point::F32, Point::Packed, Point::Pin0, Point::Pin1,
                      Point::Event0, Point::Event1, Point::H2DEvent, Point::DeviceOps,
                      Point::HostItems, Point::HostOuts, Point::HostOps}) {
        Fault fault{point};
        kvmem_capture_reserve_test_fault(Fault::inject, &fault);
        require(!kvmem_capture_workspace_reserve(1024, 4096, 2) && fault.fired && empty(),
                "failed preparation retained workspace resources or published admission");
        kvmem_capture_reserve_test_fault(nullptr, nullptr);
        require(kvmem_capture_workspace_reserve(1024, 4096, 2), "retry after failed preparation was rejected");
        kvmem_stagein_gpu_free();
        require(empty(), "retry workspace was not released");
    }
    // Real C++ allocation failure rather than only a simulated false result.
    bool threw = false;
    Point last = Point::F32;
    const auto observe = [](void * context, Point point) noexcept {
        *static_cast<Point *>(context) = point;
        return false;
    };
    kvmem_capture_reserve_test_fault(observe, &last);
    try { Deny guard; kvmem_capture_workspace_reserve(1024, 4096, 2); }
    catch (const std::bad_alloc &) { threw = true; }
    kvmem_capture_reserve_test_fault(nullptr, nullptr);
    require(threw && last == Point::HostItems && empty(), "host allocation exception did not roll back CUDA workspace");
    std::puts("PASS capture workspace: 11 preparation failures plus bad_alloc, exact rollback and retry");
}
static void reuse() {
    Domain d;
    std::array<uint8_t, 1024> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * 17 + 3);
    uint8_t * gpu = nullptr;
    require(cudaMalloc(reinterpret_cast<void **>(&gpu), bytes.size()) == cudaSuccess, "source allocation");
    struct Free { uint8_t * p; ~Free() { cudaDeviceSynchronize(); cudaFree(p); } } free{gpu};
    require(cudaMemcpy(gpu, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) == cudaSuccess, "upload");
    require(cudaStreamSynchronize(nullptr) == cudaSuccess, "source upload completion");
    require(kvmem_capture_workspace_reserve(1024, bytes.size(), 2), "workspace preparation");
    const auto before = kvmem_capture_workspace_status();
    require(before.reserved && before.job_limit == 2 && before.pinned_bytes == 64 * 1024 * 1024 && before.events == 3,
            "workspace missing a required resource");
    require(!kvmem_capture_workspace_reserve(1024, bytes.size(), 2), "duplicate reservation accepted");
    {
        Deny guard;
        require(kvmem_stagein_gpu_ready(1024, bytes.size()) &&
                !kvmem_stagein_gpu_ready(1025, bytes.size()) &&
                !kvmem_stagein_gpu_ready(1024, 33 * 1024 * 1024), "fixed workspace grew or rejected valid reuse");
        int previous_slot = -1;
        for (unsigned i = 0; i < 24; ++i) {
            require(kvmem_stageout_enqueue(gpu, 512) && kvmem_stageout_enqueue(gpu + 512, 512), "enqueue");
            require(!kvmem_stageout_enqueue(gpu, 1) && kvmem_stageout_used() == 1024,
                    "full descriptor queue grew or changed offsets");
            if (previous_slot >= 0) {
                require(kvmem_stageout_wait(previous_slot, nullptr), "wait");
                const auto * packed = kvmem_stageout_slot_base(previous_slot);
                require(packed && !std::memcmp(packed, bytes.data(), bytes.size()), "incorrect previous slab");
            }
            previous_slot = kvmem_stageout_submit(nullptr);
            require(previous_slot == int(i % 2), "two slabs were not reused");
        }
        require(kvmem_stageout_wait(previous_slot, nullptr), "last wait");
        require(!std::memcmp(kvmem_stageout_slot_base(previous_slot), bytes.data(), bytes.size()), "last slab");
        kvmem_stageout_clear();
        require(equal(before, kvmem_capture_workspace_status()), "active capture changed workspace allocations");
    }
    kvmem_stagein_gpu_free();
    require(empty(), "workspace release leaked buffers or descriptor capacity");
    // A legacy warmed workspace must remain untouched on rejected preparation.
    require(kvmem_stagein_gpu_ready(1024, bytes.size()), "legacy preparation");
    const auto legacy = kvmem_capture_workspace_status();
    require(!kvmem_capture_workspace_reserve(1024, bytes.size(), 2) &&
            equal(legacy, kvmem_capture_workspace_status()), "capture discarded legacy resources");
    std::puts("PASS capture workspace: 24 saturated batches without C++ allocations, two pins reused, oversize rejected");
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try { failures(); reuse(); }
    catch (const std::exception & e) { std::fprintf(stderr, "FAIL capture workspace: %s\n", e.what()); return 1; }
    return 0;
}

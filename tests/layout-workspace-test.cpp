#include "llama-kvmem-stagein.h"
#include "llama-kvmem-hooks.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

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
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class Fn> static void rejected(Fn fn, const char * message) {
    bool threw = false;
    try { fn(); } catch (const std::exception &) { threw = true; }
    require(threw, message);
}
using Point = KvmemLayoutPoint;
struct Fault {
    Point point;
    bool undrained, fired = false;
    static bool inject(void * context, Point p) noexcept {
        auto & f = *static_cast<Fault *>(context);
        if (p == Point::Drain) return f.undrained;
        if (!f.fired && p == f.point) { f.fired = true; return true; }
        return false;
    }
};
struct Fixture {
    llama_kvmem_execution_state * domain = llama_kvmem_execution_create();
    llama_kvmem_execution_state * previous = llama_kvmem_execution_exchange(domain);
    uint8_t * gpu = nullptr;
    uint8_t * scratch = nullptr;
    std::vector<uint8_t> bytes;
    explicit Fixture(size_t size = 1024, bool admitted = true) : bytes(size) {
        require(cudaMalloc(reinterpret_cast<void **>(&gpu), size) == cudaSuccess, "source allocation");
        for (size_t i = 0; i < size; ++i) bytes[i] = uint8_t(i * 13 + i / 256);
        require(cudaMemcpy(gpu, bytes.data(), size, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaStreamSynchronize(nullptr) == cudaSuccess, "source initialization");
        if (admitted) require(kvmem_capture_workspace_reserve(1024, size, 2, size), "admission");
        scratch = kvmem_layout_workspace(size);
        require(scratch != nullptr, "layout allocation");
    }
    ~Fixture() {
        kvmem_layout_test_fault(nullptr, nullptr);
        try { kvmem_layout_cancel(); } catch (...) { std::abort(); }
        cudaFree(gpu);
        llama_kvmem_execution_exchange(previous);
        llama_kvmem_execution_free(domain);
    }
};
static void cyclic_reuse(bool admitted) {
    Fixture f(1024, admitted);
    const auto before = kvmem_capture_workspace_status();
    const void * src[] = {f.gpu + 512, f.gpu, f.gpu + 256};
    void * mid[] = {f.scratch, f.scratch + 256, f.scratch + 512};
    const void * mid_src[] = {mid[0], mid[1], mid[2]};
    void * dst[] = {f.gpu, f.gpu + 256, f.gpu + 512};
    size_t size[] = {256, 256, 256};
    std::array<uint8_t, 1024> expected{}, actual{};
    std::memcpy(expected.data(), f.bytes.data(), expected.size());
    for (unsigned round = 0; round < 30; ++round) {
        {
            Deny guard;
            kvmem_layout_copy(src, mid, size, 3);
            kvmem_layout_copy(mid_src, dst, size, 3);
        }
        const auto old = expected;
        std::memcpy(expected.data(), old.data() + 512, 256);
        std::memcpy(expected.data() + 256, old.data(), 512);
        require(cudaMemcpy(actual.data(), f.gpu, actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
                actual == expected, "cyclic layout bytes or untouched tail changed");
        require(kvmem_layout_workspace(f.bytes.size()) == f.scratch, "scratch was not reused");
    }
    const auto after = kvmem_capture_workspace_status();
    require(before.device_bytes == after.device_bytes && before.host_bytes == after.host_bytes,
            "layout grew admitted buffers");
    if (admitted) require(!kvmem_layout_workspace(f.bytes.size() + 1), "oversize layout accepted");
    std::printf("PASS layout: admitted=%d, 30 cyclic permutations without C++ allocation, exact bytes and reuse\n", int(admitted));
}
static void split_descriptors() {
    constexpr size_t count = KVMEM_CAPTURE_MAX_JOBS + 3;
    Fixture f(count);
    std::vector<const void *> src(count);
    std::vector<void *> dst(count);
    std::vector<size_t> bytes(count, 1);
    for (size_t i = 0; i < count; ++i) { src[i] = f.gpu + i; dst[i] = f.scratch + i; }
    { Deny guard; kvmem_layout_copy(src.data(), dst.data(), bytes.data(), count); }
    std::vector<uint8_t> actual(count);
    require(cudaMemcpy(actual.data(), f.scratch, count, cudaMemcpyDeviceToHost) == cudaSuccess && actual == f.bytes,
            "descriptor split lost or reused unfinished copy records");
    std::puts("PASS layout: 8195 copies split within fixed descriptor storage");
}
static void failures() {
    for (auto point : {Point::OpsCopy, Point::KernelResult, Point::Wait}) {
        for (bool undrained : {false, true}) {
            Fixture f;
            Fault fault{point, undrained};
            const void * src[] = {f.gpu}; void * dst[] = {f.scratch}; size_t bytes[] = {f.bytes.size()};
            const auto before = kvmem_capture_workspace_status();
            kvmem_layout_test_fault(Fault::inject, &fault);
            rejected([&] { kvmem_layout_copy(src, dst, bytes, 1); }, "failed layout returned success");
            require(fault.fired, "fault did not reach CUDA path");
            rejected([&] { kvmem_layout_copy(src, dst, bytes, 1); }, "partial operation was retried");
            rejected([&] { kvmem_layout_workspace(2048); }, "failed scratch reused");
            if (undrained) {
                rejected([&] { kvmem_layout_cancel(); }, "unconfirmed work cancelled");
                rejected([&] { kvmem_stagein_gpu_free(); }, "unconfirmed scratch freed");
                const auto held = kvmem_capture_workspace_status();
                require(held.device_bytes == before.device_bytes && held.pinned_bytes == before.pinned_bytes &&
                        held.host_bytes == before.host_bytes, "failed free lost resource ownership");
            }
            kvmem_layout_test_fault(nullptr, nullptr);
            kvmem_layout_cancel();
            rejected([&] { kvmem_layout_copy(src, dst, bytes, 1); }, "cancel reopened failed cache");
            kvmem_stagein_gpu_free();
            require(kvmem_capture_workspace_status().device_bytes == 0, "checked release leaked scratch");
        }
    }
    std::puts("PASS layout: six submission/completion failures, retained scratch until checked cancel, no replay");
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try { cyclic_reuse(true); cyclic_reuse(false); split_descriptors(); failures(); }
    catch (const std::exception & e) { std::fprintf(stderr, "FAIL layout workspace: %s\n", e.what()); return 1; }
    return 0;
}

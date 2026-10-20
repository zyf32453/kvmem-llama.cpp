#pragma once
#include <cstddef>

// Optional Linux diagnostic for the ordinary operator-new hook in the model
// test. Fixed storage; symbol lookup and file output happen outside decoding.
// This measures cumulative requests, not retained memory, C/CUDA allocation,
// aligned allocation, or allocations on untraced background threads.
#ifdef __linux__
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <stdexcept>

class ForegroundAllocationTrace {
    struct Site {
        uint64_t calls = 0, bytes = 0;
        int depth = 0;
        std::array<void *, 16> frames{};
    };
    std::array<Site, 2048> sites_{};
    uint64_t overflow_calls_ = 0, overflow_bytes_ = 0;
    FILE * output_ = nullptr;
    static inline thread_local ForegroundAllocationTrace * current_ = nullptr;
public:
    ForegroundAllocationTrace() {
        const char * path = std::getenv("KVMEM_TEST_ALLOCATION_TRACE");
        if (!path || !*path) return;
        output_ = std::fopen(path, "w");
        if (!output_) throw std::runtime_error("cannot create allocation trace");
        // Resolve unwinder dependencies before tracing a foreground interval.
        void * frames[16];
        (void) ::backtrace(frames, 16);
    }
    ~ForegroundAllocationTrace() {
        if (!output_) return;
        std::fprintf(output_, "overflow\t%llu\t%llu\n",
                     (unsigned long long) overflow_calls_, (unsigned long long) overflow_bytes_);
        for (size_t id = 0; id < sites_.size(); ++id) {
            const auto & site = sites_[id];
            if (!site.calls) continue;
            std::fprintf(output_, "site\t%zu\t%llu\t%llu\n", id,
                         (unsigned long long) site.calls, (unsigned long long) site.bytes);
            for (int frame = 0; frame < site.depth; ++frame) {
                Dl_info info{};
                const auto address = reinterpret_cast<uintptr_t>(site.frames[frame]) - 1;
                if (::dladdr(reinterpret_cast<void *>(address), &info)) {
                    const auto offset = address - reinterpret_cast<uintptr_t>(info.dli_fbase);
                    std::fprintf(output_, "frame\t%zu\t%d\t%s\t0x%llx\t%s\n", id, frame,
                                 info.dli_fname, (unsigned long long) offset,
                                 info.dli_sname ? info.dli_sname : "?");
                }
            }
        }
        std::fclose(output_);
    }
    ForegroundAllocationTrace(const ForegroundAllocationTrace &) = delete;
    ForegroundAllocationTrace & operator=(const ForegroundAllocationTrace &) = delete;

    class Interval {
        ForegroundAllocationTrace * previous_;
    public:
        explicit Interval(ForegroundAllocationTrace & trace) : previous_(current_) {
            current_ = trace.output_ ? &trace : nullptr;
        }
        ~Interval() { current_ = previous_; }
        Interval(const Interval &) = delete;
        Interval & operator=(const Interval &) = delete;
    };
    static void record(size_t bytes) noexcept {
        auto * trace = current_;
        if (!trace) return;
        current_ = nullptr; // An unwinder allocation must not recursively trace.
        std::array<void *, 16> frames{};
        const int depth = ::backtrace(frames.data(), frames.size());
        uint64_t hash = 14695981039346656037ULL;
        for (int i = 0; i < depth; ++i) {
            hash ^= reinterpret_cast<uintptr_t>(frames[i]);
            hash *= 1099511628211ULL;
        }
        bool recorded = false;
        for (size_t probe = 0; probe < trace->sites_.size(); ++probe) {
            auto & site = trace->sites_[(hash + probe) % trace->sites_.size()];
            if (!site.calls || (site.depth == depth &&
                    std::memcmp(site.frames.data(), frames.data(), depth * sizeof(void *)) == 0)) {
                if (!site.calls) { site.frames = frames; site.depth = depth; }
                ++site.calls;
                site.bytes += bytes;
                recorded = true;
                break;
            }
        }
        if (!recorded) { ++trace->overflow_calls_; trace->overflow_bytes_ += bytes; }
        current_ = trace;
    }
};
#else
class ForegroundAllocationTrace {
public:
    class Interval {
    public:
        explicit Interval(ForegroundAllocationTrace &) {}
    };
    static void record(size_t) noexcept {}
};
#endif

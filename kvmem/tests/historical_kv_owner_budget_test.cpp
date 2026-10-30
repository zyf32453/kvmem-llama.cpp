#include "kvmem/historical_kv_page_gc.hpp"
#include "kvmem/historical_kv_redundancy.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/memory_contract.hpp"

#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <malloc.h>
#endif

// Reject unaccounted C++ allocations as well as allocations through the budget
// during the actual registration/commit/ACK/reclaim path. libc stdio/RSS are
// outside this test's allocation contract.
static thread_local bool reject_new = false;
void * operator new(size_t bytes) {
    if (reject_new) throw std::bad_alloc();
    if (auto * pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void * operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void * pointer) noexcept { std::free(pointer); }
void operator delete[](void * pointer) noexcept { std::free(pointer); }
void operator delete(void * pointer, size_t) noexcept { std::free(pointer); }
void operator delete[](void * pointer, size_t) noexcept { std::free(pointer); }
void * operator new(size_t bytes, std::align_val_t alignment) {
    if (reject_new) throw std::bad_alloc();
    void * pointer = nullptr;
#if defined(_WIN32)
    pointer = _aligned_malloc(bytes ? bytes : 1, size_t(alignment));
#else
    if (posix_memalign(&pointer, std::max(size_t(alignment), sizeof(void *)), bytes ? bytes : 1)) pointer = nullptr;
#endif
    if (!pointer) throw std::bad_alloc();
    return pointer;
}
void operator delete(void * pointer, std::align_val_t) noexcept {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}
void operator delete(void * pointer, size_t, std::align_val_t alignment) noexcept { ::operator delete(pointer, alignment); }
void * operator new[](size_t bytes, std::align_val_t alignment) { return ::operator new(bytes, alignment); }
void operator delete[](void * pointer, std::align_val_t alignment) noexcept { ::operator delete(pointer, alignment); }
void operator delete[](void * pointer, size_t, std::align_val_t alignment) noexcept { ::operator delete(pointer, alignment); }

using namespace kvmem;
using Budget = HistoricalKvAuxBudget;
using Kind = HistoricalKvAuxKind;
using Repo = HistoricalKvRepository;
template<class T> using Alloc = HistoricalKvAuxAllocator<T>;
static void check(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
template<class F> static void denied(F operation) {
    try { operation(); }
    catch (const MemoryError & error) { check(error.code() == MemoryErrorCode::BudgetExceeded, "wrong budget failure"); return; }
    throw std::runtime_error("budget exhaustion accepted");
}
template<class F> static void invalid(F operation) {
    try { operation(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("unaccounted/mismatched component accepted");
}
struct RejectNew {
    RejectNew() { reject_new = true; }
    ~RejectNew() { reject_new = false; }
};
struct RecordingResource final : std::pmr::memory_resource {
    size_t charged = 0, requests = 0, fail_request = SIZE_MAX;
    std::vector<size_t> below_request_limits;
    std::map<void *, std::pair<size_t, size_t>> live;
    void * do_allocate(size_t bytes, size_t alignment) override {
        if (++requests == fail_request) throw std::bad_alloc();
        below_request_limits.push_back(charged + bytes - 1);
        auto * pointer = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        try { live.emplace(pointer, std::make_pair(bytes, alignment)); }
        catch (...) { std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment); throw; }
        charged += bytes; return pointer;
    }
    void do_deallocate(void * pointer, size_t bytes, size_t alignment) override {
        const auto found = live.find(pointer);
        if (found == live.end() || found->second != std::make_pair(bytes, alignment)) std::terminate();
        live.erase(found); charged -= bytes;
        std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    }
    bool do_is_equal(const memory_resource & other) const noexcept override { return this == &other; }
};
static void audit(bool enabled) {
#if defined(_WIN32)
    _putenv_s("KVMEM_GC_AUDIT", enabled ? "1" : "0");
#else
    setenv("KVMEM_GC_AUDIT", enabled ? "1" : "0", 1);
#endif
}
static HistoricalKvPageGc::Config native_config(const std::shared_ptr<Budget> & budget) {
    HistoricalKvPageGc::Config config;
    config.slots = 8; config.logical_pages = 16;
    config.low_ready = 1; config.high_ready = 2; config.batch_blocks = 2;
    config.auxiliary = budget;
    return config;
}

static void native_hot_path_and_constructor_failures(bool auditing) {
    audit(auditing);
    size_t constructor_requests;
    std::vector<size_t> limits;
    {
        RecordingResource upstream;
        auto budget = std::make_shared<Budget>(1 << 20, &upstream);
        {
            auto bridge = historical_kv_aux_make<HistoricalKvPageGc>(budget, Kind::NativeBridge, native_config(budget));
            constructor_requests = upstream.requests;
            limits = upstream.below_request_limits;
            HostKvStorage host(8 * 16);
            std::vector<HostKvRecord> records;
            for (unsigned i = 0; i < 8; ++i) records.push_back(host.allocate(16));
            check(budget->stats().charged == upstream.charged, "native owner accounting drift");
            const auto charge = budget->stats().charged;
            {
                RejectNew guard;
                for (unsigned i = 0; i < 8; ++i) {
                    check(bridge->register_page(i, 16, i + 1), "native register");
                    bridge->protect_page(i, false);
                }
                bridge->manual_tick();
                const auto & retired = bridge->commit_boundary();
                check(retired.size() == 2, "native prepared batch missing");
                for (const auto & page : retired)
                    bridge->acknowledge_host_release(page, std::move(records[page.page]));
                bridge->manual_tick();
            }
            check(bridge->stats().reclaimed == 2 && host.charged_bytes() == 6 * 16, "native physical reclaim missing");
            check(upstream.requests == constructor_requests && budget->stats().charged == charge,
                  "hot path grew metadata/decision/bridge storage");
            auto first = host.allocate(16), second = host.allocate(16);
            {
                RejectNew guard;
                check(bridge->register_page(8, 16, 9) && bridge->register_page(9, 16, 10), "budgeted slot reuse");
            }
            check(upstream.requests == constructor_requests, "reused generation allocated callback storage");
        }
        check(budget->stats().charged == 0 && upstream.live.empty(), "native destructor leak");
    }
    // Fault every allocation in the whole root/policy/repository/manager graph.
    for (size_t failure = 1; failure <= constructor_requests; ++failure) {
        RecordingResource upstream;
        upstream.fail_request = failure;
        auto budget = std::make_shared<Budget>(1 << 20, &upstream);
        bool failed = false;
        try { auto bridge = historical_kv_aux_make<HistoricalKvPageGc>(budget, Kind::NativeBridge, native_config(budget)); }
        catch (const std::bad_alloc &) { failed = true; }
        check(failed && upstream.requests == failure && upstream.live.empty() && budget->stats().charged == 0,
              "partial native owner construction leaked");
    }
    for (size_t limit : limits) {
        RecordingResource upstream;
        auto budget = std::make_shared<Budget>(limit, &upstream);
        denied([&] { auto bridge = historical_kv_aux_make<HistoricalKvPageGc>(budget, Kind::NativeBridge, native_config(budget)); });
        check(budget->stats().charged == 0 && upstream.live.empty() && budget->stats().peak <= limit,
              "partial native budget denial leaked");
    }
    auto small = std::make_shared<Budget>(1);
    denied([&] { auto bridge = historical_kv_aux_make<HistoricalKvPageGc>(small, Kind::NativeBridge, native_config(small)); });
    check(small->stats().charged == 0, "failed root budget charge leaked");
    std::cout << "native_constructor_failure_points audit=" << auditing << " count=" << constructor_requests << '\n';
}

static void retry_batches_do_not_grow() {
    RecordingResource upstream;
    auto budget = std::make_shared<Budget>(1 << 20, &upstream);
    auto repository = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 4, 0, 0, 0, budget);
    HistoricalKvHandle last;
    for (unsigned i = 0; i < 4; ++i) {
        auto writer = repository->begin_write(i, i * 64, 64, i + 1, 16,
            +[](const HistoricalKvHandle &) { return true; });
        check(bool(writer) && repository->publish_write(*writer), "retry repository fixture");
        last = writer->handle();
    }
    HistoricalKvGc::Config config;
    config.mode = HistoricalKvGc::Mode::Delete; config.retire_at_decode_boundary = true;
    config.ready_low = 1; config.ready_high = 2;
    config.batch_blocks = 2; config.scan_blocks = config.reclaim_blocks = 4;
    auto policy = historical_kv_aux_make<HistoricalKvPolicy, HistoricalKvLruPolicy>(budget, Kind::Policy,
        4, true, std::shared_ptr<HistoricalKvThinkingRanges>{}, budget);
    HistoricalKvGc gc(repository, std::move(policy), config);
    gc.manual_tick();
    const auto requests = upstream.requests, charged = budget->stats().charged;
    struct Pause { std::mutex mutex; std::condition_variable cv; bool entered = false, opened = false; } pause;
    std::thread thread([&] {
        repository->retire(last, {}, [](const HistoricalKvBlockMeta &, const void * state) noexcept {
            auto & p = *static_cast<Pause *>(const_cast<void *>(state));
            std::unique_lock<std::mutex> lock(p.mutex);
            p.entered = true; p.cv.notify_all(); p.cv.wait(lock, [&] { return p.opened; });
            return false;
        }, &pause);
    });
    const auto open = [&] { { std::lock_guard<std::mutex> lock(pause.mutex); pause.opened = true; pause.cv.notify_all(); } thread.join(); };
    bool entered;
    { std::unique_lock<std::mutex> lock(pause.mutex); entered = pause.cv.wait_for(lock, std::chrono::seconds(5), [&] { return pause.entered; }); }
    try {
        check(entered, "retry turn not entered");
        RejectNew guard;
        for (unsigned n = 0; n < 5; ++n) {
            check(gc.commit_at_decode_boundary().committed == 0, "contended transaction committed");
            gc.manual_tick();
        }
    } catch (...) { open(); throw; }
    open();
    {
        RejectNew guard;
        check(gc.commit_at_decode_boundary().committed == 2, "retained retry batch did not commit");
        gc.manual_tick();
    }
    check(gc.stats().retry == 10 && gc.stats().committed == 2 && gc.stats().reclaimed == 2 &&
          !gc.stats().policy_failures && !gc.stats().worker_failures, "retry ownership/count changed");
    check(upstream.requests == requests && budget->stats().charged == charged, "retry ring reallocated or refunded early");
}

static void old_repository_lease_holds_real_account() {
    size_t domain_bytes;
    {
        auto budget = std::make_shared<Budget>(1 << 20);
        auto repository = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 2, 0, 0, 0, budget);
        domain_bytes = budget->stats().charged;
    }
    RecordingResource upstream;
    auto budget = std::make_shared<Budget>(domain_bytes, &upstream);
    auto repository = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 2, 0, 0, 0, budget);
    auto writer = repository->begin_write(0, 0, 64, 1, 16, +[](const HistoricalKvHandle &) { return true; });
    check(bool(writer) && repository->publish_write(*writer), "old domain fixture");
    auto reader = repository->acquire_read(writer->handle());
    writer->reset();
    repository.reset();
    const auto held = budget->stats().charged;
    check(held > 0 && held == upstream.charged, "closed directory refunded live lease");
    denied([&] { auto next = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 2, 0, 0, 0, budget); });
    check(reader->metadata().has_value() && budget->stats().charged == held, "failed replacement corrupted old lease");
    reader->reset();
    check(budget->stats().charged == 0, "old lease did not release directory");
    repository = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 2, 0, 0, 0, budget);
    repository.reset();
    check(budget->stats().charged == 0 && upstream.live.empty(), "replacement domain leaked");
}

static void reservation_and_failed_owner_claim() {
    auto budget = std::make_shared<Budget>(1 << 20);
    auto repository = std::allocate_shared<Repo>(Alloc<Repo>(budget, Kind::Metadata), 4, 0, 0, 0, budget);
    const auto setup = budget->stats().charged;
    {
        HistoricalKvAuxVector<uint8_t> held(budget->stats().capacity - setup, 0, Alloc<uint8_t>(budget, Kind::Features));
        denied([&] { repository->try_reserve_slots(4); });
        const auto observed = repository->observe_capacity();
        check(observed.ready == 4 && observed.pending == 0, "failed reservation mutated slots");
    }
    {
        auto reserved = repository->try_reserve_slots(4);
        check(bool(reserved) && reserved->remaining() == 4 && reserved->next_handle()->generation == 1,
              "failed reservation consumed generation");
        const auto during = budget->stats().charged;
        reserved->reset();
        check(repository->ready_count() == 4 && budget->stats().charged == during, "reset refunded retained handle capacity");
    }
    check(budget->stats().charged == setup, "reservation destructor leak");
    int foreign = 0;
    invalid([&] { repository->begin_write(0, 0, 64, 1, 16, [&foreign](const HistoricalKvHandle &) { ++foreign; return true; }); });
    check(repository->ready_count() == 4 && foreign == 0, "rejected callback touched generation");
    HistoricalKvGc::Config config;
    config.mode = HistoricalKvGc::Mode::Delete; config.batch_blocks = 2;
    invalid([&] { HistoricalKvGc gc(repository, std::make_unique<HistoricalKvLruPolicy>(4), config); });
    check(budget->stats().charged == setup, "failed manager leaked or claimed repository");
    {
        auto policy = historical_kv_aux_make<HistoricalKvPolicy, HistoricalKvLruPolicy>(budget, Kind::Policy, 4,
            true, std::shared_ptr<HistoricalKvThinkingRanges>{}, budget);
        HistoricalKvGc gc(repository, std::move(policy), config);
        check(budget->stats().by_kind[static_cast<size_t>(Kind::Decisions)] > 0, "manager did not inherit account");
    }
    check(budget->stats().charged == setup, "manager claim not releasable");
    auto other = std::make_shared<Budget>(1 << 20);
    invalid([&] {
        auto source = std::make_shared<HistoricalKvPageFeatures>(4, 3, other);
        HistoricalKvRedundancyPolicy::Config policy_config; policy_config.auxiliary = budget;
        HistoricalKvRedundancyPolicy policy(4, source, policy_config);
    });
    check(budget->stats().charged == setup && other->stats().charged == 0, "foreign source mismatch leaked");
    repository.reset();
    check(budget->stats().charged == 0, "repository owner leak");
}

int main() {
    const char * original = std::getenv("KVMEM_GC_AUDIT");
    const bool had_original = original != nullptr;
    const std::string saved = original ? original : "";
    int code = 0;
    try {
        native_hot_path_and_constructor_failures(false);
        native_hot_path_and_constructor_failures(true);
        audit(false);
        old_repository_lease_holds_real_account();
        reservation_and_failed_owner_claim();
        retry_batches_do_not_grow();
        std::cout << "PASS owner budget: native fixed allocation path, full constructor faults, old domains, reservation atomicity, owner-claim rollback and account identity\n";
    } catch (const std::exception & error) { std::cerr << "FAIL owner budget: " << error.what() << '\n'; code = 1; }
#if defined(_WIN32)
    _putenv_s("KVMEM_GC_AUDIT", saved.c_str());
#else
    if (had_original) setenv("KVMEM_GC_AUDIT", saved.c_str(), 1); else unsetenv("KVMEM_GC_AUDIT");
#endif
    return code;
}

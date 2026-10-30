#pragma once

#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/host_kv_storage.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace kvmem {

// One NInfer logical-history owner. The background manager never touches the
// native archive. The serialized request owner registers sealed pages, updates
// protection, commits at a decode boundary, drops exact Host records, and then
// acknowledges physical release. This object has a stable address across a
// movable native sequence/window and must outlive its GC manager.
class HistoricalKvPageGc final {
public:
    struct Config {
        uint32_t slots = 0;
        uint32_t logical_pages = 0;
        uint32_t page_tokens = 64;
        uint64_t sink_tokens = 0;
        uint64_t recent_tokens = 0;
        uint32_t low_ready = 0;
        uint32_t high_ready = 0;
        uint32_t batch_blocks = 64;
        uint32_t poll_ms = 25;
        // Native RAM charges include unsealed records and outstanding readers,
        // which do not necessarily occupy repository slots yet. The worker
        // reads only this allocator's thread-safe byte counters.
        std::optional<HostKvStorage> physical_storage;
        uint64_t physical_page_bytes = 0;
    };
    struct Watermarks {
        uint32_t low_ready = 0;
        uint32_t high_ready = 0;
    };
    // Size free-page headroom from an actual append chunk, not a whole output
    // reservation. protected_pages bounds the native active view (sink/recent
    // included); temporary Query pins/readers still obey normal admission.
    static Watermarks derive_watermarks(uint32_t slots, uint32_t protected_pages,
                                       uint32_t append_pages);

    struct RetiredPage {
        uint32_t page = 0;
        HistoricalKvHandle handle{};
    };

    explicit HistoricalKvPageGc(Config config);
    // A native backend may inject a strategy; the bridge still owns protection,
    // boundary retirement and physical Host-record release.
    HistoricalKvPageGc(Config config, std::unique_ptr<HistoricalKvPolicy> policy);
    ~HistoricalKvPageGc();
    HistoricalKvPageGc(const HistoricalKvPageGc &) = delete;
    HistoricalKvPageGc & operator=(const HistoricalKvPageGc &) = delete;

    // Register only a fully captured, sealed native page. A failed admission
    // leaves the page unregistered; the caller must not continue silently.
    bool register_page(uint32_t page, uint64_t payload_bytes, uint64_t birth_seq);
    // The request owner updates this before each boundary. Protection includes
    // sink/recent, media, current view, checkpoint restore rows and in-flight IO.
    void protect_page(uint32_t page, bool protect) noexcept;
    void note_use(uint32_t page, uint64_t sequence) noexcept;
    void request_capacity(uint32_t ready);
    void start();
    void manual_tick();
    // Returned pages are stable until the next commit_boundary() call. The
    // caller must mark each page unavailable before another working-set choice.
    const std::vector<RetiredPage> & commit_boundary() noexcept;
    // Move the exact native owner out of the selectable archive at the boundary.
    // The worker normally destroys it later. An urgent prefill admission may
    // drop the owner now; any in-flight Host reader still holds its lease and
    // the worker waits for the release ticket before reusing this slot.
    void acknowledge_host_release(const RetiredPage & page,
                                  HostKvRecord record, bool release_immediately = false) noexcept;
    bool contains(uint32_t page) const noexcept;
    bool is_live(uint32_t page) const noexcept;
    HistoricalKvGc::Stats stats() const;
    HistoricalKvPolicy::Metrics policy_metrics() const noexcept;

private:
    struct ReleaseState;
    class ProtectedPolicy;
    static void on_retired(void * context, HistoricalKvHandle handle) noexcept;
    const Config config_;
    std::shared_ptr<ReleaseState> releases_;
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::unique_ptr<HistoricalKvGc> manager_;
    std::vector<HistoricalKvHandle> handles_;
    std::vector<uint32_t> slot_to_page_;
    std::vector<RetiredPage> committed_;
};

} // namespace kvmem

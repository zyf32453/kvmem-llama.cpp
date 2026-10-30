#include "kvmem/historical_kv_page_gc.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kvmem {

struct HistoricalKvPageGc::ReleaseState {
    ReleaseState(uint32_t pages, uint32_t slots)
        : protected_pages(new std::atomic<uint8_t>[pages]),
          released_generation(new std::atomic<uint64_t>[slots]),
          release_tickets(new HostKvReleaseTicket[slots]),
          detached_records(new HostKvRecord[slots]),
          logical_pages(pages) {
        for (uint32_t i = 0; i < pages; ++i)
            protected_pages[i].store(1, std::memory_order_relaxed);
        for (uint32_t i = 0; i < slots; ++i)
            released_generation[i].store(0, std::memory_order_relaxed);
    }
    bool protected_page(uint64_t page) const noexcept {
        return page >= logical_pages ||
            protected_pages[page].load(std::memory_order_acquire) != 0;
    }
    std::unique_ptr<std::atomic<uint8_t>[]> protected_pages;
    std::unique_ptr<std::atomic<uint64_t>[]> released_generation;
    std::unique_ptr<HostKvReleaseTicket[]> release_tickets;
    std::unique_ptr<HostKvRecord[]> detached_records;
    uint32_t logical_pages;
};

class HistoricalKvPageGc::ProtectedPolicy final : public HistoricalKvPolicy {
public:
    ProtectedPolicy(std::unique_ptr<HistoricalKvPolicy> inner,
                    std::shared_ptr<ReleaseState> state, uint32_t capacity,
                    std::optional<HostKvStorage> storage, uint64_t page_bytes)
        : inner_(std::move(inner)), state_(std::move(state)), capacity_(capacity),
          physical_storage_(std::move(storage)), physical_page_bytes_(page_bytes) {
        filtered_.blocks.reserve(capacity);
    }
    bool permits(const HistoricalKvBlockMeta & block) const noexcept override {
        return !state_->protected_page(block.logical_id) && inner_->permits(block);
    }
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure & pressure,
                                HistoricalKvWorkBudget budget,
                                HistoricalKvDecisionBuffer & out) override {
        if (round_ != snapshot.round_id || incarnation_ != snapshot.incarnation) {
            round_ = snapshot.round_id;
            incarnation_ = snapshot.incarnation;
            filtered_.round_id = snapshot.round_id;
            filtered_.incarnation = snapshot.incarnation;
            filtered_.blocks.clear();
            for (const auto & block : snapshot.blocks) {
                auto annotated = block;
                // Protected pages may witness redundancy, but must never be
                // proposed as victims, including after protection changes.
                annotated.mandatory |= state_->protected_page(block.logical_id);
                filtered_.blocks.push_back(annotated);
            }
        }
        auto observed = pressure;
        if (physical_storage_) {
            const auto capacity = physical_storage_->capacity_bytes();
            const auto charged = physical_storage_->charged_bytes();
            const auto bytes = charged < capacity ? capacity - charged : 0;
            observed.ready = static_cast<uint32_t>(std::min<uint64_t>(
                capacity_, bytes / physical_page_bytes_));
        }
        return inner_->step(filtered_, observed, budget, out);
    }
    void on_result(const HistoricalKvCommitResult & result) noexcept override {
        inner_->on_result(result);
    }
    Metrics metrics() const noexcept override { return inner_->metrics(); }
private:
    std::unique_ptr<HistoricalKvPolicy> inner_;
    std::shared_ptr<ReleaseState> state_;
    HistoricalKvSnapshot filtered_;
    uint32_t capacity_;
    std::optional<HostKvStorage> physical_storage_;
    uint64_t physical_page_bytes_;
    uint64_t round_ = 0, incarnation_ = 0;
};

HistoricalKvPageGc::HistoricalKvPageGc(Config config)
    : HistoricalKvPageGc(config, std::make_unique<HistoricalKvLruPolicy>(config.slots)) {}

HistoricalKvPageGc::HistoricalKvPageGc(Config config, std::unique_ptr<HistoricalKvPolicy> inner)
    : config_(config),
      releases_(std::make_shared<ReleaseState>(config.logical_pages, config.slots)),
      repository_(std::make_shared<HistoricalKvRepository>(
          config.slots, 0, config.sink_tokens, config.recent_tokens)),
      handles_(config.logical_pages),
      slot_to_page_(config.slots, UINT32_MAX) {
    if (!inner || !config.slots || !config.logical_pages || !config.batch_blocks ||
        !config.page_tokens || !config.low_ready || config.low_ready >= config.high_ready ||
        config.high_ready > config.slots || !config.poll_ms ||
        (config.physical_storage && !config.physical_page_bytes))
        throw std::invalid_argument("NInfer historical GC page capacity/watermarks");
    committed_.reserve(config.batch_blocks);
    HistoricalKvGc::Config gc;
    gc.mode = HistoricalKvGc::Mode::Delete;
    gc.ready_low = config.low_ready;
    gc.ready_high = config.high_ready;
    gc.scan_blocks = std::min<uint32_t>(config.slots, 256);
    gc.batch_blocks = std::min(config.batch_blocks, config.slots);
    gc.reclaim_blocks = std::min<uint32_t>(config.slots, 64);
    gc.poll_ms = config.poll_ms;
    gc.retire_at_decode_boundary = true;
    gc.on_boundary_retired = &HistoricalKvPageGc::on_retired;
    gc.boundary_context = this;
    auto policy = std::make_unique<ProtectedPolicy>(std::move(inner), releases_, config.slots,
        config.physical_storage, config.physical_page_bytes);
    manager_ = std::make_unique<HistoricalKvGc>(repository_, std::move(policy), gc);
}

HistoricalKvPageGc::~HistoricalKvPageGc() {
    manager_->stop();
    repository_->close();
}

bool HistoricalKvPageGc::register_page(uint32_t page, uint64_t payload_bytes, uint64_t birth_seq) {
    if (page >= handles_.size() || !payload_bytes || handles_[page].generation)
        return false;
    const std::weak_ptr<ReleaseState> weak = releases_;
    auto writer = repository_->begin_write(page, uint64_t(page) * config_.page_tokens,
        config_.page_tokens,
        birth_seq, payload_bytes, [weak](const HistoricalKvHandle & handle) {
            const auto state = weak.lock();
            if (!state) return true;
            if (state->released_generation[handle.slot].load(
                    std::memory_order_acquire) != handle.generation) return false;
            // The foreground only moved the record into this preallocated slot.
            // Destruction and any last-reader wait happen on the GC worker.
            state->detached_records[handle.slot].reset();
            return state->release_tickets[handle.slot].released();
        });
    if (!writer) return false;
    const auto handle = writer->handle();
    slot_to_page_[handle.slot] = page;
    handles_[page] = handle;
    if (!repository_->publish_write(*writer))
        throw std::logic_error("NInfer historical GC page publication failed");
    return true;
}

void HistoricalKvPageGc::protect_page(uint32_t page, bool protect) noexcept {
    if (page < config_.logical_pages)
        releases_->protected_pages[page].store(protect ? 1 : 0, std::memory_order_release);
}

void HistoricalKvPageGc::note_use(uint32_t page, uint64_t sequence) noexcept {
    if (!contains(page)) return;
    if (auto lease = repository_->acquire_gpu(handles_[page]))
        (void)repository_->mark_executed(*lease, sequence);
}

void HistoricalKvPageGc::request_capacity(uint32_t ready) { manager_->request_capacity(ready); }
void HistoricalKvPageGc::start() { manager_->start(); }
void HistoricalKvPageGc::manual_tick() { manager_->manual_tick(); }

void HistoricalKvPageGc::on_retired(void * context, HistoricalKvHandle handle) noexcept {
    auto& self = *static_cast<HistoricalKvPageGc *>(context);
    if (handle.slot >= self.slot_to_page_.size() ||
        self.committed_.size() == self.committed_.capacity()) std::terminate();
    const auto page = self.slot_to_page_[handle.slot];
    if (page >= self.handles_.size() ||
        self.handles_[page].generation != handle.generation) std::terminate();
    self.committed_.push_back({page, handle});
}

const std::vector<HistoricalKvPageGc::RetiredPage> &
HistoricalKvPageGc::commit_boundary() noexcept {
    committed_.clear();
    manager_->commit_at_decode_boundary();
    return committed_;
}

void HistoricalKvPageGc::acknowledge_host_release(const RetiredPage & page,
                                                 HostKvRecord record,
                                                 bool release_immediately) noexcept {
    if (page.page >= handles_.size() || page.handle.slot >= config_.slots) return;
    const auto current = handles_[page.page];
    if (current.incarnation != page.handle.incarnation ||
        current.slot != page.handle.slot || current.generation != page.handle.generation) return;
    releases_->release_tickets[page.handle.slot] = record.release_ticket();
    if (release_immediately) record.reset();
    releases_->detached_records[page.handle.slot] = std::move(record);
    releases_->released_generation[page.handle.slot].store(page.handle.generation,
        std::memory_order_release);
}

bool HistoricalKvPageGc::contains(uint32_t page) const noexcept {
    return page < handles_.size() && handles_[page].generation != 0;
}

bool HistoricalKvPageGc::is_live(uint32_t page) const noexcept {
    if (!contains(page)) return false;
    return repository_->completion_state(handles_[page]) ==
        HistoricalKvRepository::CompletionState::Live;
}

HistoricalKvGc::Stats HistoricalKvPageGc::stats() const { return manager_->stats(); }
HistoricalKvPolicy::Metrics HistoricalKvPageGc::policy_metrics() const noexcept {
    return manager_->policy_metrics();
}

} // namespace kvmem

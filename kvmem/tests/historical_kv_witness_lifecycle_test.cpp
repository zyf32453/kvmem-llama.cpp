// This regression uses only B2 public APIs, so the exact same source can be
// compiled against the old and current headers/libraries as a negative control.
#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/host_kv_storage.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

using namespace kvmem;
static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        HostKvStorage storage(32);
        std::array<HostKvRecord, 2> records;
        auto repository = std::make_shared<HistoricalKvRepository>(2);
        std::array<HistoricalKvHandle, 2> handles{};
        auto features = std::make_shared<HistoricalKvPageFeatures>(2, 3);
        for (unsigned id = 0; id < 2; ++id) {
            records[id] = storage.allocate(16);
            records[id].seal();
            auto writer = repository->begin_write(id, id * 64, 64, id + 1, 16,
                [&, id](const HistoricalKvHandle &) { records[id].reset(); return true; });
            require(bool(writer) && repository->publish_write(*writer), "writer fixture");
            handles[id] = writer->handle();
            features->publish_page(id, {1, 0, 0});
        }
        features->publish_question_scores({0, 0});
        HistoricalKvRedundancyPolicy::Config policy;
        policy.max_candidate_blocks = 2;
        HistoricalKvGc::Config cfg;
        cfg.mode = HistoricalKvGc::Mode::Delete;
        cfg.retire_at_decode_boundary = true;
        cfg.ready_low = 1; cfg.ready_high = 2;
        cfg.scan_blocks = cfg.reclaim_blocks = 2; cfg.batch_blocks = 1;
        HistoricalKvGc gc(repository, std::make_unique<HistoricalKvRedundancyPolicy>(2, features, policy), cfg);
        for (unsigned tick = 0; tick < 30 && !gc.stats().proposed; ++tick) gc.manual_tick();
        require(gc.stats().proposed == 1, "missing scored proposal");
        // Equal vectors/importance: the newer page is the retained witness.
        require(repository->retire(handles[1]) == HistoricalKvRepository::RetireResult::Committed,
                "witness retirement failed");
        require(repository->try_reclaim(handles[1]) == HistoricalKvRepository::ReclaimResult::Complete,
                "witness release failed");
        require(storage.charged_bytes() == 16 && records[0] && !records[1], "witness bytes not really released");
        const auto result = gc.commit_at_decode_boundary();
        require(result.committed == 0, "redundancy victim committed after witness was retired and freed");
        require(repository->completion_state(handles[0]) == HistoricalKvRepository::CompletionState::Live,
                "last live candidate was removed using an unavailable witness");
        gc.manual_tick();
        require(gc.stats().reevaluate >= 1, "lost witness rejection result");
        std::cout << "PASS lost witness: actual Host release prevents stale redundancy commit\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL witness lifecycle: " << error.what() << '\n';
        return 1;
    }
}

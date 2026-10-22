#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_raw_store.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace kvmem;

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

template <class Predicate>
static void wait_for(Predicate done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done()) {
        check(std::chrono::steady_clock::now() < deadline, "background GC timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main() {
    try {
        constexpr uint32_t capacity = 3;
        const auto foreground = std::this_thread::get_id();
        std::atomic<unsigned> foreground_releases{0};
        auto repo = std::make_shared<HistoricalKvRepository>(capacity);
        HistoricalKvRawStoreConfig raw_cfg;
        raw_cfg.n_layer = 1;
        raw_cfg.max_tokens_per_slot = 4;
        raw_cfg.packed_k_bytes_per_token = 1;
        raw_cfg.packed_v_bytes_per_token = 1;
        auto raw = std::make_shared<HistoricalKvRawStore>(raw_cfg, capacity);
        const std::weak_ptr<HistoricalKvRepository> weak_repo = repo;
        const auto release = [raw, weak_repo, foreground, &foreground_releases](HistoricalKvHandle h) {
            if (std::this_thread::get_id() == foreground) ++foreground_releases;
            auto owner = weak_repo.lock();
            if (!owner) return false;
            const auto code = raw->release(*owner, h).code;
            return code == HistoricalKvRawStore::ReleaseCode::Released ||
                   code == HistoricalKvRawStore::ReleaseCode::AlreadyReleased;
        };
        const auto append = [&](uint64_t id) {
            auto writer = repo->begin_write(id, id * 4, 4, id + 1, 8, release);
            check(writer.has_value(), "no free slot for append");
            const auto h = writer->handle();
            check(raw->reserve(*writer, h, id * 4, 4) &&
                  raw->write_layer(*writer, h, 0, {1, 2, 3, 4}, {5, 6, 7, 8}, {}, 0) &&
                  raw->publish_complete(*repo, *writer, h), "payload publication failed");
            return h;
        };
        const auto a = append(0);
        const auto one_block = raw->allocated_payload_bytes();
        const auto b = append(1);
        const auto c = append(2);
        const auto use = [&](HistoricalKvHandle h, uint64_t sequence) {
            auto execution = repo->acquire_new(h);
            check(execution && repo->mark_executed(*execution, sequence), "cannot record use");
        };
        use(a, 30);
        use(b, 20);
        use(c, 10);
        auto old_reader = repo->acquire_read(c);
        check(old_reader.has_value(), "cannot start old C reader");
        check(!repo->begin_write(3, 12, 4, 4, 8, release), "full store admitted another block");
        check(raw->allocated_payload_bytes() == 3 * one_block, "unexpected initial allocation");
        std::cout << "[1] CPU store full: A(last_use=30), B(20), C(10). C has an existing reader.\n"
                  << "    payload_bytes=" << raw->allocated_payload_bytes()
                  << "; fourth block rejected immediately.\n";

        HistoricalKvGc::Config cfg;
        cfg.mode = HistoricalKvGc::Mode::Delete;
        cfg.ready_low = 1;
        cfg.ready_high = 2;
        cfg.scan_blocks = cfg.batch_blocks = cfg.reclaim_blocks = capacity;
        cfg.poll_ms = 1;
        HistoricalKvGc gc(repo, std::make_unique<HistoricalKvLruPolicy>(capacity), cfg);
        gc.start();
        wait_for([&] { const auto s = gc.stats(); return s.committed == 2 && s.reclaimed == 1 && s.deferred > 0; });
        gc.stop(); // Pause only to make the demonstration output deterministic.
        check(repo->acquire_new(a).has_value() && !repo->acquire_new(b) && !repo->acquire_new(c),
              "LRU retired the wrong blocks or allowed new use");
        check(raw->allocated_payload_bytes() == 2 * one_block, "C freed before its reader completed");
        const auto old_data = raw->read_layer(*old_reader, c, 0);
        check(old_data && old_data->packed_k == std::vector<uint8_t>({1, 2, 3, 4}) &&
              old_data->packed_v == std::vector<uint8_t>({5, 6, 7, 8}), "old reader lost C data");
        std::cout << "[2] LRU proposed C and B; GC closed new use of both. Only A remains selectable.\n"
                  << "    B freed in background; C is still readable by its old reader.\n"
                  << "    payload_bytes=" << raw->allocated_payload_bytes() << "; ready_slots=1.\n";

        old_reader->reset();
        gc.start();
        wait_for([&] { return repo->ready_count() == 2; });
        gc.stop();
        check(raw->allocated_payload_bytes() == one_block, "C payload not released");
        std::cout << "[3] C reader finished; background GC freed C. payload_bytes="
                  << raw->allocated_payload_bytes() << "; ready_slots=2.\n";

        const auto d = append(3);
        check((d.slot == b.slot && d.generation > b.generation) ||
              (d.slot == c.slot && d.generation > c.generation), "slot not safely reused");
        check(!repo->acquire_new(b) && !repo->acquire_new(c), "old identity became valid again");
        check(raw->allocated_payload_bytes() == 2 * one_block, "reuse changed the payload bound");
        std::cout << "[4] New block D reused a freed slot with a new generation; old handles stay invalid.\n"
                  << "    payload_bytes=" << raw->allocated_payload_bytes() << ".\n";

        // Explicit shutdown, independent of LRU's pressure threshold.
        check(repo->retire(a) == HistoricalKvRepository::RetireResult::Committed &&
              repo->retire(d) == HistoricalKvRepository::RetireResult::Committed, "shutdown retirement failed");
        gc.start();
        wait_for([&] { return repo->ready_count() == capacity; });
        gc.stop();
        const auto stats = gc.stats();
        check(raw->allocated_payload_bytes() == 0 && foreground_releases == 0 &&
              stats.reclaimed == 4 && !stats.policy_failures && !stats.worker_failures,
              "cleanup failed or release ran on foreground");
        std::cout << "[5] Shutdown drained all four allocations in background. payload_bytes=0.\n"
                  << "PASS: LRU -> retire -> drain existing reader -> free CPU payload -> reuse slot.\n";
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}

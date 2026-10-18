// NvmeKvTier host-logic test. Covers slot sizing, block residency, read/write,
// explicit LRU eviction, release/reuse, and crash-safe backing-file cleanup.

#include "kvmem/nvme_kv_tier.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static std::string temp_dir() {
    const char *base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    return std::string(base) + "/qw3_nvme_kv_tier_test";
}

static void test_disabled() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 0;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    CHECK(!t.enabled());
    CHECK(t.slot_count() == 0);
}

static void test_write_read_release() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 256;
    cfg.slot_bytes = 64;
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.drops_page_cache());
    CHECK(t.slot_count() == 4);
    // The open descriptor remains usable, but the cache has no directory
    // entry and therefore cannot survive process exit as a stale large file.
    CHECK(::access(t.path().c_str(), F_OK) != 0);

    std::vector<uint8_t> a(64), b(64), out(64);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(i);
        b[i] = static_cast<uint8_t>(255 - i);
    }

    t.write_block(10, a.data(), a.size());
    CHECK(t.block_slot(10) == 0);
    t.read_block(10, out.data(), out.size());
    CHECK(out == a);

    t.write_block(10, b.data(), b.size());
    t.read_block(10, out.data(), out.size());
    CHECK(out == b);

    t.release_block(10);
    CHECK(t.block_slot(10) == -1);
    CHECK(t.free_slots() == 4);
}

static void test_slot_ranges_and_file_names() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.file_name = "qw3_raw_k_range_test.bin";
    cfg.total_bytes = 256;
    cfg.slot_bytes = 128;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.path().find(cfg.file_name) != std::string::npos);
    CHECK(::access(t.path().c_str(), F_OK) != 0);

    const auto p = t.place_block(7);
    CHECK(p.slot == 0);
    std::vector<uint8_t> main(80, 0x31);
    std::vector<uint8_t> mtp(24, 0x92);
    std::vector<uint8_t> main_out(main.size(), 0);
    std::vector<uint8_t> mtp_out(mtp.size(), 0);
    t.write_slot_range(p.slot, 0, main.data(), main.size());
    t.write_slot_range(p.slot, 96, mtp.data(), mtp.size());
    t.read_slot_range(p.slot, 0, main_out.data(), main_out.size());
    t.read_slot_range(p.slot, 96, mtp_out.data(), mtp_out.size());
    CHECK(main_out == main);
    CHECK(mtp_out == mtp);

    bool rejected = false;
    try {
        t.read_slot_range(p.slot, 120, mtp_out.data(), mtp_out.size());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);
}

static void test_evicting_place() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 128;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    std::vector<uint8_t> a(64, 1), b(64, 2), c(64, 3), out(64);
    t.write_block(1, a.data(), a.size());
    t.write_block(2, b.data(), b.size());
    t.touch(1);  // block 2 becomes LRU.

    auto p = t.place_block_evicting(3);
    CHECK(p.slot == 1);
    CHECK(p.evicted_block == 2);
    CHECK(t.block_slot(2) == -1);
    CHECK(t.block_slot(3) == 1);

    t.write_block(3, c.data(), c.size());
    t.read_block(3, out.data(), out.size());
    CHECK(out == c);
}

static void test_coalesced_batch_io() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.drops_page_cache());

    std::vector<uint8_t> input(64 * 3), output(64 * 3, 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17) & 0xff);
    }
    std::vector<NvmeIoSpan> spans;
    for (uint32_t block = 0; block < 3; ++block) {
        const auto p = t.place_block(100 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block) * 64, 64});
    }

    NvmeBatchIoStats writes;
    t.write_spans(spans, input.data(), input.size(), &writes);
    CHECK(writes.bytes == input.size());
    CHECK(writes.syscalls == 1);
    CHECK(writes.cache_drop_bytes == input.size());
    CHECK(writes.cache_drop_failures == 0);

    NvmeBatchIoStats reads;
    t.read_spans(spans, output.data(), output.size(), &reads);
    CHECK(reads.bytes == output.size());
    CHECK(reads.syscalls == 1);
    CHECK(reads.cache_drop_bytes == output.size());
    CHECK(reads.cache_drop_failures == 0);
    CHECK(output == input);
}

static void test_concurrent_positional_batches() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);

    std::vector<uint8_t> a(64 * 3, 0x35);
    std::vector<uint8_t> b(64 * 3, 0xca);
    std::vector<uint8_t> output(64 * 6, 0);
    std::vector<NvmeIoSpan> a_spans;
    std::vector<NvmeIoSpan> b_spans;
    for (uint32_t block = 0; block < 6; ++block) {
        const auto p = t.place_block(200 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        auto &spans = block < 3 ? a_spans : b_spans;
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block % 3) * 64, 64});
    }

    auto aw = std::async(std::launch::async, [&]() {
        t.write_spans(a_spans, a.data(), a.size());
    });
    auto bw = std::async(std::launch::async, [&]() {
        t.write_spans(b_spans, b.data(), b.size());
    });
    aw.get();
    bw.get();

    std::vector<NvmeIoSpan> all;
    for (uint32_t slot = 0; slot < 6; ++slot) {
        all.push_back(NvmeIoSpan{
            static_cast<int32_t>(slot),
            static_cast<uint64_t>(slot) * 64, 64});
    }
    NvmeBatchIoStats reads;
    t.read_spans(all, output.data(), output.size(), &reads);
    CHECK(reads.syscalls == 1);
    CHECK(std::equal(a.begin(), a.end(), output.begin()));
    CHECK(std::equal(
        b.begin(), b.end(), output.begin() +
            static_cast<std::ptrdiff_t>(a.size())));
}

static void test_exact_allocation_reclamation_with_delayed_io() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.slot_bytes = 65536;
    cfg.total_bytes = 2 * cfg.slot_bytes;
    NvmeKvTier tier(cfg);
    const auto a = tier.place_block(1), b = tier.place_block(2);
    auto lease = tier.acquire_io(a.allocation);
    CHECK(lease && tier.free_slots() == 0);
    std::vector<uint8_t> input(cfg.slot_bytes, 0x73);
    std::promise<void> first_half, continue_io;
    auto gate = continue_io.get_future();
    auto started = first_half.get_future();
    // An actual multi-part file write is admitted before retirement. Keep its
    // lease through the second write, readback and completion, even while the
    // allocation is closed to all new jobs.
    auto job = std::async(std::launch::async,
        [&, ownership = std::move(lease), gate = std::move(gate)]() mutable {
            try {
                tier.write_allocation(ownership, 0, input.data(), input.size() / 2);
            } catch (...) { first_half.set_exception(std::current_exception()); throw; }
            first_half.set_value();
            gate.get();
            tier.write_allocation(ownership, input.size() / 2,
                                  input.data() + input.size() / 2, input.size() / 2);
            std::vector<uint8_t> output(input.size());
            std::vector<NvmeIoSpan> spans{{a.slot, 0, cfg.slot_bytes, &ownership}};
            tier.read_spans(spans, output.data(), output.size());
            ownership.reset();
            return output == input;
        });
    started.get();
    CHECK(tier.try_release_allocation(a.allocation) == NvmeReleaseResult::Deferred);
    CHECK(!tier.acquire_io(a.allocation));
    CHECK(tier.place_block(1).slot == -1 && tier.place_block_evicting(1).slot == -1);
    CHECK(tier.free_slots() == 0);
    // Independent allocations remain reclaimable/reusable during the old IO.
    CHECK(tier.try_release_allocation(b.allocation) == NvmeReleaseResult::Released);
    const auto c = tier.place_block(3);
    CHECK(c.slot == b.slot && c.allocation.generation > b.allocation.generation);
    auto protect_c = tier.acquire_io(c.allocation);
    CHECK(tier.place_block_evicting(4).slot == -1);
    bool rejected = false;
    try { tier.release_block(1); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && tier.free_slots() == 0);
    CHECK(tier.try_release_allocation(b.allocation) == NvmeReleaseResult::Stale);
    continue_io.set_value();
    CHECK(job.get());
    CHECK(tier.try_release_allocation(a.allocation) == NvmeReleaseResult::Released);
    CHECK(tier.try_release_allocation(a.allocation) == NvmeReleaseResult::AlreadyReleased);
    CHECK(tier.free_slots() == 1);
    const auto next = tier.place_block(4);
    CHECK(next.slot == a.slot && next.allocation.generation > a.allocation.generation);
    CHECK(tier.try_release_allocation(a.allocation) == NvmeReleaseResult::Stale);
    CHECK(!tier.acquire_io(a.allocation) && tier.free_slots() == 0);
    auto current = tier.acquire_io(next.allocation);
    tier.write_allocation(current, 0, input.data(), input.size());
    NvmeKvTier unrelated(cfg);
    CHECK(unrelated.try_release_allocation(next.allocation) == NvmeReleaseResult::Stale);
    CHECK(!unrelated.acquire_io(next.allocation));
    rejected = false;
    try { unrelated.write_allocation(current, 0, input.data(), input.size()); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    // clear cannot make either old allocation reusable while jobs own it.
    rejected = false;
    try { tier.clear(); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && tier.free_slots() == 0 && !tier.acquire_io(next.allocation));
    current.reset();
    protect_c.reset();
    tier.clear();
    CHECK(tier.free_slots() == 2);
    const auto after_clear = tier.place_block(4);
    CHECK(!tier.acquire_io(next.allocation));
    CHECK(after_clear.allocation.generation > 1);
    std::puts("PASS NVMe delayed real IO, independent release, stale/cross-arena handles and clear");
}

static void test_concurrent_admission_retirement_and_reuse() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.slot_bytes = cfg.total_bytes = 256;
    NvmeKvTier tier(cfg);
    NvmeAllocationHandle previous;
    for (uint32_t cycle = 0; cycle < 64; ++cycle) {
        const auto p = tier.place_block(cycle);
        CHECK(p.slot == 0 && p.allocation.generation == cycle + 1);
        CHECK(!tier.acquire_io(previous));
        auto writer = tier.acquire_io(p.allocation);
        std::vector<uint8_t> bytes(256, uint8_t(cycle));
        tier.write_allocation(writer, 0, bytes.data(), bytes.size());
        writer.reset();
        std::atomic<bool> stop{false}, failed{false}, read_once{false};
        std::thread reader([&] {
            std::vector<uint8_t> output(256);
            while (!stop.load()) {
                auto read = tier.acquire_io(p.allocation);
                if (!read) { std::this_thread::yield(); continue; }
                tier.read_allocation(read, 0, output.data(), output.size());
                if (output != bytes) failed = true;
                read_once = true;
            }
        });
        while (!read_once.load()) std::this_thread::yield();
        auto result = tier.try_release_allocation(p.allocation);
        CHECK(result == NvmeReleaseResult::Released || result == NvmeReleaseResult::Deferred ||
              result == NvmeReleaseResult::Retry);
        stop = true;
        reader.join();
        result = tier.try_release_allocation(p.allocation);
        CHECK(result == NvmeReleaseResult::Released || result == NvmeReleaseResult::AlreadyReleased);
        CHECK(tier.free_slots() == 1 && !failed && !tier.acquire_io(p.allocation));
        previous = p.allocation;
    }
    std::puts("PASS NVMe 64 generations through one physical extent with concurrent real readers");
}

static void test_concurrent_arenas_have_independent_files() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.slot_bytes = cfg.total_bytes = 4096;
    std::vector<std::future<std::unique_ptr<NvmeKvTier>>> workers;
    for (unsigned i = 0; i < 16; ++i) {
        workers.push_back(std::async(std::launch::async, [cfg, i] {
            auto tier = std::make_unique<NvmeKvTier>(cfg);
            std::vector<uint8_t> input(4096, uint8_t(i));
            tier->write_block(0, input.data(), input.size());
            return tier;
        }));
    }
    std::set<std::string> paths;
    std::vector<std::unique_ptr<NvmeKvTier>> arenas;
    for (auto & worker : workers) arenas.push_back(worker.get());
    for (unsigned i = 0; i < arenas.size(); ++i) {
        CHECK(paths.insert(arenas[i]->path()).second);
        std::vector<uint8_t> output(4096);
        arenas[i]->read_block(0, output.data(), output.size());
        CHECK(std::all_of(output.begin(), output.end(), [i](uint8_t v) { return v == i; }));
    }
    std::puts("PASS NVMe simultaneous arenas use independent anonymous files");
}

static void test_direct_mapped_overlay_reuse() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.file_name = "allocation-archive.bin";
    cfg.slot_bytes = cfg.total_bytes = 4096;
    cfg.durable = cfg.direct_mapped = true;
    const std::vector<uint8_t> base(4096, 0x13);
    {
        NvmeKvTier archive(cfg);
        archive.write_block(0, base.data(), base.size());
    }
    cfg.read_only = true;
    cfg.overlay_dir = cfg.dir;
    {
        std::vector<std::future<std::unique_ptr<NvmeKvTier>>> workers;
        for (unsigned i = 0; i < 8; ++i) {
            workers.push_back(std::async(std::launch::async, [cfg, i] {
                auto tier = std::make_unique<NvmeKvTier>(cfg);
                tier->mark_present_range(0, 1);
                auto lease = tier->acquire_io(tier->allocation_for_block(0));
                const uint8_t byte = uint8_t(i + 32);
                tier->write_allocation(lease, 10, &byte, 1);
                return tier;
            }));
        }
        std::vector<std::unique_ptr<NvmeKvTier>> overlays;
        for (auto & worker : workers) overlays.push_back(worker.get());
        for (unsigned i = 0; i < overlays.size(); ++i) {
            auto & tier = *overlays[i];
            auto old = tier.allocation_for_block(0);
            std::vector<uint8_t> output(4096);
            tier.read_block(0, output.data(), output.size());
            auto expected = base;
            expected[10] = uint8_t(i + 32);
            CHECK(output == expected);
            CHECK(tier.try_release_allocation(old) == NvmeReleaseResult::Released);
            const auto fresh = tier.place_block(0);
            CHECK(fresh.allocation.generation > old.generation && !tier.acquire_io(old));
            // Closing a session-local overlay allocation cannot carry its old
            // bytes into a fresh binding to the immutable base archive.
            tier.read_block(0, output.data(), output.size());
            CHECK(output == base);
        }
        cfg.overlay_dir.clear();
        NvmeKvTier archive(cfg);
        archive.mark_present_range(0, 1);
        std::vector<uint8_t> output(4096);
        archive.read_block(0, output.data(), output.size());
        CHECK(output == base);
    }
    CHECK(::unlink((cfg.dir + "/" + cfg.file_name).c_str()) == 0);
    std::puts("PASS NVMe independent overlays, generation reset and unchanged archive");
}

int main() {
    test_disabled();
    test_write_read_release();
    test_slot_ranges_and_file_names();
    test_evicting_place();
    test_coalesced_batch_io();
    test_concurrent_positional_batches();
    test_exact_allocation_reclamation_with_delayed_io();
    test_concurrent_admission_retirement_and_reuse();
    test_concurrent_arenas_have_independent_files();
    test_direct_mapped_overlay_reuse();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}

// Shared behavioral probe: compile this unchanged against both frozen cores.
// No engine/GPU types and no assumptions about native page protection.
#include "kvmem/historical_kv_repository.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

using Repo = kvmem::HistoricalKvRepository;
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void retired_identity_cannot_authorize_reuse() {
    Repo repository(1, 9001);
    unsigned released = 0;
    auto writer = repository.begin_write(10, 17, 3, 1, 80,
        [&](const kvmem::HistoricalKvHandle &) { ++released; return true; });
    require(bool(writer), "writer admission");
    const auto old = writer->handle();
    require(repository.publish_write(*writer), "publish");
    writer->reset();
    auto reader = repository.acquire_read(old);
    require(bool(reader), "old read admission");
    require(repository.retire(old) == Repo::RetireResult::Committed, "retirement with reader");
    require(!repository.acquire_new(old) && !repository.acquire_read(old) && !repository.acquire_gpu(old),
            "retired identity granted new access");
    repository.try_reclaim(old);
    require(released == 0 && repository.ready_count() == 0, "reader released early");
    reader->reset();
    require(repository.try_reclaim(old) == Repo::ReclaimResult::Complete, "release did not finish");
    require(released == 1 && repository.ready_count() == 1, "incorrect release accounting");
    auto next = repository.begin_write(11, 20, 1, 2, 80,
        [&](const kvmem::HistoricalKvHandle &) { ++released; return true; });
    require(bool(next), "slot reuse");
    const auto fresh = next->handle();
    require(fresh.slot == old.slot && fresh.generation != old.generation, "generation not advanced");
    require(repository.publish_write(*next), "new publish");
    next->reset();
    require(repository.retire(old) == Repo::RetireResult::Stale, "old retirement touched new generation");
    require(!repository.acquire_read(old), "old read crossed reuse");
    auto valid = repository.acquire_read(fresh);
    require(bool(valid), "stale intent damaged fresh generation");
    valid->reset();
    require(repository.retire(fresh) == Repo::RetireResult::Committed, "fresh retirement");
    require(repository.try_reclaim(fresh) == Repo::ReclaimResult::Complete && released == 2,
            "fresh final release");
    repository.try_reclaim(old);
    repository.try_reclaim(fresh);
    require(released == 2, "duplicate release");
}

static void cpu_release_does_not_revoke_gpu_permission() {
    Repo repository(1, 9002);
    std::array<unsigned, 5> releases{};
    Repo::ResourceRelease release;
    release.packed_kv_bytes = 64;
    release.mean_k_bytes = 16;
    release.callback = [&](const kvmem::HistoricalKvHandle &, Repo::HostResource resource) {
        ++releases.at(static_cast<size_t>(resource));
        return true;
    };
    auto writer = repository.begin_write(20, 101, 2, 1, 80, std::move(release));
    require(bool(writer), "split writer admission");
    const auto handle = writer->handle();
    require(repository.publish_write(*writer), "split publish");
    writer->reset();
    auto gpu = repository.acquire_gpu(handle);
    require(bool(gpu) && !gpu->has_host_access(), "GPU-only resource rights");
    require(repository.retire(handle) == Repo::RetireResult::Committed, "GPU-held retirement");
    for (unsigned i = 0; i < 6; ++i) repository.try_reclaim(handle);
    require(releases[static_cast<size_t>(Repo::HostResource::PackedKv)] == 1 &&
            releases[static_cast<size_t>(Repo::HostResource::MeanK)] == 1,
            "GPU-only permission unnecessarily retained CPU resources");
    require(releases[static_cast<size_t>(Repo::HostResource::Metadata)] == 0 &&
            repository.ready_count() == 0 && bool(*gpu), "GPU identity reused too early");
    require(!repository.acquire_packed_kv(handle), "retired GPU right upgraded to Host read");
    gpu->reset();
    require(repository.try_reclaim(handle) == Repo::ReclaimResult::Complete, "GPU final release");
    require(releases[static_cast<size_t>(Repo::HostResource::Metadata)] == 1 &&
            repository.ready_count() == 1, "metadata completion");
}

int main() {
    try {
        retired_identity_cannot_authorize_reuse();
        std::cout << "PASS C01-C03 retirement, held reader, generation reuse and exactly-once release\n";
        cpu_release_does_not_revoke_gpu_permission();
        std::cout << "PASS C04 CPU early release, preserved GPU permission and delayed Ready\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL lifecycle compatibility: " << error.what() << '\n';
        return 1;
    }
}

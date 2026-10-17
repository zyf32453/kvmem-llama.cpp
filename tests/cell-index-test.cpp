#include "llama-kv-cells.h"
#include "kvmem/historical_kv_capture_session.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

static thread_local int64_t allocation_budget = -1;
static thread_local size_t frees = 0;
void * operator new(size_t n) {
    if (!allocation_budget) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { if (p) ++frees; std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

static void prepare_and_rollback() {
    llama_kv_cells cells;
    cells.resize(8);
    unsigned failures = 0;
    bool succeeded = false;
    for (int64_t budget = 0; budget < 100; ++budget) {
        allocation_budget = budget;
        try { auto prepared = cells.prepare_index_nodes(0); succeeded = bool(prepared); }
        catch (const std::bad_alloc &) { ++failures; }
        allocation_budget = -1;
        check(!cells.index_status().reserved && !cells.get_used(), "failed preparation mutated cells");
        if (succeeded) break;
    }
    check(succeeded && failures, "cell preparation failure sweep missed coverage");
    kvmem::KvMemRuntimeConfig config;
    config.store.block_tokens = 4;
    kvmem::KvMemRuntime runtime(config);
    kvmem::HistoricalKvRamDemand demand;
    demand.uncached_input_tokens = 4; demand.max_accepted_output_tokens = 0; demand.max_append_boundaries = 1;
    auto capture = kvmem::HistoricalKvCaptureSession::try_prepare(runtime, {1, 4, 1, 1, 0, {}}, 1, demand);
    auto prepared = cells.prepare_index_nodes(0);
    check(capture && prepared, "transaction fixture did not prepare");
    allocation_budget = 0;
    const bool installed = prepared->install();
    allocation_budget = -1;
    runtime.register_append(1); // The other resource domain changes before commit.
    check(installed && !capture->admit(), "changed runtime accepted a stale transaction");
    allocation_budget = 0;
    prepared.reset(); capture.reset();
    allocation_budget = -1;
    check(!cells.index_status().reserved && !cells.index_status().nodes && !cells.get_used(),
          "failed admission retained installed cell nodes");
    auto stale = cells.prepare_index_nodes(0);
    cells.pos_set(0, 19); cells.seq_add(0, 0);
    check(!stale->install() && cells.pos_get(0) == 19, "preparation overwrote newer cell state");
    cells.reset();
    cells.pos_set(0, 3); cells.seq_add(0, 0);
    cells.pos_add(0, -4);
    check(!cells.get_used() && cells.get_has_shift() && !cells.prepare_index_nodes(0),
          "preparation ignored pending position-shift work");
    cells.reset_shift();
    check(bool(cells.prepare_index_nodes(0)), "settled empty cells could not prepare");
}

static void exercise_reserved_cells() {
    llama_kv_cells cells;
    cells.resize(8);
    auto prepared = cells.prepare_index_nodes(0);
    check(prepared && prepared->install(), "cell node installation failed");
    prepared->accept(); prepared.reset();
    check(cells.index_status().nodes == 16 && cells.index_status().available == 16,
          "cell reservation did not allocate both real indexes");
    for (unsigned round = 0; round < 128; ++round) {
        const auto before = frees;
        allocation_budget = 0;
        for (uint32_t i = 0; i < 8; ++i) {
            cells.pos_set(i, 2 * round + i / 2); // Repeated M-RoPE positions use distinct cell keys.
            llama_kv_cell_ext ext;
            ext.tok = 100 + i; ext.logical_pos = round * 8 + i;
            cells.ext_set(i, ext); cells.seq_add(i, 0);
        }
        const auto used = cells.get_used();
        const auto token = cells.seq_pos_tok_le(0, 2 * round + 1);
        cells.seq_rm(1, 0);
        cells.rm(4);
        cells.pos_add(6, 3); cells.pos_div(7, 2);
        cells.seq_keep(2, 0);
        cells.reset_shift();
        allocation_budget = -1;
        check(used == 8 && token == 103 && frees == before && cells.get_used() == 6,
              "reserved cell mutation allocated/freed or changed ordered position semantics");
        auto snapshot = cells.cp(0, 8); // Snapshot storage is outside this reservation.
        const std::vector<uint32_t> reverse{7, 6, 5, 4, 3, 2, 1, 0};
        const auto before_restore = frees;
        allocation_budget = 0;
        cells.reset();
        cells.set(reverse, snapshot);
        cells.reset();
        cells.set(0, snapshot);
        const auto count = cells.get_used();
        cells.resize(8); // Same size resets with all prepared capacities intact.
        allocation_budget = -1;
        check(count == 6 && frees == before_restore && !cells.get_used() &&
              cells.index_status().nodes == 16 && cells.index_status().available == 16,
              "restore/reset lost node credits or allocated/freed");
    }
    cells.pos_set(0, 7); cells.seq_add(0, 0);
    bool rejected = false;
    try { cells.seq_add(0, 1); } catch (const std::logic_error &) { rejected = true; }
    check(rejected && !cells.seq_has(0, 1), "unreserved sequence partially mutated the cell");
    llama_kv_cells foreign;
    foreign.resize(1); foreign.pos_set(0, 12); foreign.seq_add(0, 1);
    rejected = false;
    try { cells.set(0, foreign); } catch (const std::logic_error &) { rejected = true; }
    check(rejected && cells.pos_get(0) == 7 && cells.seq_has(0, 0), "foreign restore partially mutated reserved cells");
    rejected = false;
    try { cells.resize(9); } catch (const std::logic_error &) { rejected = true; }
    check(rejected && cells.size() == 8 && cells.pos_get(0) == 7, "reserved cache grew or lost contents");
}

static void ordinary_multi_sequence_and_reserved_nonzero_sequence() {
    llama_kv_cells cells;
    cells.resize(3);
    cells.pos_set(0, 17); cells.seq_add(0, 0); cells.seq_add(0, 1);
    cells.pos_set(1, 17); cells.seq_add(1, 1);
    llama_kv_cell_ext ext;
    ext.tok = 31; cells.ext_set(0, ext);
    ext.tok = 32; cells.ext_set(1, ext);
    check(cells.seq_pos_tok_le(1, 17) == 32 && !cells.prepare_index_nodes(0),
          "ordinary shared-cell ordering or nonempty preparation changed");
    check(!cells.seq_rm(0, 0) && cells.seq_has(0, 1), "ordinary sequence removal lost another owner");
    auto snapshot = cells.cp(0, 3);
    cells.reset(); cells.set(0, snapshot);
    check(cells.get_used() == 2 && cells.seq_pos_tok_le(1, 17) == 32,
          "ordinary multi-sequence restore changed");
    cells.reset();
    auto prepared = cells.prepare_index_nodes(1);
    check(prepared && prepared->install(), "nonzero sequence reservation failed");
    prepared->accept();
    allocation_budget = 0;
    cells.set(0, snapshot);
    const auto used = cells.get_used();
    cells.reset();
    allocation_budget = -1;
    check(used == 2 && cells.index_status().available == 6, "reserved nonzero sequence restore lost nodes");
}

int main() try {
    prepare_and_rollback(); exercise_reserved_cells();
    ordinary_multi_sequence_and_reserved_nonzero_sequence();
    std::puts("PASS real upstream cells: allocation faults, admission rollback, 128 node reuse/restore rounds, ordered lookup, bounded sequences");
} catch (const std::exception & e) {
    allocation_budget = -1;
    std::fprintf(stderr, "FAIL real upstream cells: %s\n", e.what());
    return 1;
}

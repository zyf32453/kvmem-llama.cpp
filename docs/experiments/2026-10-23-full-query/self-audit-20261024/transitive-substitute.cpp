#include "kvmem/historical_kv_redundancy.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

using namespace kvmem;

struct AuditFeatures final : HistoricalKvRedundancySource {
    std::array<std::array<float, 2>, 3> keys;
    AuditFeatures() {
        const double pi = std::acos(-1.0);
        for (size_t i = 0; i < keys.size(); ++i) {
            const double angle = double(i) * 20.0 * pi / 180.0;
            keys[i] = {{float(std::cos(angle)), float(std::sin(angle))}};
        }
    }
    size_t dimensions() const noexcept override { return 2; }
    bool begin_round(uint64_t) override { return true; }
    bool read(const HistoricalKvBlockMeta & meta, float * out, double & logit) override {
        std::copy(keys.at(meta.logical_id).begin(), keys.at(meta.logical_id).end(), out);
        logit = meta.logical_id == 0 ? 10.0 : 0.0;
        return true;
    }
    double similarity(size_t a, size_t b) const {
        return double(keys[a][0])*keys[b][0] + double(keys[a][1])*keys[b][1];
    }
};

int main() {
    auto source = std::make_shared<AuditFeatures>();
    auto thinking = std::make_shared<HistoricalKvThinkingRanges>();
    thinking->arm();
    thinking->note(0, 96);
    HistoricalKvRedundancyPolicy::Config cfg;
    cfg.thinking = thinking;
    cfg.pool_blocks = 3;
    cfg.segment_blocks = 1;
    HistoricalKvRedundancyPolicy policy(3, source, cfg);
    HistoricalKvSnapshot snapshot;
    snapshot.incarnation = 7;
    snapshot.round_id = 1;
    for (unsigned i = 0; i < 3; ++i) {
        HistoricalKvBlockMeta meta;
        meta.handle = {7, i, 1};
        meta.logical_id = i;
        meta.original_begin = i * 32;
        meta.valid_tokens = 32;
        meta.content_version = 9;
        meta.last_use_seq = i;
        meta.life = HistoricalKvLife::Live;
        meta.sealed = true;
        meta.mandatory = i == 0;
        snapshot.blocks.push_back(meta);
    }
    HistoricalKvDecisionBuffer decisions(3);
    // requested_ready disables the separate lossy fallback: every emitted
    // deletion here must have been classified as redundancy.
    HistoricalKvPressure pressure{0, 1, 2, 0, 2};
    for (unsigned i = 0; i < 100; ++i) {
        if (policy.step(snapshot, pressure, {3, 3}, decisions).status != HistoricalKvStepStatus::Yielded) break;
        if (i == 99) return 2;
    }
    std::array<bool, 3> deleted{};
    bool only_redundancy = true;
    for (const auto & intent : decisions.values()) {
        deleted[intent.victim.slot] = true;
        only_redundancy &= intent.reason == HistoricalKvDeleteReason::Redundancy;
    }
    size_t uncovered = 0;
    for (size_t i = 0; i < deleted.size(); ++i) {
        if (!deleted[i]) continue;
        bool covered = false;
        for (size_t j = 0; j < deleted.size(); ++j)
            if (!deleted[j] && source->similarity(i,j) >= cfg.representative_similarity) covered = true;
        if (!covered) ++uncovered;
    }
    std::cout << "{\"similarity_ab\":" << source->similarity(0,1)
              << ",\"similarity_bc\":" << source->similarity(1,2)
              << ",\"similarity_ac\":" << source->similarity(0,2)
              << ",\"threshold\":" << cfg.representative_similarity
              << ",\"deleted\":[" << deleted[0] << ',' << deleted[1] << ',' << deleted[2]
              << "],\"only_redundancy\":" << (only_redundancy ? "true" : "false")
              << ",\"lossy_intents\":" << policy.metrics().lossy_intents
              << ",\"uncovered_deleted_blocks\":" << uncovered << "}\n";
    return uncovered ? 1 : 0;
}

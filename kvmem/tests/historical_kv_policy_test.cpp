#include "kvmem/historical_kv_policy.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace kvmem;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static HistoricalKvBlockMeta block(uint32_t slot, uint64_t last_use,
                                   uint64_t birth, HistoricalKvLife life =
                                       HistoricalKvLife::Live) {
    HistoricalKvBlockMeta meta;
    meta.handle = {7, slot, 1};
    meta.logical_id = slot;
    meta.last_use_seq = last_use;
    meta.birth_seq = birth;
    meta.life = life;
    return meta;
}

static HistoricalKvSnapshot snapshot(uint64_t round) {
    HistoricalKvSnapshot s;
    s.round_id = round;
    s.incarnation = 7;
    s.blocks = {block(0, 50, 0), block(1, 5, 2), block(2, 5, 1),
                block(3, 999, 3, HistoricalKvLife::Writing),
                block(4, 0, 0, HistoricalKvLife::Retired),
                block(5, 0, 0), block(6, 0, 0, HistoricalKvLife::Ready)};
    s.blocks[3].active = true;
    s.blocks[3].references = 12;
    s.blocks[3].sealed = false;
    s.blocks[5].handle.incarnation = 6;
    return s;
}

static void test_incremental_full_directory_lru() {
    HistoricalKvLruPolicy policy(8);
    HistoricalKvDecisionBuffer out(4);
    auto s = snapshot(1);
    const HistoricalKvPressure pressure{0, 1, 4, 1};
    const HistoricalKvWorkBudget budget{2, 3};
    for (int i = 0; i < 3; ++i) {
        const auto result = policy.step(s, pressure, budget, out);
        check(result.status == HistoricalKvStepStatus::Yielded &&
              result.examined == 2 && result.emitted == 0 &&
              out.values().empty(), "LRU emitted before scanning full directory");
    }
    const auto result = policy.step(s, pressure, budget, out);
    check(result.status == HistoricalKvStepStatus::Complete &&
          result.examined == 1 && result.emitted == 3,
          "LRU did not complete the bounded full scan");
    check(out.values().size() == 3 &&
          out.values()[0].victim.slot == 2 &&
          out.values()[1].victim.slot == 1 &&
          out.values()[2].victim.slot == 0,
          "LRU order or tie break was incorrect");
    for (const auto & intent : out.values()) {
        check(intent.reason == HistoricalKvDeleteReason::Lru &&
              intent.round_id == 1 &&
              !intent.expected_content_version,
              "default LRU emitted a conditional or mismatched intent");
    }
    check(policy.step(s, pressure, budget, out).emitted == 0,
          "completed round emitted a duplicate decision");
}

static void test_hysteresis_and_pending_credit() {
    HistoricalKvLruPolicy policy(8);
    HistoricalKvDecisionBuffer out(8);
    auto s = snapshot(1);
    const HistoricalKvWorkBudget budget{8, 8};
    auto result = policy.step(s, {1, 2, 4, 3}, budget, out);
    check(result.emitted == 0, "pending credit failed to cover deficit");
    s.round_id = 2;
    result = policy.step(s, {2, 2, 4, 0}, budget, out);
    check(result.emitted == 2, "LRU exited pressure mode below high watermark");
    out.clear_after_ack();
    s.round_id = 3;
    result = policy.step(s, {4, 2, 4, 0}, budget, out);
    check(result.emitted == 0, "LRU ignored high watermark");
    s.round_id = 4;
    result = policy.step(s, {2, 2, 4, 0}, budget, out);
    check(result.emitted == 0, "LRU reentered before crossing low watermark");
    s.round_id = 5;
    result = policy.step(s, {1, 2, 4, 1}, budget, out);
    check(result.emitted == 2, "LRU did not reenter below low watermark");
}

static void test_output_backpressure_and_validation() {
    HistoricalKvLruPolicy policy(8);
    HistoricalKvDecisionBuffer out(1);
    check(out.push({}), "cannot fill decision buffer");
    auto s = snapshot(1);
    const HistoricalKvPressure pressure{0, 1, 2, 0};
    auto result = policy.step(s, pressure, {8, 1}, out);
    check(result.status == HistoricalKvStepStatus::Yielded &&
          result.examined == 0, "full output buffer did not pause the policy");
    out.clear_after_ack();
    result = policy.step(s, pressure, {8, 1}, out);
    check(result.emitted == 1 && out.values()[0].victim.slot == 2,
          "policy did not resume after the manager acknowledged output");

    bool invalid = false;
    try { policy.step(s, {0, 2, 2, 0}, {8, 1}, out); }
    catch (const std::invalid_argument &) { invalid = true; }
    check(invalid, "invalid watermarks were accepted");
    invalid = false;
    try { policy.step(s, pressure, {0, 1}, out); }
    catch (const std::invalid_argument &) { invalid = true; }
    check(invalid, "zero scan budget was accepted");
}

class ActivePolicy final : public HistoricalKvPredicatePolicyTemplate {
    std::optional<HistoricalKvDeleteIntent> decide(
        const HistoricalKvBlockMeta & meta,
        const HistoricalKvSnapshot &,
        const HistoricalKvPressure &) override {
        if (!meta.active) return std::nullopt;
        HistoricalKvDeleteIntent intent;
        intent.reason = HistoricalKvDeleteReason::Custom;
        intent.expected_content_version = meta.content_version;
        return intent;
    }
};

static void test_custom_policy_can_select_active_writing_block() {
    ActivePolicy policy;
    HistoricalKvDecisionBuffer out(2);
    auto s = snapshot(9);
    s.blocks[3].content_version = 17;
    const auto result = policy.step(s, {8, 0, 1, 0}, {8, 2}, out);
    check(result.status == HistoricalKvStepStatus::Complete &&
          result.emitted == 1 && out.values().size() == 1 &&
          out.values()[0].victim.slot == 3 &&
          out.values()[0].round_id == 9 &&
          out.values()[0].expected_content_version == 17,
          "mechanism-style filter rejected an active unsealed referenced block");
}

int main() {
    try {
        test_incremental_full_directory_lru();
        test_hysteresis_and_pending_credit();
        test_output_backpressure_and_validation();
        test_custom_policy_can_select_active_writing_block();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV policy tests passed\n";
}

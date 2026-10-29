#include "kvmem/historical_kv_redundancy.hpp"
#include "kvmem/historical_kv_audit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace kvmem {
namespace {
bool same(HistoricalKvHandle a, HistoricalKvHandle b) {
    return a.incarnation == b.incarnation && a.slot == b.slot && a.generation == b.generation;
}
size_t product(size_t a, size_t b) {
    if (!a || !b || a > SIZE_MAX / b) throw std::invalid_argument("redundancy feature size overflow");
    return a * b;
}
auto age(const HistoricalKvBlockMeta & m) {
    return std::make_tuple(m.original_begin, m.logical_id, m.handle.slot, m.handle.generation);
}
}

void HistoricalKvThinkingRanges::arm() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    armed_ = true;
}
void HistoricalKvThinkingRanges::disarm() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    armed_ = false;
}
bool HistoricalKvThinkingRanges::armed() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return armed_;
}
void HistoricalKvThinkingRanges::clear_spans() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    spans_.clear();
    completed_replies_.clear();
}
void HistoricalKvThinkingRanges::truncate(uint64_t end) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    const auto trim = [end](std::vector<Span> & spans) {
        for (auto & span : spans) span.end = std::min(span.end, end);
        spans.erase(std::remove_if(spans.begin(), spans.end(),
            [](const Span & span) { return span.begin >= span.end; }), spans.end());
    };
    trim(spans_);
    trim(completed_replies_);
    // A restored prefix with no proven thinking must not remain restricted to
    // an empty thinking tier: pressure would have no legal victim.
    if (spans_.empty()) armed_ = false;
}
void HistoricalKvThinkingRanges::enable_completed_replies(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    tiered_ = enabled;
}
bool HistoricalKvThinkingRanges::tiered() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return tiered_;
}
void HistoricalKvThinkingRanges::note_completed_reply(uint64_t begin, uint64_t end) {
    if (end <= begin) return;
    std::lock_guard<std::mutex> lock(mu_);
    // Exact continuation only. Duplicate/overlapping completion must not turn
    // the latest reply into an older one, nor authorize a user-message gap.
    if (!completed_replies_.empty() && begin < completed_replies_.back().end) return;
    completed_replies_.push_back({begin, end});
}
unsigned HistoricalKvThinkingRanges::priority(uint64_t begin, uint32_t tokens) const noexcept {
    if (!tokens || begin > UINT64_MAX - tokens) return 2;
    const uint64_t end = begin + tokens;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & span : spans_)
        if (begin >= span.begin && end <= span.end) return 0;
    if (tiered_ && completed_replies_.size() > 1)
        for (size_t i = 0; i + 1 < completed_replies_.size(); ++i)
            if (begin >= completed_replies_[i].begin && end <= completed_replies_[i].end) return 1;
    return 2;
}
void HistoricalKvThinkingRanges::note(uint64_t begin, uint64_t end) {
    if (end <= begin) return;
    std::lock_guard<std::mutex> lock(mu_);
    for (auto & span : spans_) {
        if (span.begin == begin) {
            if (end > span.end) span.end = end;
            return;
        }
    }
    spans_.push_back({begin, end});
}
bool HistoricalKvThinkingRanges::contains(uint64_t position) const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & span : spans_)
        if (position >= span.begin && position < span.end) return true;
    return false;
}
bool HistoricalKvThinkingRanges::covers(uint64_t begin, uint32_t tokens) const noexcept {
    if (!tokens || begin > UINT64_MAX - tokens) return false;
    const uint64_t end = begin + tokens;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto & span : spans_)
        if (begin >= span.begin && end <= span.end) return true;
    return false;
}

HistoricalKvRedundancyPolicy::HistoricalKvRedundancyPolicy(size_t capacity,
    std::shared_ptr<HistoricalKvRedundancySource> source, Config config)
    : capacity_(capacity), dimensions_(source ? source->dimensions() : 0),
      source_(std::move(source)), config_(config) {
    audit_ = historical_kv_audit_enabled();
    if (config_.pool_blocks > capacity_) config_.pool_blocks = capacity_;
    if (!config_.segment_blocks) config_.segment_blocks = 1;
    if (config_.pool_blocks && config_.pool_blocks < config_.segment_blocks)
        config_.pool_blocks = std::min(capacity_, config_.segment_blocks);
    if (!source_ || !capacity_ || !dimensions_ || !config_.reference_blocks ||
        (config_.max_candidate_blocks &&
         (config_.max_candidate_blocks < 2 || config_.max_candidate_blocks > capacity_)) ||
        !std::isfinite(config_.importance_weight) || config_.importance_weight < 0 || config_.importance_weight > 1 ||
        !std::isfinite(config_.representative_similarity) || config_.representative_similarity < 0 ||
        config_.representative_similarity > 1)
        throw std::invalid_argument("invalid historical redundancy configuration");
    rows_.reserve(capacity_); references_.reserve(capacity_); ranked_.reserve(capacity_);
    candidate_indices_.reserve(capacity_); candidate_scratch_.reserve(capacity_);
    protected_indices_.reserve(capacity_);
    neighbors_.reserve(capacity_);
    seen_.assign(capacity_, {});
    descriptors_.resize(product(capacity_, dimensions_));
}

bool HistoricalKvRedundancyPolicy::permits(const HistoricalKvBlockMeta & meta) const noexcept {
    return meta.life == HistoricalKvLife::Live && meta.sealed && !meta.mandatory &&
        source_->question_epoch() == question_epoch_ &&
        (!thinking_mode_ || config_.thinking->priority(meta.original_begin, meta.valid_tokens) == content_tier_);
}

bool HistoricalKvRedundancyPolicy::carries_retired_substitute(
        const HistoricalKvBlockMeta & meta) const noexcept {
    if (meta.handle.slot >= seen_.size()) return false;
    const auto & seen = seen_[meta.handle.slot];
    return seen.carries_retired && seen.generation == meta.handle.generation &&
        seen.content_version == meta.content_version;
}

void HistoricalKvRedundancyPolicy::on_result(const HistoricalKvCommitResult & result) noexcept {
    const auto & intent = result.intent;
    if (result.code != HistoricalKvCommitCode::Committed ||
        intent.reason != HistoricalKvDeleteReason::Redundancy ||
        intent.round_id != round_ || intent.victim.incarnation != incarnation_ ||
        intent.victim.slot >= seen_.size() || !intent.expected_content_version) return;
    const auto & victim = seen_[intent.victim.slot];
    if (victim.generation != intent.victim.generation ||
        victim.content_version != *intent.expected_content_version ||
        victim.mate_slot >= seen_.size() ||
        victim.mate_cosine < config_.representative_similarity) return;
    auto & witness = seen_[victim.mate_slot];
    if (witness.generation == victim.mate_generation &&
        witness.content_version == victim.mate_version)
        witness.carries_retired = true;
}

bool HistoricalKvRedundancyPolicy::thinking_block(const HistoricalKvBlockMeta & meta) const noexcept {
    return config_.thinking && config_.thinking->covers(meta.original_begin, meta.valid_tokens);
}

bool HistoricalKvRedundancyPolicy::eligible_candidate(const HistoricalKvBlockMeta & meta) const noexcept {
    if (meta.handle.incarnation != incarnation_ ||
        meta.life != HistoricalKvLife::Live || !meta.sealed) return false;
    if (thinking_mode_ && config_.thinking->priority(meta.original_begin, meta.valid_tokens) != content_tier_) return false;
    return permits(meta) || meta.mandatory;
}

bool HistoricalKvRedundancyPolicy::better_representative(size_t a, size_t b) const noexcept {
    const auto & xa = rows_[a];
    const auto & xb = rows_[b];
    if (xa.meta->mandatory != xb.meta->mandatory) return xa.meta->mandatory;
    if (xa.importance != xb.importance) return xa.importance > xb.importance;
    return age(*xa.meta) > age(*xb.meta);
}

void HistoricalKvRedundancyPolicy::select_thinking_pool() {
    const size_t n = candidate_indices_.size();
    // Mandatory thinking blocks are useful substitute references, but they
    // still consume feature reads and pair comparisons in this round.
    const size_t cap = config_.pool_blocks
        ? config_.pool_blocks - std::min(config_.pool_blocks, protected_indices_.size()) : n;
    if (!cap) {
        candidate_indices_.clear();
        return;
    }
    const size_t segment = std::min({config_.segment_blocks, n, cap});
    const size_t older = n - segment;
    if (n <= cap) return;
    const size_t older_slots = cap - segment;
    candidate_scratch_.clear();
    if (older_slots && older) {
        if (older_slots >= older) {
            candidate_scratch_.insert(candidate_scratch_.end(),
                candidate_indices_.begin(), candidate_indices_.begin() + older);
        } else {
            const size_t start_max = older - older_slots;
            const size_t start = older_origin_ % (start_max + 1);
            candidate_scratch_.insert(candidate_scratch_.end(),
                candidate_indices_.begin() + start, candidate_indices_.begin() + start + older_slots);
            const size_t stride = std::max<size_t>(older_slots / 2, 1);
            // Visit the tail even when the stride does not divide start_max;
            // otherwise the last few old blocks can be excluded forever.
            older_origin_ = start == start_max ? 0 : std::min(start + stride, start_max);
        }
    }
    candidate_scratch_.insert(candidate_scratch_.end(),
        candidate_indices_.begin() + older, candidate_indices_.end());
    candidate_indices_.swap(candidate_scratch_);
}

void HistoricalKvRedundancyPolicy::limit_older_candidates(size_t limit) {
    if (candidate_indices_.size() <= limit) return;
    if (!limit) {
        candidate_indices_.clear();
        return;
    }
    const size_t anchors = std::min(config_.reference_blocks, limit / 4);
    const size_t older = candidate_indices_.size() - anchors;
    const size_t bands = limit - anchors;
    candidate_scratch_.clear();
    if (bands && older) {
        for (size_t i = 0; i < bands; ++i) {
            const size_t begin = i * older / bands;
            const size_t end = (i + 1) * older / bands;
            if (begin >= end) continue;
            candidate_scratch_.push_back(candidate_indices_[begin + round_ % (end - begin)]);
        }
    }
    candidate_scratch_.insert(candidate_scratch_.end(),
        candidate_indices_.begin() + older, candidate_indices_.end());
    candidate_indices_.swap(candidate_scratch_);
}

void HistoricalKvRedundancyPolicy::refresh_question_peaks() {
    const uint64_t epoch = source_->question_epoch();
    if (epoch == question_epoch_) return;
    question_epoch_ = epoch;
    for (auto & seen : seen_) seen.peak = -std::numeric_limits<double>::infinity();
}

double HistoricalKvRedundancyPolicy::peaked_logit(const HistoricalKvBlockMeta & meta, double logit) const noexcept {
    if (meta.handle.slot >= seen_.size()) return logit;
    const auto & seen = seen_[meta.handle.slot];
    if (seen.generation != meta.handle.generation || seen.content_version != meta.content_version)
        return logit;
    return std::isfinite(seen.peak) ? std::max(logit, seen.peak) : logit;
}

size_t HistoricalKvRedundancyPolicy::row_for(uint32_t slot, uint64_t generation, uint64_t version) const noexcept {
    for (size_t i = 0; i < rows_.size(); ++i) {
        const auto & meta = *rows_[i].meta;
        if (meta.handle.slot == slot && meta.handle.generation == generation &&
            meta.content_version == version) return i;
    }
    return SIZE_MAX;
}

size_t HistoricalKvRedundancyPolicy::snapshot_index(const HistoricalKvSnapshot & snapshot, uint32_t slot,
    uint64_t generation, uint64_t version) const noexcept {
    for (size_t i = 0; i < snapshot.blocks.size(); ++i) {
        const auto & meta = snapshot.blocks[i];
        if (meta.handle.slot == slot && meta.handle.generation == generation &&
            meta.content_version == version) return i;
    }
    return SIZE_MAX;
}

void HistoricalKvRedundancyPolicy::add_remembered_references() {
    const size_t cap = references_.size() + config_.reference_blocks;
    for (size_t i = 0; i < rows_.size() && references_.size() < cap; ++i) {
        const auto & meta = *rows_[i].meta;
        if (meta.handle.slot >= seen_.size()) continue;
        const auto & seen = seen_[meta.handle.slot];
        if (seen.generation != meta.handle.generation || seen.content_version != meta.content_version ||
            seen.mate_slot == UINT32_MAX || seen.mate_cosine < config_.representative_similarity) continue;
        const size_t mate = row_for(seen.mate_slot, seen.mate_generation, seen.mate_version);
        if (mate == SIZE_MAX) continue;
        if (std::find(references_.begin(), references_.end(), mate) != references_.end()) continue;
        references_.push_back(mate);
    }
}

void HistoricalKvRedundancyPolicy::select_references() {
    references_.resize(rows_.size());
    std::iota(references_.begin(), references_.end(), 0);
    std::sort(references_.begin(), references_.end(), [&](size_t a, size_t b) {
        return age(*rows_[a].meta) > age(*rows_[b].meta);
    });
    if (config_.max_candidate_blocks || thinking_mode_ || references_.size() <= config_.reference_blocks) {
        add_remembered_references();
        return;
    }
    const size_t newest = std::max<size_t>(config_.reference_blocks / 2, 1);
    const size_t older_limit = config_.reference_blocks - newest;
    std::vector<size_t> older(references_.begin() + newest, references_.end());
    std::reverse(older.begin(), older.end());
    candidate_scratch_.assign(references_.begin(), references_.begin() + newest);
    if (older_limit && !older.empty()) {
        if (older.size() <= older_limit) {
            candidate_scratch_.insert(candidate_scratch_.end(), older.begin(), older.end());
        } else {
            for (size_t i = 0; i < older_limit; ++i) {
                const size_t begin = i * older.size() / older_limit;
                const size_t end = (i + 1) * older.size() / older_limit;
                if (begin >= end) continue;
                candidate_scratch_.push_back(older[begin + round_ % (end - begin)]);
            }
        }
    }
    references_.swap(candidate_scratch_);
    add_remembered_references();
}

void HistoricalKvRedundancyPolicy::pin_remembered_neighbors(const HistoricalKvSnapshot & snapshot) {
    const size_t extra_cap = std::max(config_.segment_blocks, size_t(4));
    auto in_pool = [&](size_t index) {
        return std::find(candidate_indices_.begin(), candidate_indices_.end(), index) != candidate_indices_.end();
    };
    auto remember_ok = [&](const HistoricalKvBlockMeta & meta, const Seen & seen) {
        return meta.handle.slot < seen_.size() &&
            seen.generation == meta.handle.generation && seen.content_version == meta.content_version &&
            seen.mate_slot != UINT32_MAX && seen.mate_cosine >= config_.representative_similarity;
    };
    size_t added = 0;
    const size_t start = candidate_indices_.size();
    for (size_t n = 0; n < start && added < extra_cap; ++n) {
        const auto & meta = snapshot.blocks[candidate_indices_[n]];
        if (meta.handle.slot >= seen_.size()) continue;
        const auto & seen = seen_[meta.handle.slot];
        if (!remember_ok(meta, seen)) continue;
        const size_t index = snapshot_index(snapshot, seen.mate_slot, seen.mate_generation, seen.mate_version);
        if (index == SIZE_MAX || !eligible_candidate(snapshot.blocks[index]) || in_pool(index)) continue;
        candidate_indices_.push_back(index);
        ++added;
    }
    for (size_t i = 0; i < snapshot.blocks.size() && added < extra_cap; ++i) {
        const auto & meta = snapshot.blocks[i];
        if (!eligible_candidate(meta) || in_pool(i) || meta.handle.slot >= seen_.size()) continue;
        const auto & seen = seen_[meta.handle.slot];
        if (!remember_ok(meta, seen)) continue;
        const size_t mate = snapshot_index(snapshot, seen.mate_slot, seen.mate_generation, seen.mate_version);
        if (mate == SIZE_MAX || !in_pool(mate)) continue;
        candidate_indices_.push_back(i);
        ++added;
    }
}

void HistoricalKvRedundancyPolicy::remember_scored() {
    for (size_t i = 0; i < rows_.size(); ++i) {
        const auto & meta = *rows_[i].meta;
        if (meta.handle.slot >= seen_.size()) continue;
        auto & seen = seen_[meta.handle.slot];
        if (seen.generation != meta.handle.generation || seen.content_version != meta.content_version) {
            seen = {};
            seen.generation = meta.handle.generation;
            seen.content_version = meta.content_version;
        }
        seen.peak = std::max(seen.peak, rows_[i].importance);
        if (rows_[i].mate == SIZE_MAX || rows_[i].max_sim < config_.representative_similarity) continue;
        const auto & other = *rows_[rows_[i].mate].meta;
        seen.mate_slot = other.handle.slot;
        seen.mate_generation = other.handle.generation;
        seen.mate_version = other.content_version;
        seen.mate_cosine = rows_[i].max_sim;
    }
}

HistoricalKvStepResult HistoricalKvRedundancyPolicy::step(const HistoricalKvSnapshot & snapshot,
    const HistoricalKvPressure & pressure, HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer & out) {
    if (snapshot.blocks.size() > capacity_ || !budget.scan || !budget.intents ||
        !(pressure.low < pressure.high) || pressure.high > capacity_)
        throw std::invalid_argument("invalid historical redundancy step");
    if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round_id) {
        if (!initialized_ || incarnation_ != snapshot.incarnation) {
            pressure_mode_ = false;
            older_origin_ = 0;
            seen_.assign(capacity_, {});
            question_epoch_ = source_->question_epoch();
        } else {
            refresh_question_peaks();
        }
        thinking_mode_ = config_.thinking && (config_.thinking->armed() || config_.thinking->tiered());
        content_tier_ = 0;
        if (thinking_mode_ && config_.thinking->tiered()) {
            // Exhaust eligible thinking before considering older replies.
            // One tier per round retains the existing bounded comparison N.
            bool has_thinking = false;
            for (const auto & meta : snapshot.blocks)
                if (meta.handle.incarnation == snapshot.incarnation &&
                    meta.life == HistoricalKvLife::Live && meta.sealed && !meta.mandatory &&
                    config_.thinking->priority(meta.original_begin, meta.valid_tokens) == 0) {
                    has_thinking = true; break;
                }
            if (!has_thinking) content_tier_ = 1;
        }
        initialized_ = true; incarnation_ = snapshot.incarnation; round_ = snapshot.round_id;
        cursor_ = pair_row_ = pair_reference_ = refresh_cursor_ = 0;
        done_ = collected_ = source_ready_ = pool_built_ = false;
        refresh_started_ = false;
        rows_.clear(); references_.clear(); ranked_.clear();
        candidate_indices_.clear(); protected_indices_.clear();
    }
    if (done_) return {HistoricalKvStepStatus::Complete, 0, 0};
    if (source_->question_epoch() != question_epoch_) {
        done_ = true;
        return {HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
    if (pressure.ready < pressure.low) pressure_mode_ = true;
    if (pressure.ready >= pressure.high) pressure_mode_ = false;
    const uint64_t target = std::max(pressure_mode_ ? pressure.high : 0u, pressure.requested_ready);
    const uint64_t supply = uint64_t(pressure.ready) + pressure.pending_slot_credit;
    if (target <= supply) { done_ = true; return {HistoricalKvStepStatus::Complete, 0, 0}; }
    if (!out.remaining()) return {HistoricalKvStepStatus::Yielded, 0, 0};
    if (!source_ready_) {
        if (!source_->begin_round(snapshot.incarnation)) {
            ++no_query_; done_ = true;
            return {HistoricalKvStepStatus::NoCandidate, 0, 0};
        }
        source_ready_ = true;
    }
    const bool index_pool = config_.max_candidate_blocks || thinking_mode_;
    if (!pool_built_ && index_pool) {
        pool_built_ = true;
        // As in a budgeted compression round, only a bounded pool is scored.
        // Keep a recent anchor set and sample the older history in age bands.
        // Advancing the pick within each band across rounds prevents the same
        // old blocks from being excluded forever when no deletion succeeds.
        // Armed thinking rounds instead keep the newest segment and slide a
        // contiguous window across older thinking blocks, then compare the
        // whole pool. Prompt and answer blocks stay out of that matrix.
        candidate_indices_.clear();
        protected_indices_.clear();
        for (size_t i = 0; i < snapshot.blocks.size(); ++i) {
            if (!eligible_candidate(snapshot.blocks[i])) continue;
            if (snapshot.blocks[i].mandatory) protected_indices_.push_back(i);
            else candidate_indices_.push_back(i);
        }
        const auto by_age = [&](size_t a, size_t b) {
            return age(snapshot.blocks[a]) < age(snapshot.blocks[b]);
        };
        std::sort(candidate_indices_.begin(), candidate_indices_.end(), by_age);
        std::sort(protected_indices_.begin(), protected_indices_.end(), by_age);
        // Protected references also consume the comparison budget, including
        // when only max_candidate_blocks is enabled. Keep room for evictable
        // blocks instead of letting a mandatory prefix exceed N or starve GC.
        const size_t core_cap = thinking_mode_ ? config_.pool_blocks : config_.max_candidate_blocks;
        if (core_cap) {
            const size_t protected_cap = std::max<size_t>(1, core_cap / 2);
            if (protected_indices_.size() > protected_cap) {
                candidate_scratch_.clear();
                const size_t n = protected_indices_.size();
                for (size_t band = 0; band < protected_cap; ++band) {
                    const size_t begin = band * n / protected_cap;
                    const size_t end = (band + 1) * n / protected_cap;
                    candidate_scratch_.push_back(protected_indices_[begin + round_ % (end - begin)]);
                }
                protected_indices_.swap(candidate_scratch_);
            }
        }
        if (thinking_mode_) {
            select_thinking_pool();
        }
        else if (config_.max_candidate_blocks) {
            const size_t others = config_.max_candidate_blocks > protected_indices_.size()
                ? config_.max_candidate_blocks - protected_indices_.size() : 0;
            limit_older_candidates(others);
        }
        candidate_scratch_ = protected_indices_;
        candidate_scratch_.insert(candidate_scratch_.end(),
            candidate_indices_.begin(), candidate_indices_.end());
        candidate_indices_.swap(candidate_scratch_);
        if (thinking_mode_) pin_remembered_neighbors(snapshot);
    }
    size_t examined = 0;
    const size_t candidate_count = index_pool ? candidate_indices_.size() : snapshot.blocks.size();
    while (!collected_ && cursor_ < candidate_count && examined < budget.scan) {
        const auto & meta = snapshot.blocks[index_pool ? candidate_indices_[cursor_] : cursor_]; ++examined;
        if (meta.life != HistoricalKvLife::Live || !meta.sealed || meta.handle.incarnation != incarnation_) {
            ++cursor_;
            continue;
        }
        double logit = 0;
        float * descriptor = descriptors_.data() + rows_.size() * dimensions_;
        const auto result = source_->read_bounded(meta, descriptor, logit);
        if (result == HistoricalKvRedundancySource::ReadResult::Yielded)
            return {HistoricalKvStepStatus::Yielded, examined, 0};
        ++cursor_;
        if (result != HistoricalKvRedundancySource::ReadResult::Ready || !std::isfinite(logit)) {
            ++missing_; continue;
        }
        double norm = 0;
        for (size_t d = 0; d < dimensions_; ++d) norm += double(descriptor[d]) * descriptor[d];
        if (!std::isfinite(norm) || norm < 1e-16) { ++missing_; continue; }
        for (size_t d = 0; d < dimensions_; ++d) descriptor[d] /= float(std::sqrt(norm));
        rows_.push_back({&meta, peaked_logit(meta, logit)}); ++feature_blocks_;
    }
    if (!collected_) {
        if (cursor_ < candidate_count) return {HistoricalKvStepStatus::Yielded, examined, 0};
        collected_ = true;
        uint64_t peak = max_round_features_.load();
        while (rows_.size() > peak && !max_round_features_.compare_exchange_weak(peak, rows_.size())) {}
        if (rows_.empty()) { done_ = true; return {HistoricalKvStepStatus::NoCandidate, examined, 0}; }
        select_references();
        neighbors_.assign(rows_.size(), {});
        return {HistoricalKvStepStatus::Yielded, examined, 0};
    }
    // Default mode compares against newest plus age-banded older references,
    // then any remembered substitutes. Bounded/thinking modes compare the
    // selected pool. Neither allocates an N*N matrix.
    const size_t pair_budget = budget.scan > SIZE_MAX / 32 ? SIZE_MAX : budget.scan * 32;
    size_t pairs = 0;
    while (pair_row_ < rows_.size() && pairs < pair_budget) {
        auto & row = rows_[pair_row_];
        const size_t other = references_[pair_reference_++];
        ++pairs;
        if (other != pair_row_) {
            const auto * a = descriptors_.data() + pair_row_ * dimensions_;
            const auto * b = descriptors_.data() + other * dimensions_;
            double cosine = 0;
            for (size_t d = 0; d < dimensions_; ++d) cosine += double(a[d]) * b[d];
            cosine = std::clamp(cosine, -1.0, 1.0);
            ++row.compared; ++comparisons_;
            if (cosine > row.max_sim) {
                row.max_sim = cosine;
                row.mate = other;
            } else if (row.mate != SIZE_MAX && cosine == row.max_sim &&
                    better_representative(other, row.mate)) {
                row.mate = other;
            }
            if (cosine >= config_.representative_similarity) {
                neighbors_[pair_row_].push_back({other, cosine});
                neighbors_[other].push_back({pair_row_, cosine});
            }
        }
        if (pair_reference_ == references_.size()) { ++pair_row_; pair_reference_ = 0; }
    }
    if (pair_row_ < rows_.size()) return {HistoricalKvStepStatus::Yielded, pairs, 0};
    if (source_->refresh_before_rank()) {
        // Full-question scoring may span many worker polls. Snapshot the
        // latest generated Query *after* that work and pair comparisons, so
        // representative selection and deletion do not use the old focus.
        // Question scores are already cached; do not repeat their dot products.
        if (!refresh_started_) {
            if (!source_->begin_round(snapshot.incarnation))
                return {HistoricalKvStepStatus::Yielded, pairs, 0};
            refresh_started_ = true;
            return {HistoricalKvStepStatus::Yielded, pairs, 0};
        }
        size_t refreshed = 0;
        while (refresh_cursor_ < rows_.size() && refreshed < budget.scan) {
            auto & row = rows_[refresh_cursor_];
            double logit = 0;
            const auto result = source_->read_bounded(*row.meta,
                descriptors_.data() + refresh_cursor_ * dimensions_, logit);
            ++refreshed;
            if (result == HistoricalKvRedundancySource::ReadResult::Yielded)
                return {HistoricalKvStepStatus::Yielded, refreshed, 0};
            if (result != HistoricalKvRedundancySource::ReadResult::Ready || !std::isfinite(logit)) {
                ++missing_;
                done_ = true;
                return {HistoricalKvStepStatus::NoCandidate, refreshed, 0};
            }
            // Preserve a need observed earlier in this round even if that
            // observation has since aged out of the short generated rings.
            row.importance = std::max(row.importance, peaked_logit(*row.meta, logit));
            ++refresh_cursor_;
        }
        if (refresh_cursor_ != rows_.size())
            return {HistoricalKvStepStatus::Yielded, refreshed, 0};
        pairs = refreshed;
    }
    for (auto & row : rows_) {
        row.redundancy = row.compared ? row.max_sim : -1.0;
        row.retained = false;
    }
    // Similarity is not transitive: A~B and B~C do not establish A~C.
    // Keep a stronger row before classifying a weaker one as redundant, and
    // require a direct, retained witness for every redundancy intent. Reuse
    // only edges proven by the budgeted comparison pass, including reverse
    // edges when a non-reference row compared against a retained reference.
    std::vector<size_t> by_preference(rows_.size());
    std::iota(by_preference.begin(), by_preference.end(), 0);
    std::sort(by_preference.begin(), by_preference.end(), [&](size_t a, size_t b) {
        if (better_representative(a, b)) return true;
        if (better_representative(b, a)) return false;
        return a < b;
    });
    std::vector<uint8_t> supports(rows_.size(), 0);
    std::vector<size_t> prior_witness_fallback;
    ranked_.clear();
    candidate_scratch_.clear();
    for (size_t i : by_preference) {
        auto & row = rows_[i];
        if (!permits(*row.meta) || carries_retired_substitute(*row.meta)) {
            row.retained = true;
            continue;
        }
        size_t witness = SIZE_MAX;
        double similarity = config_.representative_similarity;
        for (const auto & edge : neighbors_[i]) {
            const size_t other = edge.index;
            if (other == i || !rows_[other].retained) continue;
            if (witness == SIZE_MAX || edge.similarity > similarity ||
                (edge.similarity == similarity && better_representative(other, witness))) {
                witness = other;
                similarity = edge.similarity;
            }
        }
        if (witness == SIZE_MAX) {
            row.retained = true;
            continue;
        }
        row.redundancy = row.max_sim = similarity;
        row.mate = witness;
        supports[witness] = 1;
        ranked_.push_back(i);
    }
    for (size_t i = 0; i < rows_.size(); ++i) {
        if (!rows_[i].retained) continue;
        if (supports[i]) ++representatives_;
        // A witness for a redundancy intent must not also enter this round's
        // lossy fallback when capacity pressure exceeds the redundant supply.
        else if (permits(*rows_[i].meta)) {
            if (carries_retired_substitute(*rows_[i].meta)) prior_witness_fallback.push_back(i);
            else candidate_scratch_.push_back(i);
        }
    }
    remember_scored();
    ++scored_;
    const size_t need = std::min({size_t(target - supply), budget.intents, out.remaining()});
    size_t emitted = emit_ranked(ranked_, need, HistoricalKvDeleteReason::Redundancy, snapshot, out);
    // Admission wait still has the previous request's Query. Only retire
    // blocks that already have a substitute; do not infer future need.
    if (emitted < need && pressure.requested_ready == 0) {
        const size_t extra = emit_ranked(candidate_scratch_, need - emitted,
            HistoricalKvDeleteReason::Lru, snapshot, out);
        emitted += extra;
        lossy_ += extra;
        // A witness from an earlier committed deletion is a last resort:
        // retire an independent block first, otherwise its old victim loses
        // its only known direct substitute. This final tier is still lossy.
        if (emitted < need) {
            const size_t dependent = emit_ranked(prior_witness_fallback, need - emitted,
                HistoricalKvDeleteReason::Lru, snapshot, out);
            emitted += dependent;
            lossy_ += dependent;
        }
    }
    done_ = true;
    return {emitted ? HistoricalKvStepStatus::Complete : HistoricalKvStepStatus::NoCandidate, pairs, emitted};
}

size_t HistoricalKvRedundancyPolicy::emit_ranked(std::vector<size_t> & group, size_t take,
    HistoricalKvDeleteReason reason, const HistoricalKvSnapshot & snapshot,
    HistoricalKvDecisionBuffer & out) {
    if (!take || group.empty()) return 0;
    const size_t n = std::min(take, group.size());
    if (reason == HistoricalKvDeleteReason::Redundancy) {
        double max_importance = -std::numeric_limits<double>::infinity();
        double max_redundancy = -std::numeric_limits<double>::infinity();
        for (size_t i : group) {
            max_importance = std::max(max_importance, rows_[i].importance);
            max_redundancy = std::max(max_redundancy, rows_[i].redundancy);
        }
        std::vector<double> importance(group.size()), redundancy(group.size());
        double importance_sum = 0, redundancy_sum = 0;
        for (size_t k = 0; k < group.size(); ++k) {
            importance[k] = std::exp(rows_[group[k]].importance - max_importance);
            redundancy[k] = std::exp(rows_[group[k]].redundancy - max_redundancy);
            importance_sum += importance[k];
            redundancy_sum += redundancy[k];
        }
        for (size_t k = 0; k < group.size(); ++k)
            rows_[group[k]].score = config_.importance_weight * importance[k] / importance_sum -
                (1.0 - config_.importance_weight) * redundancy[k] / redundancy_sum;
        std::partial_sort(group.begin(), group.begin() + n, group.end(), [&](size_t a, size_t b) {
            return std::make_tuple(rows_[a].score, rows_[a].meta->last_use_seq, age(*rows_[a].meta)) <
                   std::make_tuple(rows_[b].score, rows_[b].meta->last_use_seq, age(*rows_[b].meta));
        });
    } else {
        std::partial_sort(group.begin(), group.begin() + n, group.end(), [&](size_t a, size_t b) {
            return std::make_tuple(rows_[a].importance, rows_[a].meta->last_use_seq, age(*rows_[a].meta)) <
                   std::make_tuple(rows_[b].importance, rows_[b].meta->last_use_seq, age(*rows_[b].meta));
        });
    }
    for (size_t i = 0; i < n; ++i) {
        const auto & row = rows_[group[i]];
        const auto & meta = *row.meta;
        const bool pushed = out.push({meta.handle, snapshot.round_id, reason, meta.content_version});
        if (!pushed || !audit_) continue;
        const bool redundant = reason == HistoricalKvDeleteReason::Redundancy;
        const Row * witness = redundant && row.mate < rows_.size() && rows_[row.mate].retained
            ? &rows_[row.mate] : nullptr;
        const HistoricalKvBlockMeta empty{};
        const auto & other = witness ? *witness->meta : empty;
        // One stdio call keeps each event a single line across worker threads.
        // This is a sampled direct witness at proposal time, not a claim that
        // text semantics or a witness's later lifetime are interchangeable.
        std::fprintf(stderr, "KVMEM_GC_AUDIT event=proposed incarnation=%llu slot=%u generation=%llu content_version=%llu round=%llu original_begin=%llu valid_tokens=%u reason=%s intent_reason=%s importance=%.17g redundancy=%.17g score=%.17g score_kind=%s threshold=%.17g thinking_mode=%d thinking_full=%d content_tier=%u witness=%d witness_incarnation=%llu witness_slot=%u witness_generation=%llu witness_content_version=%llu witness_original_begin=%llu witness_valid_tokens=%u witness_cosine=%.17g witness_retained=%d witness_thinking_full=%d\n",
            (unsigned long long)meta.handle.incarnation, meta.handle.slot,
            (unsigned long long)meta.handle.generation, (unsigned long long)meta.content_version,
            (unsigned long long)snapshot.round_id, (unsigned long long)meta.original_begin, meta.valid_tokens,
            redundant ? "redundancy" : "capacity_fallback", historical_kv_audit_reason(reason),
            row.importance, row.redundancy, redundant ? row.score : row.importance,
            redundant ? "joint" : "importance", config_.representative_similarity,
            int(thinking_mode_), int(thinking_block(meta)), content_tier_, int(witness != nullptr),
            (unsigned long long)other.handle.incarnation, other.handle.slot,
            (unsigned long long)other.handle.generation, (unsigned long long)other.content_version,
            (unsigned long long)other.original_begin, other.valid_tokens, witness ? row.max_sim : -1.0,
            int(witness != nullptr), int(witness && thinking_block(other)));
    }
    return n;
}

HistoricalKvPolicy::Metrics HistoricalKvRedundancyPolicy::metrics() const noexcept {
    return {feature_blocks_.load(), missing_.load(), comparisons_.load(), scored_.load(),
            representatives_.load(), no_query_.load(), lossy_.load(), max_round_features_.load()};
}
} // namespace kvmem

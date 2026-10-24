#include "kvmem/historical_kv_redundancy.hpp"

#include <algorithm>
#include <cmath>
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

HistoricalKvMeanKeySource::HistoricalKvMeanKeySource(
    std::shared_ptr<HistoricalKvRepository> repository,
    std::shared_ptr<HistoricalKvRawStore> storage, uint32_t dimension,
    uint32_t kv_heads, uint32_t query_heads)
    : repository_(std::move(repository)), storage_(std::move(storage)),
      head_dimension_(dimension), kv_heads_(kv_heads), query_heads_(query_heads) {
    if (!repository_ || !storage_ || !dimension || !kv_heads || query_heads < kv_heads ||
        query_heads % kv_heads || storage_->config().mean_k_elements != product(dimension, kv_heads) ||
        storage_->config().attention_layers.empty())
        throw std::invalid_argument("redundancy requires complete Mean-K and compatible GQA dimensions");
    const auto & all = storage_->config().attention_layers;
    const size_t count = std::min<size_t>(4, all.size());
    for (size_t i = 0; i < count; ++i)
        layers_.push_back(all[count == 1 ? 0 : i * (all.size() - 1) / (count - 1)]);
    dimensions_ = product(layers_.size(), product(dimension, kv_heads));
    query_width_ = product(dimension, query_heads);
    cached_.resize(repository_->capacity());
    question_scores_.resize(repository_->capacity());
    keys_.resize(product(cached_.size(), dimensions_));
    set_query_windows(32, 32, 8);
}

bool HistoricalKvMeanKeySource::uses_layer(uint32_t layer) const noexcept {
    return std::find(layers_.begin(), layers_.end(), layer) != layers_.end();
}

void HistoricalKvMeanKeySource::reset_ring(Ring & ring) {
    const size_t layers = layers_.size();
    ring.samples.assign(product(product(layers, ring.window), query_width_), 0.f);
    ring.mean.assign(product(layers, query_width_), 0.f);
    ring.count.assign(layers, 0);
    ring.cursor.assign(layers, 0);
    ring.end.assign(layers, 0);
    ring.accepted.assign(layers, 0);
    ring.valid = false;
    ring.batch_end = 0;
    ring.published_accepted = 0;
}

void HistoricalKvMeanKeySource::reset_observations() {
    reset_ring(observation_);
    reset_ring(observation_fast_);
    observation_keep_count_ = 0;
    observation_keep_head_ = 0;
}

void HistoricalKvMeanKeySource::remember_observation() {
    const size_t width = observation_.mean.size();
    if (!width) return;
    if (observation_keep_.size() != kObservationKeep * width)
        observation_keep_.assign(kObservationKeep * width, 0.f);
    std::copy(observation_.mean.begin(), observation_.mean.end(),
        observation_keep_.begin() + observation_keep_head_ * width);
    observation_keep_head_ = (observation_keep_head_ + 1) % kObservationKeep;
    if (observation_keep_count_ < kObservationKeep) ++observation_keep_count_;
}

void HistoricalKvMeanKeySource::set_query_windows(uint32_t question_segment_tokens, uint32_t observation_tokens,
                                                 uint32_t fast_tokens) {
    if (!question_segment_tokens || !observation_tokens || !fast_tokens)
        throw std::invalid_argument("redundancy query window must be positive");
    std::lock_guard<std::mutex> lock(query_mutex_);
    question_segment_tokens_ = question_segment_tokens;
    observation_.window = observation_tokens;
    observation_fast_.window = fast_tokens;
    reset_question_locked();
    question_.reset();
    question_collecting_ = false;
    question_building_.reset();
    reset_observations();
    question_round_.reset();
    observation_keep_round_.assign(kObservationKeep * observation_.mean.size(), 0.f);
    fast_round_.assign(observation_fast_.mean.size(), 0.f);
    fast_round_valid_ = false;
    observation_keep_round_count_ = 0;
    retire_observation_ = false;
}

void HistoricalKvMeanKeySource::set_question_byte_budget(size_t bytes) {
    if (!bytes) throw std::invalid_argument("question byte budget must be positive");
    std::lock_guard<std::mutex> lock(query_mutex_);
    if (question_collecting_) throw std::logic_error("cannot change an active question budget");
    question_byte_budget_ = bytes;
}

void HistoricalKvMeanKeySource::reset_question_locked() {
    // Keep the last immutable bank only until the caller establishes whether
    // this is the same exact-prefix question. It is hidden while collecting.
    question_building_ = std::make_shared<Question>();
    question_building_->epoch = ++question_epoch_;
    question_building_->layers.resize(layers_.size());
    question_collecting_ = true;
    question_invalid_ = false;
    question_tokens_ = question_segments_ = question_bytes_ = 0;
}

bool HistoricalKvMeanKeySource::append_ring(Ring & ring, uint32_t layer, uint64_t end,
                                            const float * rows, uint32_t count) {
    const auto it = std::find(layers_.begin(), layers_.end(), layer);
    if (it == layers_.end() || !rows || !count || uint64_t(count) > end) return false;
    const size_t index = size_t(it - layers_.begin());
    if (ring.batch_end != end) {
        // A nonblocking producer can miss one layer while GC snapshots the
        // Query. Reset all layers *before* accepting the next batch, otherwise
        // their accepted counts diverge permanently and no new Query publishes.
        bool gap = false;
        for (size_t l = 0; l < layers_.size(); ++l) {
            if (ring.count[l] != ring.count[0] || ring.accepted[l] != ring.accepted[0] ||
                (ring.count[l] && end - count != ring.end[l])) {
                gap = true;
                break;
            }
        }
        if (gap) reset_ring(ring);
        ring.batch_end = end;
    }
    for (uint32_t t = 0; t < count; ++t) {
        float * dst = ring.samples.data() + (index * ring.window + ring.cursor[index]) * query_width_;
        const float * src = rows + size_t(t) * query_width_;
        std::copy(src, src + query_width_, dst);
        ring.cursor[index] = (ring.cursor[index] + 1) % ring.window;
        if (ring.count[index] < ring.window) ++ring.count[index];
        ++ring.accepted[index];
    }
    ring.end[index] = end;
    return true;
}

void HistoricalKvMeanKeySource::ring_mean(const Ring & ring, size_t layer, float * dst) const {
    const uint32_t n = ring.count[layer];
    std::fill(dst, dst + query_width_, 0.f);
    if (!n) return;
    uint32_t index = (ring.cursor[layer] + ring.window - n) % ring.window;
    for (uint32_t t = 0; t < n; ++t) {
        const float * src = ring.samples.data() + (layer * ring.window + index) * query_width_;
        for (size_t d = 0; d < query_width_; ++d) dst[d] += src[d];
        index = (index + 1) % ring.window;
    }
    const float scale = 1.f / float(n);
    for (size_t d = 0; d < query_width_; ++d) dst[d] *= scale;
}

bool HistoricalKvMeanKeySource::publish_ring(Ring & ring, uint64_t end, bool block_aligned, bool require_full) {
    if (ring.accepted.empty() || !ring.accepted[0]) return false;
    const uint64_t accepted = ring.accepted[0];
    const uint32_t count = ring.count[0];
    for (size_t layer = 0; layer < layers_.size(); ++layer)
        if (ring.end[layer] != end || ring.accepted[layer] != accepted || ring.count[layer] != count)
            return false;
    if (block_aligned && (accepted < ring.window ||
            (ring.published_accepted && accepted / ring.window <= ring.published_accepted / ring.window)))
        return false;
    if (!block_aligned && require_full && count < ring.window) return false;
    for (size_t layer = 0; layer < layers_.size(); ++layer)
        ring_mean(ring, layer, ring.mean.data() + layer * query_width_);
    if (!std::all_of(ring.mean.begin(), ring.mean.end(), [](float v) { return std::isfinite(v); }))
        return false;
    ring.valid = true;
    ring.published_accepted = accepted;
    return true;
}

void HistoricalKvMeanKeySource::publish_query(uint32_t layer, uint64_t end,
                                              const float * rows, uint32_t count) {
    if (!uses_layer(layer) || !rows || !count) return;
    // Only prefill uses this path. The worker takes this mutex just long
    // enough to snapshot a shared_ptr; it never scores under the mutex.
    // Dropping a contended input batch would violate full-question coverage.
    std::lock_guard<std::mutex> lock(query_mutex_);
    if (!question_collecting_) {
        if (question_invalid_) return;
        reset_question_locked();
    }
    if (question_invalid_) return;
    if (count > end) { question_invalid_ = true; return; }
    const size_t index = size_t(std::find(layers_.begin(), layers_.end(), layer) - layers_.begin());
    auto & segments = question_building_->layers[index];
    const uint64_t begin = end - count;
    if (!segments.empty() && begin < segments.back().end) {
        question_invalid_ = true; // replay/overlap is not a second contribution
        return;
    }
    const size_t bytes = product(query_width_, sizeof(float)) + sizeof(QuestionSegment);
    for (uint32_t t = 0; t < count; ++t) {
        const uint64_t position = begin + t;
        if (segments.empty() || segments.back().count == question_segment_tokens_ ||
                segments.back().end != position) {
            if (bytes > question_byte_budget_ - question_building_->bytes) {
                question_invalid_ = true;
                return;
            }
            segments.push_back({position, position, 0, std::vector<float>(query_width_, 0.f)});
            question_building_->bytes += bytes;
        }
        auto & segment = segments.back();
        const float * src = rows + size_t(t) * query_width_;
        for (size_t d = 0; d < query_width_; ++d) {
            if (!std::isfinite(src[d])) { question_invalid_ = true; return; }
            segment.mean[d] += src[d];
        }
        ++segment.count;
        segment.end = position + 1;
    }
}

bool HistoricalKvMeanKeySource::finish_question(uint64_t expected_tokens) {
    std::lock_guard<std::mutex> lock(query_mutex_);
    if (!question_collecting_) {
        if (!question_ || question_invalid_) return false;
        if (expected_tokens == UINT64_MAX || expected_tokens == question_tokens_.load()) return true;
        // Repeated finalization, including an exact-prefix reuse, must still
        // honor the caller's coverage contract and invalidate delayed intents.
        question_invalid_ = true;
        question_.reset();
        question_tokens_ = question_segments_ = question_bytes_ = 0;
        ++question_epoch_;
        ++invalid_questions_;
        return false;
    }
    question_collecting_ = false;
    const auto & first = question_building_->layers.front();
    if (first.empty()) question_invalid_ = true;
    uint64_t tokens = 0;
    for (const auto & segment : first) tokens += segment.count;
    if (expected_tokens != UINT64_MAX && tokens != expected_tokens) question_invalid_ = true;
    for (auto & layer : question_building_->layers) {
        if (layer.size() != first.size()) { question_invalid_ = true; break; }
        for (size_t s = 0; s < layer.size(); ++s) {
            auto & segment = layer[s];
            if (segment.begin != first[s].begin || segment.end != first[s].end ||
                    segment.count != first[s].count || !segment.count) {
                question_invalid_ = true;
                break;
            }
            for (float & v : segment.mean) {
                v /= float(segment.count);
                if (!std::isfinite(v)) question_invalid_ = true;
            }
        }
    }
    if (question_invalid_) {
        ++invalid_questions_;
        question_.reset();
        question_building_.reset();
        return false;
    }
    question_tokens_ = tokens;
    question_segments_ = first.size();
    question_bytes_ = question_building_->bytes;
    question_ = std::move(question_building_);
    ++samples_;
    if (retire_observation_) {
        reset_observations();
        retire_observation_ = false;
    }
    return true;
}

bool HistoricalKvMeanKeySource::reuse_question(uint64_t expected_tokens) {
    std::lock_guard<std::mutex> lock(query_mutex_);
    if (!question_ || !question_collecting_ || question_invalid_ ||
        std::any_of(question_building_->layers.begin(), question_building_->layers.end(),
            [](const auto & layer) { return !layer.empty(); })) return false;
    uint64_t tokens = 0;
    for (const auto & segment : question_->layers.front()) tokens += segment.count;
    if (!tokens || tokens != expected_tokens) return false;
    question_tokens_ = tokens;
    question_segments_ = question_->layers.front().size();
    question_bytes_ = question_->bytes;
    question_building_.reset();
    question_collecting_ = retire_observation_ = false;
    ++samples_;
    return true;
}

void HistoricalKvMeanKeySource::publish_observation(uint32_t layer, uint64_t end,
                                                   const float * rows, uint32_t count) {
    if (!uses_layer(layer) || !rows || !count) return;
    std::unique_lock<std::mutex> lock(query_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) { ++skipped_; return; }
    if (!append_ring(observation_, layer, end, rows, count)) return;
    if (!append_ring(observation_fast_, layer, end, rows, count)) return;
    if (publish_ring(observation_, end, true)) {
        remember_observation();
        ++observation_samples_;
    }
    if (publish_ring(observation_fast_, end, false, true)) ++fast_samples_;
}

void HistoricalKvMeanKeySource::flush_observation() {
    std::lock_guard<std::mutex> lock(query_mutex_);
    if (observation_.accepted.empty() || !observation_.accepted[0]) return;
    if (observation_.valid && observation_.published_accepted == observation_.accepted[0]) return;
    if (publish_ring(observation_, observation_.end[0], false)) {
        remember_observation();
        ++observation_samples_;
    }
    if (observation_fast_.accepted.empty() || !observation_fast_.accepted[0]) return;
    if (publish_ring(observation_fast_, observation_fast_.end[0], false, true)) ++fast_samples_;
}

void HistoricalKvMeanKeySource::begin_question() {
    std::lock_guard<std::mutex> lock(query_mutex_);
    reset_question_locked();
    retire_observation_ = true;
}

bool HistoricalKvMeanKeySource::begin_round(uint64_t incarnation) {
    if (incarnation != repository_->incarnation()) return false;
    std::unique_lock<std::mutex> lock(query_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || question_collecting_ || question_invalid_ ||
        (!question_ && !observation_keep_count_ && !observation_fast_.valid)) return false;
    question_round_ = question_;
    fast_round_valid_ = observation_fast_.valid;
    observation_keep_round_count_ = observation_keep_count_;
    if (fast_round_valid_)
        std::copy(observation_fast_.mean.begin(), observation_fast_.mean.end(), fast_round_.begin());
    const size_t width = observation_.mean.size();
    if (observation_keep_round_.size() != kObservationKeep * width)
        observation_keep_round_.assign(kObservationKeep * width, 0.f);
    if (observation_keep_count_ && width && observation_keep_.size() == kObservationKeep * width) {
        const size_t start = (observation_keep_head_ + kObservationKeep - observation_keep_count_) % kObservationKeep;
        for (size_t i = 0; i < observation_keep_count_; ++i) {
            const size_t src = (start + i) % kObservationKeep;
            std::copy(observation_keep_.begin() + src * width, observation_keep_.begin() + (src + 1) * width,
                observation_keep_round_.begin() + i * width);
        }
    }
    return true;
}

double HistoricalKvMeanKeySource::score_query(const float * key, const float * query) const {
    double logit = 0;
    for (size_t l = 0; l < layers_.size(); ++l)
        logit += score_query_layer(key + l * kv_heads_ * head_dimension_, query + l * query_width_);
    return logit / double(layers_.size());
}

double HistoricalKvMeanKeySource::score_question_segment(const float * key, size_t segment) const {
    double logit = 0;
    for (size_t l = 0; l < layers_.size(); ++l)
        logit += score_query_layer(key + l * kv_heads_ * head_dimension_,
            question_round_->layers[l][segment].mean.data());
    return logit / double(layers_.size());
}

double HistoricalKvMeanKeySource::score_query_layer(const float * key, const float * query) const {
    const size_t groups = kv_heads_, gqa = query_heads_ / kv_heads_;
    double logit = 0;
    for (size_t h = 0; h < groups; ++h) {
        const auto * head = key + h * head_dimension_;
        double best = -std::numeric_limits<double>::infinity();
        for (size_t q = 0; q < gqa; ++q) {
            const auto * row = query + (h * gqa + q) * head_dimension_;
            double dot = 0;
            for (size_t d = 0; d < head_dimension_; ++d) dot += double(head[d]) * row[d];
            best = std::max(best, dot);
        }
        logit += best / (double(groups) * std::sqrt(double(head_dimension_)));
    }
    return logit;
}

bool HistoricalKvMeanKeySource::read(const HistoricalKvBlockMeta & meta,
                                     float * descriptor, double & logit) {
    // Convenience for synchronous callers. The GC policy uses read_bounded.
    ReadResult result;
    do { result = read_bounded(meta, descriptor, logit); } while (result == ReadResult::Yielded);
    return result == ReadResult::Ready;
}

HistoricalKvRedundancySource::ReadResult HistoricalKvMeanKeySource::read_bounded(
        const HistoricalKvBlockMeta & meta, float * descriptor, double & logit) {
    if (meta.handle.slot >= cached_.size() || meta.handle.incarnation != repository_->incarnation()) return ReadResult::Missing;
    auto & entry = cached_[meta.handle.slot];
    auto * key = keys_.data() + meta.handle.slot * dimensions_;
    if (!same(entry.handle, meta.handle) || entry.version != meta.content_version || !entry.valid) {
        auto lease = repository_->acquire_background_mean_k(meta.handle);
        if (!lease) return ReadResult::Missing;
        const auto current = lease->metadata();
        if (!current || !current->sealed || current->content_version != meta.content_version) return ReadResult::Missing;
        const size_t width = size_t(kv_heads_) * head_dimension_;
        entry.valid = false;
        for (size_t l = 0; l < layers_.size(); ++l)
            if (!storage_->copy_mean_k(*lease, meta.handle, layers_[l], key + l * width, width)) return ReadResult::Missing;
        entry = {meta.handle, meta.content_version, true};
        question_scores_[meta.handle.slot] = {};
    }
    logit = -std::numeric_limits<double>::infinity();
    if (question_round_) {
        auto & score = question_scores_[meta.handle.slot];
        if (score.epoch != question_round_->epoch) {
            score = {};
            score.epoch = question_round_->epoch;
        }
        const size_t total = question_round_->layers.front().size();
        const size_t end = score.next + std::min(kQuestionSegmentsPerRead, total - score.next);
        for (; score.next < end; ++score.next) {
            score.peak = std::max(score.peak, score_question_segment(key, score.next));
            ++question_score_evaluations_;
        }
        if (score.next != total) {
            ++question_score_yields_;
            return ReadResult::Yielded;
        }
        logit = score.peak;
    }
    const size_t groups = layers_.size() * kv_heads_;
    const double scale = 1.0 / std::sqrt(double(groups));
    for (size_t h = 0; h < groups; ++h) {
        const auto * k = key + h * head_dimension_;
        double norm = 0;
        for (size_t d = 0; d < head_dimension_; ++d) norm += double(k[d]) * k[d];
        if (!std::isfinite(norm) || norm <= 1e-16) return ReadResult::Missing;
        for (size_t d = 0; d < head_dimension_; ++d)
            descriptor[h * head_dimension_ + d] = float(k[d] / std::sqrt(norm) * scale);
    }
    const size_t width = layers_.size() * query_width_;
    for (size_t i = 0; i < observation_keep_round_count_ && width; ++i)
        logit = std::max(logit, score_query(key, observation_keep_round_.data() + i * width));
    if (fast_round_valid_) logit = std::max(logit, score_query(key, fast_round_.data()));
    return std::isfinite(logit) ? ReadResult::Ready : ReadResult::Missing;
}

HistoricalKvRedundancyPolicy::HistoricalKvRedundancyPolicy(size_t capacity,
    std::shared_ptr<HistoricalKvRedundancySource> source, Config config)
    : capacity_(capacity), dimensions_(source ? source->dimensions() : 0),
      source_(std::move(source)), config_(config) {
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
        source_->question_epoch() == question_epoch_;
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
    if (thinking_mode_ && !thinking_block(meta)) return false;
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
        thinking_mode_ = config_.thinking && config_.thinking->armed();
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
        const auto & meta = *rows_[group[i]].meta;
        out.push({meta.handle, snapshot.round_id, reason, meta.content_version});
    }
    return n;
}

HistoricalKvPolicy::Metrics HistoricalKvRedundancyPolicy::metrics() const noexcept {
    return {feature_blocks_.load(), missing_.load(), comparisons_.load(), scored_.load(),
            representatives_.load(), no_query_.load(), lossy_.load(), max_round_features_.load()};
}
} // namespace kvmem

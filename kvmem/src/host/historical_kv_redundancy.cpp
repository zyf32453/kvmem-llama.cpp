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
    keys_.resize(product(cached_.size(), dimensions_));
    staged_query_.resize(product(layers_.size(), query_width_));
    published_query_.resize(staged_query_.size());
    round_query_.resize(staged_query_.size());
    query_ends_.resize(layers_.size(), UINT64_MAX);
}

bool HistoricalKvMeanKeySource::uses_layer(uint32_t layer) const noexcept {
    return std::find(layers_.begin(), layers_.end(), layer) != layers_.end();
}

void HistoricalKvMeanKeySource::publish_query(uint32_t layer, uint64_t end,
                                              const float * rows, uint32_t count) {
    const auto it = std::find(layers_.begin(), layers_.end(), layer);
    if (it == layers_.end() || !rows || !count) return;
    std::unique_lock<std::mutex> lock(query_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) { ++skipped_; return; }
    const size_t index = size_t(it - layers_.begin()), take = std::min(count, 8u);
    auto * dst = staged_query_.data() + index * query_width_;
    std::fill(dst, dst + query_width_, 0.f);
    for (size_t t = count - take; t < count; ++t)
        for (size_t d = 0; d < query_width_; ++d) dst[d] += rows[t * query_width_ + d] / float(take);
    query_ends_[index] = end;
    if (std::all_of(query_ends_.begin(), query_ends_.end(), [end](uint64_t v) { return v == end; }) &&
        std::all_of(staged_query_.begin(), staged_query_.end(), [](float v) { return std::isfinite(v); })) {
        std::copy(staged_query_.begin(), staged_query_.end(), published_query_.begin());
        query_valid_ = true;
        ++samples_;
    }
}

bool HistoricalKvMeanKeySource::begin_round(uint64_t incarnation) {
    if (incarnation != repository_->incarnation()) return false;
    std::unique_lock<std::mutex> lock(query_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || !query_valid_) return false;
    std::copy(published_query_.begin(), published_query_.end(), round_query_.begin());
    return true;
}

bool HistoricalKvMeanKeySource::read(const HistoricalKvBlockMeta & meta,
                                     float * descriptor, double & logit) {
    if (meta.handle.slot >= cached_.size() || meta.handle.incarnation != repository_->incarnation()) return false;
    auto & entry = cached_[meta.handle.slot];
    auto * key = keys_.data() + meta.handle.slot * dimensions_;
    if (!same(entry.handle, meta.handle) || entry.version != meta.content_version || !entry.valid) {
        auto lease = repository_->acquire_background_mean_k(meta.handle);
        if (!lease) return false;
        const auto current = lease->metadata();
        if (!current || !current->sealed || current->content_version != meta.content_version) return false;
        const size_t width = size_t(kv_heads_) * head_dimension_;
        entry.valid = false;
        for (size_t l = 0; l < layers_.size(); ++l)
            if (!storage_->copy_mean_k(*lease, meta.handle, layers_[l], key + l * width, width)) return false;
        entry = {meta.handle, meta.content_version, true};
    }
    const size_t groups = layers_.size() * kv_heads_, gqa = query_heads_ / kv_heads_;
    const double scale = 1.0 / std::sqrt(double(groups));
    logit = 0;
    for (size_t h = 0; h < groups; ++h) {
        const auto * k = key + h * head_dimension_;
        double norm = 0, best = -std::numeric_limits<double>::infinity();
        for (size_t d = 0; d < head_dimension_; ++d) norm += double(k[d]) * k[d];
        if (!std::isfinite(norm) || norm <= 1e-16) return false;
        for (size_t d = 0; d < head_dimension_; ++d)
            descriptor[h * head_dimension_ + d] = float(k[d] / std::sqrt(norm) * scale);
        for (size_t q = 0; q < gqa; ++q) {
            const auto * query = round_query_.data() + (h * gqa + q) * head_dimension_;
            double dot = 0;
            for (size_t d = 0; d < head_dimension_; ++d) dot += double(k[d]) * query[d];
            best = std::max(best, dot);
        }
        logit += best / (double(groups) * std::sqrt(double(head_dimension_)));
    }
    return std::isfinite(logit);
}

HistoricalKvRedundancyPolicy::HistoricalKvRedundancyPolicy(size_t capacity,
    std::shared_ptr<HistoricalKvRedundancySource> source, Config config)
    : capacity_(capacity), dimensions_(source ? source->dimensions() : 0),
      source_(std::move(source)), config_(config) {
    if (!source_ || !capacity_ || !dimensions_ || !config_.reference_blocks ||
        !std::isfinite(config_.importance_weight) || config_.importance_weight < 0 || config_.importance_weight > 1 ||
        !std::isfinite(config_.representative_similarity) || config_.representative_similarity < 0 ||
        config_.representative_similarity > 1)
        throw std::invalid_argument("invalid historical redundancy configuration");
    rows_.reserve(capacity_); references_.reserve(capacity_); ranked_.reserve(capacity_);
    descriptors_.resize(product(capacity_, dimensions_));
}

bool HistoricalKvRedundancyPolicy::permits(const HistoricalKvBlockMeta & meta) const noexcept {
    return meta.life == HistoricalKvLife::Live && meta.sealed && !meta.mandatory;
}

HistoricalKvStepResult HistoricalKvRedundancyPolicy::step(const HistoricalKvSnapshot & snapshot,
    const HistoricalKvPressure & pressure, HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer & out) {
    if (snapshot.blocks.size() > capacity_ || !budget.scan || !budget.intents ||
        !(pressure.low < pressure.high) || pressure.high > capacity_)
        throw std::invalid_argument("invalid historical redundancy step");
    if (!initialized_ || incarnation_ != snapshot.incarnation || round_ != snapshot.round_id) {
        if (!initialized_ || incarnation_ != snapshot.incarnation) pressure_mode_ = false;
        initialized_ = true; incarnation_ = snapshot.incarnation; round_ = snapshot.round_id;
        cursor_ = pair_row_ = pair_reference_ = 0;
        done_ = collected_ = false;
        rows_.clear(); references_.clear(); ranked_.clear();
    }
    if (done_) return {HistoricalKvStepStatus::Complete, 0, 0};
    if (pressure.ready < pressure.low) pressure_mode_ = true;
    if (pressure.ready >= pressure.high) pressure_mode_ = false;
    const uint64_t target = std::max(pressure_mode_ ? pressure.high : 0u, pressure.requested_ready);
    const uint64_t supply = uint64_t(pressure.ready) + pressure.pending_slot_credit;
    if (target <= supply) { done_ = true; return {HistoricalKvStepStatus::Complete, 0, 0}; }
    if (!out.remaining()) return {HistoricalKvStepStatus::Yielded, 0, 0};
    if (!cursor_ && !collected_ && !source_->begin_round(snapshot.incarnation)) {
        ++no_query_; done_ = true;
        return {HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
    size_t examined = 0;
    while (!collected_ && cursor_ < snapshot.blocks.size() && examined < budget.scan) {
        const auto & meta = snapshot.blocks[cursor_++]; ++examined;
        if (meta.life != HistoricalKvLife::Live || !meta.sealed || meta.handle.incarnation != incarnation_) continue;
        double logit = 0;
        float * descriptor = descriptors_.data() + rows_.size() * dimensions_;
        if (!source_->read(meta, descriptor, logit) || !std::isfinite(logit)) { ++missing_; continue; }
        double norm = 0;
        for (size_t d = 0; d < dimensions_; ++d) norm += double(descriptor[d]) * descriptor[d];
        if (!std::isfinite(norm) || norm < 1e-16) { ++missing_; continue; }
        for (size_t d = 0; d < dimensions_; ++d) descriptor[d] /= float(std::sqrt(norm));
        rows_.push_back({&meta, logit}); ++feature_blocks_;
    }
    if (!collected_) {
        if (cursor_ < snapshot.blocks.size()) return {HistoricalKvStepStatus::Yielded, examined, 0};
        collected_ = true;
        if (rows_.empty()) { done_ = true; return {HistoricalKvStepStatus::NoCandidate, examined, 0}; }
        references_.resize(rows_.size()); std::iota(references_.begin(), references_.end(), 0);
        std::sort(references_.begin(), references_.end(), [&](size_t a, size_t b) {
            return age(*rows_[a].meta) > age(*rows_[b].meta);
        });
        references_.resize(std::min(references_.size(), config_.reference_blocks));
        return {HistoricalKvStepStatus::Yielded, examined, 0};
    }
    // No N*N matrix: bounded recent reference set and resumable pair work.
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
            row.redundancy += cosine; ++row.compared; ++comparisons_;
            if (cosine >= config_.representative_similarity && age(*rows_[other].meta) > age(*row.meta) &&
                (row.representative == SIZE_MAX || age(*rows_[other].meta) > age(*rows_[row.representative].meta)))
                row.representative = other;
        }
        if (pair_reference_ == references_.size()) { ++pair_row_; pair_reference_ = 0; }
    }
    if (pair_row_ < rows_.size()) return {HistoricalKvStepStatus::Yielded, pairs, 0};
    double max_importance = -std::numeric_limits<double>::infinity(), max_redundancy = -std::numeric_limits<double>::infinity();
    for (auto & row : rows_) {
        row.redundancy /= double(std::max<size_t>(row.compared, 1));
        max_importance = std::max(max_importance, row.importance);
        max_redundancy = std::max(max_redundancy, row.redundancy);
        if (row.representative != SIZE_MAX) rows_[row.representative].retained = true;
    }
    double importance_sum = 0, redundancy_sum = 0;
    for (auto & row : rows_) {
        importance_sum += row.importance = std::exp(row.importance - max_importance);
        redundancy_sum += row.redundancy = std::exp(row.redundancy - max_redundancy);
    }
    for (size_t i = 0; i < rows_.size(); ++i) {
        auto & row = rows_[i];
        row.score = config_.importance_weight * row.importance / importance_sum -
                    (1.0 - config_.importance_weight) * row.redundancy / redundancy_sum;
        if (row.retained) ++representatives_;
        if (!row.retained && permits(*row.meta)) ranked_.push_back(i);
    }
    ++scored_;
    const size_t n = std::min({size_t(target - supply), budget.intents, out.remaining(), ranked_.size()});
    std::partial_sort(ranked_.begin(), ranked_.begin() + n, ranked_.end(), [&](size_t a, size_t b) {
        const auto & x = rows_[a]; const auto & y = rows_[b];
        return std::make_tuple(x.score, x.meta->last_use_seq, age(*x.meta)) <
               std::make_tuple(y.score, y.meta->last_use_seq, age(*y.meta));
    });
    for (size_t i = 0; i < n; ++i) {
        const auto & meta = *rows_[ranked_[i]].meta;
        out.push({meta.handle, snapshot.round_id, HistoricalKvDeleteReason::Redundancy, meta.content_version});
    }
    done_ = true;
    return {n ? HistoricalKvStepStatus::Complete : HistoricalKvStepStatus::NoCandidate, pairs, n};
}

HistoricalKvPolicy::Metrics HistoricalKvRedundancyPolicy::metrics() const noexcept {
    return {feature_blocks_.load(), missing_.load(), comparisons_.load(), scored_.load(),
            representatives_.load(), no_query_.load()};
}
} // namespace kvmem

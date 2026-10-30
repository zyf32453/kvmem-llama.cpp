#include "kvmem/historical_kv_page_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvmem {

HistoricalKvPageFeatures::HistoricalKvPageFeatures(uint32_t logical_pages, size_t dimensions,
    std::shared_ptr<HistoricalKvAuxBudget> auxiliary)
    : dimensions_(dimensions),
      pages_(HistoricalKvAuxAllocator<HistoricalKvAuxVector<float>>(auxiliary, HistoricalKvAuxKind::Features)),
      question_scores_(HistoricalKvAuxAllocator<double>(auxiliary, HistoricalKvAuxKind::Features)),
      output_peaks_(question_scores_.get_allocator()), frozen_importance_(question_scores_.get_allocator()) {
    if (!logical_pages || !dimensions) throw std::invalid_argument("historical page feature geometry");
    pages_.reserve(logical_pages);
    for (uint32_t page = 0; page < logical_pages; ++page)
        pages_.emplace_back(HistoricalKvAuxAllocator<float>(auxiliary, HistoricalKvAuxKind::Features));
    output_peaks_.assign(logical_pages, -std::numeric_limits<double>::infinity());
}

void HistoricalKvPageFeatures::advance_question_epoch() noexcept {
    // Called under mu_. Saturation revokes typed certificates permanently.
    const auto current = question_epoch_.load(std::memory_order_relaxed);
    if (current != UINT64_MAX) question_epoch_.store(current + 1, std::memory_order_release);
}

bool HistoricalKvPageFeatures::has_page(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mu_);
    return page < pages_.size() && pages_[page].size() == dimensions_;
}

void HistoricalKvPageFeatures::publish_page(uint32_t page, const std::vector<float> & descriptor) {
    if (page >= pages_.size() || descriptor.size() != dimensions_)
        throw std::invalid_argument("historical page descriptor dimensions");
    for (float value : descriptor)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite historical page descriptor");
    std::lock_guard<std::mutex> lock(mu_);
    if (!pages_[page].empty()) throw std::logic_error("historical page descriptor republished");
    pages_[page].assign(descriptor.begin(), descriptor.end());
}

void HistoricalKvPageFeatures::publish_question_scores(const std::vector<double> & scores, bool new_question) {
    if (scores.size() > pages_.size()) throw std::invalid_argument("historical Query score geometry");
    for (double value : scores)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite historical Query score");
    std::lock_guard<std::mutex> lock(mu_);
    question_scores_.assign(scores.begin(), scores.end());
    if (new_question) {
        std::fill(output_peaks_.begin(), output_peaks_.end(), -std::numeric_limits<double>::infinity());
        advance_question_epoch();
    }
}

void HistoricalKvPageFeatures::publish_output_attention(uint32_t page, double mass) {
    if (page >= pages_.size() || !std::isfinite(mass) || mass < 0)
        throw std::invalid_argument("invalid historical output attention mass");
    std::lock_guard<std::mutex> lock(mu_);
    output_peaks_[page] = std::max(output_peaks_[page], mass);
}

void HistoricalKvPageFeatures::clear_question() {
    std::lock_guard<std::mutex> lock(mu_);
    question_scores_.clear();
    std::fill(output_peaks_.begin(), output_peaks_.end(), -std::numeric_limits<double>::infinity());
    advance_question_epoch();
}

void HistoricalKvPageFeatures::truncate_to(uint64_t token_frontier, uint32_t page_tokens) {
    if (!page_tokens) throw std::invalid_argument("historical page token width");
    const auto page = token_frontier / page_tokens;
    std::lock_guard<std::mutex> lock(mu_);
    if (page >= pages_.size()) return;
    for (size_t id = page; id < pages_.size(); ++id) {
        pages_[id].clear();
        output_peaks_[id] = -std::numeric_limits<double>::infinity();
    }
    if (question_scores_.size() > page) question_scores_.resize(page);
    advance_question_epoch();
}

bool HistoricalKvPageFeatures::begin_round(uint64_t) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto current = score_epoch_.load(std::memory_order_relaxed);
    if (current == UINT64_MAX) return false;
    frozen_importance_ = output_peaks_;
    for (size_t id = 0; id < std::min(question_scores_.size(), frozen_importance_.size()); ++id)
        frozen_importance_[id] = std::max(frozen_importance_[id], question_scores_[id]);
    score_epoch_.store(current + 1, std::memory_order_release);
    return true;
}

bool HistoricalKvPageFeatures::read(const HistoricalKvBlockMeta & meta,
                                   float * descriptor, double & importance) {
    if (!descriptor || meta.logical_id >= pages_.size()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const auto page = static_cast<size_t>(meta.logical_id);
    if (pages_[page].size() != dimensions_) return false;
    std::copy(pages_[page].begin(), pages_[page].end(), descriptor);
    importance = std::isfinite(frozen_importance_[page]) ? frozen_importance_[page] : 0.0;
    return true;
}

} // namespace kvmem

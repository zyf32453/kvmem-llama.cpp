#pragma once

// Small, deterministic helpers shared by the production retrieval scorer and
// its host-only tests. Keep the projection arithmetic in sync with the original
// score_retrieval() implementation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace kvmem {

inline constexpr uint32_t kRedundancyDescriptorDims = 64;

enum class RedundancyDescriptorStatus {
    Valid,
    ZeroNorm,
    NonFinite,
};

inline void redundancy_accumulate_mean_k(const float * mean_k,
                                         uint32_t n_embd_k,
                                         uint32_t layer,
                                         float * descriptor) {
    for (uint32_t d = 0; d < n_embd_k; ++d) {
        const uint32_t mixed = d * 2654435761u + layer * 1013904223u;
        const uint32_t bin = mixed % kRedundancyDescriptorDims;
        const float sign = ((d * 2246822519u + layer * 3266489917u) & 0x80000000u)
            ? 1.0f : -1.0f;
        descriptor[bin] += sign * mean_k[d];
    }
}

inline RedundancyDescriptorStatus redundancy_normalize_descriptor(
        float * descriptor, uint32_t dimensions = kRedundancyDescriptorDims) {
    float norm = 0.0f;
    for (uint32_t d = 0; d < dimensions; ++d) {
        if (!std::isfinite(descriptor[d])) return RedundancyDescriptorStatus::NonFinite;
        norm += descriptor[d] * descriptor[d];
    }
    if (!std::isfinite(norm)) return RedundancyDescriptorStatus::NonFinite;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm)) return RedundancyDescriptorStatus::NonFinite;
    if (norm <= 1e-8f) return RedundancyDescriptorStatus::ZeroNorm;
    for (uint32_t d = 0; d < dimensions; ++d) {
        descriptor[d] /= norm;
        if (!std::isfinite(descriptor[d])) return RedundancyDescriptorStatus::NonFinite;
    }
    return RedundancyDescriptorStatus::Valid;
}

inline double redundancy_cosine_similarity(const float * a,
                                           const float * b,
                                           uint32_t dimensions = kRedundancyDescriptorDims) {
    double dot = 0.0;
    for (uint32_t d = 0; d < dimensions; ++d) {
        if (!std::isfinite(a[d]) || !std::isfinite(b[d])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        dot += a[d] * b[d];
    }
    if (!std::isfinite(dot)) return std::numeric_limits<double>::quiet_NaN();
    return std::clamp(dot, -1.0, 1.0);
}

// Later partners used by score_retrieval(): the next 64 blocks, plus the
// newest 64 blocks when they fall outside that local window. Not all-pairs.
template <typename Visitor>
inline void redundancy_visit_partners(uint32_t block, uint32_t block_count, Visitor && visit) {
    const uint32_t local_end = std::min<uint32_t>(block_count, block + 65);
    for (uint32_t partner = block + 1; partner < local_end; ++partner) {
        visit(partner);
    }
    const uint32_t recent_begin = block_count > 64 ? block_count - 64 : block + 1;
    for (uint32_t partner = std::max(block + 1, recent_begin); partner < block_count; ++partner) {
        if (partner >= local_end) {
            visit(partner);
        }
    }
}

// Return false when no finite similarity was computed. The caller must keep
// the unevaluated sentinel in that case; -1 is reserved for a real cosine of
// opposite descriptors.
template <typename IsComparable, typename Similarity>
inline bool redundancy_max_later_similarity(uint32_t block,
                                            uint32_t block_count,
                                            const IsComparable & is_comparable,
                                            const Similarity & similarity,
                                            double & max_similarity,
                                            uint32_t * comparison_count = nullptr) {
    double best = -1.0;
    uint32_t count = 0;
    redundancy_visit_partners(block, block_count, [&](uint32_t partner) {
        if (!is_comparable(partner)) return;
        const double value = similarity(partner);
        if (!std::isfinite(value)) return;
        best = std::max(best, value);
        ++count;
    });
    if (comparison_count) *comparison_count = count;
    if (count == 0) return false;
    max_similarity = best;
    return true;
}

} // namespace kvmem

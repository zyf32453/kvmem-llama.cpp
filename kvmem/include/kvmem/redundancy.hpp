#pragma once

// Small, deterministic helpers shared by the production retrieval scorer and
// its host-only tests. Keep the projection arithmetic in sync with the original
// score_retrieval() implementation.

#include <algorithm>
#include <cmath>
#include <cstdint>

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
    // Same acceptance test as the original score_retrieval() loop:
    // `norm > 1e-8f` then divide. A non-finite norm fails that comparison and
    // is left unchanged. An infinite norm passes it, matching the previous
    // inlined arithmetic.
    float norm = 0.0f;
    bool non_finite = false;
    for (uint32_t d = 0; d < dimensions; ++d) {
        if (!std::isfinite(descriptor[d])) {
            non_finite = true;
        }
        norm += descriptor[d] * descriptor[d];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            descriptor[d] /= norm;
        }
        return RedundancyDescriptorStatus::Valid;
    }
    if (non_finite || !std::isfinite(norm)) {
        return RedundancyDescriptorStatus::NonFinite;
    }
    return RedundancyDescriptorStatus::ZeroNorm;
}

inline double redundancy_cosine_similarity(const float * a,
                                           const float * b,
                                           uint32_t dimensions = kRedundancyDescriptorDims) {
    double dot = 0.0;
    for (uint32_t d = 0; d < dimensions; ++d) {
        dot += a[d] * b[d];
    }
    return dot;
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

} // namespace kvmem

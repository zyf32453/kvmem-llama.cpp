#pragma once

#include "kvmem/historical_kv_policy.hpp"
#include <cstdlib>

namespace kvmem {

// Diagnostic only. Sample at construction, before worker threads start;
// changing the environment of a running manager is not supported.
inline bool historical_kv_audit_enabled() noexcept {
    const char * value = std::getenv("KVMEM_GC_AUDIT");
    return value && value[0] == '1' && value[1] == '\0';
}

inline const char * historical_kv_audit_reason(HistoricalKvDeleteReason reason) noexcept {
    switch (reason) {
    case HistoricalKvDeleteReason::Lru: return "lru";
    case HistoricalKvDeleteReason::Custom: return "custom";
    case HistoricalKvDeleteReason::Redundancy: return "redundancy";
    }
    return "unknown";
}

inline const char * historical_kv_audit_outcome(HistoricalKvCommitCode code) noexcept {
    switch (code) {
    case HistoricalKvCommitCode::Committed: return "committed";
    case HistoricalKvCommitCode::AlreadyRetired: return "already_retired";
    case HistoricalKvCommitCode::Stale: return "stale";
    case HistoricalKvCommitCode::Reevaluate: return "reevaluate";
    case HistoricalKvCommitCode::Retry: return "retry";
    }
    return "unknown";
}

} // namespace kvmem

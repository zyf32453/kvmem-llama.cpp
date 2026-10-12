#pragma once

// Optional, low-volume Redundancy diagnostics. Set KVMEM_REDUNDANCY_DIAG=1
// to enable; KVMEM_REDUNDANCY_DIAG_FILE=<path> also appends the same records
// to a file so a tmux/terminal transcript is not the only copy.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kvmem {

inline bool redundancy_diagnostics_enabled() {
    const char * value = std::getenv("KVMEM_REDUNDANCY_DIAG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

inline void redundancy_diag(const char * format, ...) {
    if (!redundancy_diagnostics_enabled()) {
        return;
    }

    char line[16384];
    va_list args;
    va_start(args, format);
    const int length = std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (length < 0) {
        return;
    }
    line[sizeof(line) - 1] = '\0';
    std::fputs(line, stderr);

    const char * path = std::getenv("KVMEM_REDUNDANCY_DIAG_FILE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::FILE * file = std::fopen(path, "a");
    if (file == nullptr) {
        std::fprintf(stderr, "KVMEM_REDUNDANCY_DIAG_FILE_ERROR open_failed=1\n");
        return;
    }
    std::fputs(line, file);
    std::fflush(file);
    std::fclose(file);
}

} // namespace kvmem

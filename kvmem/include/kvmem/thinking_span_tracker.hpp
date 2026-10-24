#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace kvmem {

// Token-id state machine for one request's thinking regions. The start tag is
// not content. A partial end tag is withheld so the published end cannot
// include it; a failed match then extends the span over that rejected prefix.
// A later region uses a new begin. The server publishes ready() spans.
class ThinkingSpanTracker {
public:
    using Token = int32_t;
    struct SpanUpdate {
        int begin = -1;
        int end = 0;
        bool closed = false;
    };

    ThinkingSpanTracker() = default;
    ThinkingSpanTracker(std::vector<Token> start, std::vector<std::vector<Token>> ends, bool enable)
        : start_(std::move(start)), ends_(std::move(ends)) {
        if (!enable || start_.empty() || ends_.empty()) return;
        enabled_ = true;
        end_progress_.assign(ends_.size(), 0);
    }

    void observe_prompt(const std::vector<Token> & tokens) {
        // Start a new request, including when a caller reuses this tracker.
        // A previous closed event must not suppress the new request's span.
        content_begin_ = -1;
        content_end_ = 0;
        open_ = closed_ = false;
        start_progress_ = 0;
        end_progress_.assign(ends_.size(), 0);
        has_published_ = false;
        if (!enabled_) return;
        int opened = -1;
        for (size_t i = 0; i + start_.size() <= tokens.size(); ++i)
            if (match_at(tokens, i, start_)) opened = int(i + start_.size());
        if (opened < 0) return;
        for (size_t i = size_t(opened); i < tokens.size(); ++i)
            for (const auto & end : ends_)
                if (match_at(tokens, i, end)) return;
        content_begin_ = opened;
        content_end_ = int(tokens.size());
        open_ = true;
        closed_ = false;
        end_progress_.assign(ends_.size(), 0);
    }

    void observe_generated(Token id, int position) {
        if (!enabled_ || position < 0) return;
        if (!open_) {
            if (start_progress_ < start_.size() && id == start_[start_progress_]) {
                if (++start_progress_ == start_.size()) {
                    open_ = true;
                    closed_ = false;
                    content_begin_ = position + 1;
                    content_end_ = content_begin_;
                    start_progress_ = 0;
                    end_progress_.assign(ends_.size(), 0);
                }
            } else start_progress_ = (!start_.empty() && id == start_[0]) ? 1 : 0;
            return;
        }
        bool partial = false;
        int partial_from = position + 1;
        for (size_t s = 0; s < ends_.size(); ++s) {
            const auto & end = ends_[s];
            if (end.empty()) continue;
            if (end_progress_[s] < end.size() && id == end[end_progress_[s]]) {
                if (++end_progress_[s] == end.size()) {
                    open_ = false;
                    closed_ = true;
                    content_end_ = position - int(end.size()) + 1;
                    if (content_end_ < content_begin_) content_end_ = content_begin_;
                    start_progress_ = 0;
                    return;
                }
                partial = true;
                partial_from = std::min(partial_from, position - int(end_progress_[s]) + 1);
            } else {
                end_progress_[s] = id == end[0] ? 1 : 0;
                if (end_progress_[s]) {
                    partial = true;
                    partial_from = std::min(partial_from, position);
                }
            }
        }
        content_end_ = partial ? partial_from : position + 1;
    }

    bool enabled() const noexcept { return enabled_; }
    bool open() const noexcept { return open_; }
    bool closed() const noexcept { return closed_; }
    int content_begin() const noexcept { return content_begin_; }
    int content_end() const noexcept { return content_end_; }
    // An open region with no content yet is not published. note() cannot shrink.
    bool ready() const noexcept {
        return enabled_ && content_begin_ >= 0 && (content_end_ > content_begin_ || closed_);
    }

    // Consume a state change once. In particular, the server calls this for
    // every answer token after </think>, but only the closing transition may
    // flush the generated observation tail. Open->closed at identical bounds
    // is a distinct event, and a later thinking region has a new begin.
    bool take_update(SpanUpdate & update) noexcept {
        if (!ready()) return false;
        if (has_published_ && published_.begin == content_begin_ &&
                published_.end == content_end_ && published_.closed == closed_) return false;
        update = {content_begin_, content_end_, closed_};
        published_ = update;
        has_published_ = true;
        return true;
    }

private:
    static bool match_at(const std::vector<Token> & tokens, size_t i, const std::vector<Token> & seq) {
        if (seq.empty() || i + seq.size() > tokens.size()) return false;
        for (size_t k = 0; k < seq.size(); ++k) if (tokens[i + k] != seq[k]) return false;
        return true;
    }

    std::vector<Token> start_;
    std::vector<std::vector<Token>> ends_;
    std::vector<size_t> end_progress_;
    size_t start_progress_ = 0;
    int content_begin_ = -1;
    int content_end_ = 0;
    bool open_ = false;
    bool closed_ = false;
    bool enabled_ = false;
    SpanUpdate published_;
    bool has_published_ = false;
};

} // namespace kvmem

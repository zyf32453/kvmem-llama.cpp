#include "kvmem/thinking_span_tracker.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using kvmem::ThinkingSpanTracker;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static ThinkingSpanTracker tracker() {
    return ThinkingSpanTracker({11, 12}, {{13, 14}}, true);
}

static void feed(ThinkingSpanTracker & t, std::initializer_list<ThinkingSpanTracker::Token> ids, int origin) {
    int position = origin;
    for (auto id : ids) t.observe_generated(id, position++);
}

static void marker_tests() {
    ThinkingSpanTracker off({11}, {{13}}, false);
    off.observe_prompt({11, 1});
    check(!off.ready() && !off.enabled(), "disabled tracker published a span");

    auto prompt_closed = tracker();
    prompt_closed.observe_prompt({11, 12, 1, 2, 13, 14, 9});
    check(!prompt_closed.open() && !prompt_closed.ready(), "closed prompt thinking stayed open");

    auto prompt_open = tracker();
    prompt_open.observe_prompt({1, 11, 12, 7, 8});
    check(prompt_open.open() && prompt_open.ready() && prompt_open.content_begin() == 3 &&
          prompt_open.content_end() == 5, "prompt thinking span excluded the start tag or the content");

    auto tag_at_end = tracker();
    tag_at_end.observe_prompt({11, 12});
    check(tag_at_end.open() && !tag_at_end.ready() && tag_at_end.content_begin() == 2,
          "start tag at the end of the prompt was published as content");
    tag_at_end.observe_generated(7, 2);
    check(tag_at_end.ready() && tag_at_end.content_begin() == 2 && tag_at_end.content_end() == 3,
          "first generated thinking token was left outside the span");

    auto generated = tracker();
    feed(generated, {11, 12}, 5);
    check(generated.open() && !generated.ready() && generated.content_begin() == 7,
          "generated start tag was included in the span");
    generated.observe_generated(7, 7);
    check(generated.ready() && generated.content_end() == 8, "generated thinking token did not extend the span");

    auto partial = tracker();
    partial.observe_prompt({11, 12});
    feed(partial, {7, 13}, 2);
    check(partial.open() && partial.content_end() == 3, "partial end tag was published as thinking content");
    partial.observe_generated(9, 4);
    check(partial.open() && partial.content_end() == 5, "rejected end-tag prefix stayed outside the span");

    auto closed = tracker();
    closed.observe_prompt({11, 12, 7});
    feed(closed, {13, 14}, 3);
    check(closed.closed() && !closed.open() && closed.content_begin() == 2 && closed.content_end() == 3,
          "completed end tag remained inside the thinking span");

    auto again = tracker();
    again.observe_prompt({11, 12, 7});
    feed(again, {13, 14, 11, 12, 8}, 3);
    check(again.open() && again.content_begin() == 7 && again.content_end() == 8,
          "a later thinking region reused the closed span");
}

static void publication_tests() {
    ThinkingSpanTracker::SpanUpdate update;
    auto t = tracker();
    t.observe_prompt({11, 12, 7});
    check(t.take_update(update) && update.begin == 2 && update.end == 3 && !update.closed,
          "initial open span was not published");
    check(!t.take_update(update), "unchanged open span was published twice");
    t.observe_generated(13, 3);
    check(!t.take_update(update), "partial end marker republished unchanged bounds");
    t.observe_generated(14, 4);
    check(t.take_update(update) && update.begin == 2 && update.end == 3 && update.closed,
          "open-to-closed transition with identical bounds was lost");
    check(!t.take_update(update), "closed event was published twice");
    for (int position = 5; position < 100; ++position) {
        t.observe_generated(7, position);
        check(!t.take_update(update), "ordinary answer token republished the closed event");
    }
    feed(t, {11, 12, 8}, 100);
    check(t.take_update(update) && update.begin == 102 && update.end == 103 && !update.closed,
          "second thinking region was suppressed by the first closed event");
    feed(t, {13, 14}, 103);
    check(t.take_update(update) && update.begin == 102 && update.end == 103 && update.closed &&
          !t.take_update(update), "second region did not close exactly once");

    // An empty region never emitted an open span, but still has one close.
    t.observe_prompt({11, 12});
    check(!t.take_update(update), "empty open region was published");
    feed(t, {13, 14}, 2);
    check(t.take_update(update) && update.begin == 2 && update.end == 2 && update.closed &&
          !t.take_update(update), "empty closed region was missing or duplicated");
    // observe_prompt is the request boundary even if all token positions repeat.
    t.observe_prompt({11, 12});
    feed(t, {13, 14}, 2);
    check(t.take_update(update) && update.begin == 2 && update.end == 2 && update.closed,
          "new request at identical positions inherited the publication cursor");
    t.observe_prompt({1, 2, 3});
    check(!t.ready() && !t.take_update(update), "new no-thinking prompt kept a stale closed span");
}

int main() {
    try {
        marker_tests();
        publication_tests();
        std::cout << "thinking span tracker: passed\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

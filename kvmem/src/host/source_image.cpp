#include "kvmem/source_image.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace kvmem {
namespace {
constexpr uint64_t magic = 0x31474d4953564bULL; // KVSIMG1, explicit little-endian framing
constexpr uint64_t version = 1, fixed_bytes = 144, event_bytes = 26;
void limits_valid(const SourceImageLimits & limits) {
    if (limits.max_bytes < fixed_bytes || !limits.max_tokens || !limits.max_events)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid source image limits");
}
bool configuration_present(const SourceConfiguration & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t x) { return x != 0; });
}
bool ref_present(SourceRef ref) {
    return ref.source.id && ref.source.incarnation && ref.revision;
}
bool metadata_valid(const SourceState & state) {
    const bool parent = ref_present(state.parent);
    const bool empty_parent = state.parent == SourceRef{};
    return ref_present(state.ref) && configuration_present(state.configuration) &&
        state.computed_frontier <= state.accepted_frontier &&
        ((empty_parent && !state.parent_prefix) || (parent &&
            !(state.parent.source == state.ref.source) && state.parent_prefix <= state.accepted_frontier));
}
uint64_t encoded_size(const SourceState & state) {
    if (state.accepted_frontier > (UINT64_MAX - fixed_bytes) / sizeof(int32_t))
        throw SnapshotCorrupt("source image token size overflow");
    const auto base = fixed_bytes + state.accepted_frontier * sizeof(int32_t);
    if (state.events > (UINT64_MAX - base) / event_bytes)
        throw SnapshotCorrupt("source image event size overflow");
    return base + state.events * event_bytes;
}
void put(SnapshotWriter & out, uint64_t value, size_t count = 8) {
    uint8_t bytes[8]{};
    for (size_t i = 0; i < count; ++i) bytes[i] = uint8_t(value >> (8*i));
    out.write(bytes, count);
}
uint64_t get(SnapshotReader & in, size_t count = 8) {
    uint8_t bytes[8]{}; in.read(bytes, count);
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value |= uint64_t(bytes[i]) << (8*i);
    return value;
}
void put_ref(SnapshotWriter & out, SourceRef ref) {
    put(out, ref.source.id); put(out, ref.source.incarnation); put(out, ref.revision);
}
SourceRef get_ref(SnapshotReader & in) { return {{get(in), get(in)}, get(in)}; }
} // namespace

uint64_t source_image_bytes(const SourceState & state, const SourceImageLimits & limits) {
    limits_valid(limits);
    if (!metadata_valid(state))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid source image metadata");
    const auto bytes = encoded_size(state);
    if (state.accepted_frontier > limits.max_tokens || state.events > limits.max_events || bytes > limits.max_bytes)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "source image exceeds transport limits");
    return bytes;
}

namespace {
template<class Rows, class Event, class Validate>
uint64_t encode_image(const SourceState & state, const SourceImageLimits & limits,
    SnapshotWriter::Sink sink, Rows read_rows, Event read_event, Validate validate) {
    if (!sink) throw MemoryError(MemoryErrorCode::InvalidPlan, "source image needs a transport sink");
    const auto bytes = source_image_bytes(state, limits);
    SnapshotWriter out(std::move(sink));
    put(out, magic); put(out, version); put(out, bytes);
    put_ref(out, state.ref); put_ref(out, state.parent);
    put(out, state.parent_prefix); put(out, state.accepted_frontier);
    put(out, state.computed_frontier); put(out, state.events);
    out.write(state.configuration.data(), state.configuration.size());
    std::array<int32_t, 256> rows{};
    std::array<uint8_t, 1024> encoded{};
    for (uint64_t begin = 0; begin < state.accepted_frontier;) {
        const auto count = size_t(std::min<uint64_t>(rows.size(), state.accepted_frontier - begin));
        read_rows(begin, rows.data(), count);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < 4; ++j) encoded[4*i+j] = uint8_t(uint32_t(rows[i]) >> (8*j));
        out.write(encoded.data(), count*sizeof(int32_t));
        begin += count;
    }
    for (uint64_t i = 0; i < state.events; ++i) {
        const auto event = read_event(i);
        put(out, event.id); put(out, event.begin); put(out, event.end);
        put(out, static_cast<uint8_t>(event.kind), 1); put(out, event.complete ? 1 : 0, 1);
    }
    validate();
    const auto hash = out.hash(); put(out, hash);
    validate();
    if (out.bytes() != bytes) throw std::logic_error("source image size disagrees with encoding");
    return bytes;
}
void ram_staging(const HostKvStorage & staging) {
    if (staging.spill_file())
        throw MemoryError(MemoryErrorCode::Unsupported, "source image requires RAM staging");
}
uint64_t staging_size(const SourceState & state) {
    if (state.accepted_frontier > SIZE_MAX/sizeof(int32_t) || state.events > SIZE_MAX/sizeof(SourceEvent))
        throw SnapshotCorrupt("source image staging geometry overflow");
    const auto token_bytes = state.accepted_frontier * sizeof(int32_t);
    const auto events_size = state.events * sizeof(SourceEvent);
    if (token_bytes > UINT64_MAX - events_size)
        throw SnapshotCorrupt("source image staging size overflow");
    return token_bytes + events_size;
}
} // namespace

uint64_t write_source_image(const SourceJournalStore & store, SourceRef ref,
    const SourceImageLimits & limits, SnapshotWriter::Sink sink) {
    const auto state = store.state(ref);
    return encode_image(state, limits, std::move(sink),
        [&](uint64_t begin, int32_t * out, size_t count) { store.read_tokens(ref, begin, out, count); },
        [&](uint64_t i) { return store.event(ref, i); }, [&] { (void)store.state(ref); });
}
uint64_t write_source_image(const SourceImage & image, const SourceImageLimits & limits, SnapshotWriter::Sink sink) {
    const auto state = image.origin();
    return encode_image(state, limits, std::move(sink),
        [&](uint64_t begin, int32_t * out, size_t count) { image.read_tokens(begin, out, count); },
        [&](uint64_t i) { return image.event(i); }, [&] {
            if (!(image.origin().ref == state.ref) || image.image_bytes() != source_image_bytes(state, limits))
                throw MemoryError(MemoryErrorCode::InvalidSession, "source image owner changed during export");
        });
}
SourceImage prepare_source_image(const SourceJournalStore & store, SourceRef ref,
    const SourceImageLimits & limits, HostKvStorage & staging) {
    ram_staging(staging);
    const auto state = store.state(ref);
    const auto bytes = source_image_bytes(state, limits);
    auto reserved = staging.reserve(staging_size(state));
    HostKvRecord tokens, events;
    if (state.accepted_frontier) tokens = reserved.allocate(size_t(state.accepted_frontier*sizeof(int32_t)));
    if (state.events) events = reserved.allocate(size_t(state.events*sizeof(SourceEvent)));
    std::array<int32_t, 256> rows{};
    for (uint64_t begin = 0; begin < state.accepted_frontier;) {
        const auto count = size_t(std::min<uint64_t>(rows.size(), state.accepted_frontier - begin));
        store.read_tokens(ref, begin, rows.data(), count);
        std::memcpy(tokens.writable_data() + begin*sizeof(int32_t), rows.data(), count*sizeof(int32_t));
        begin += count;
    }
    for (uint64_t i = 0; i < state.events; ++i) {
        const auto event = store.event(ref, i);
        std::memcpy(events.writable_data() + i*sizeof(event), &event, sizeof(event));
    }
    (void)store.state(ref);
    tokens.seal(); events.seal();
    return SourceImage(state, std::move(tokens), std::move(events), bytes);
}

SourceImage read_source_image(SnapshotReader::Source source, uint64_t bytes,
    const SourceConfiguration & expected, const SourceImageLimits & limits, HostKvStorage & staging) {
    limits_valid(limits);
    if (!source || !configuration_present(expected))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source image needs a transport and current configuration");
    ram_staging(staging);
    if (bytes < fixed_bytes) throw SnapshotCorrupt("truncated source image");
    if (bytes > limits.max_bytes)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "source image exceeds transport byte limit");
    SnapshotReader in(std::move(source), bytes);
    if (get(in) != magic || get(in) != version || get(in) != bytes)
        throw SnapshotCorrupt("incompatible source image framing");
    SourceState state;
    state.ref = get_ref(in); state.parent = get_ref(in);
    state.parent_prefix = get(in); state.accepted_frontier = get(in);
    state.computed_frontier = get(in); state.events = get(in);
    in.read(state.configuration.data(), state.configuration.size());
    if (!metadata_valid(state) || encoded_size(state) != bytes)
        throw SnapshotCorrupt("inconsistent source image metadata or length");
    if (state.configuration != expected)
        throw MemoryError(MemoryErrorCode::Unsupported, "source image configuration differs from current source encoding");
    if (state.accepted_frontier > limits.max_tokens || state.events > limits.max_events)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "source image exceeds row/event limits");
    auto reserved = staging.reserve(staging_size(state));
    const auto token_bytes = state.accepted_frontier * sizeof(int32_t);
    const auto events_size = state.events * sizeof(SourceEvent);
    HostKvRecord tokens, events;
    if (token_bytes) tokens = reserved.allocate(size_t(token_bytes));
    if (events_size) events = reserved.allocate(size_t(events_size));
    std::array<uint8_t, 1024> encoded{};
    for (uint64_t begin = 0; begin < state.accepted_frontier;) {
        const auto count = size_t(std::min<uint64_t>(256, state.accepted_frontier - begin));
        in.read(encoded.data(), count*sizeof(int32_t));
        for (size_t i = 0; i < count; ++i) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; ++j) value |= uint32_t(encoded[4*i+j]) << (8*j);
            if (value > uint32_t(INT32_MAX)) throw SnapshotCorrupt("negative source image token");
            const int32_t row = int32_t(value);
            std::memcpy(tokens.writable_data() + (begin+i)*sizeof(row), &row, sizeof(row));
        }
        begin += count;
    }
    uint64_t previous_end = 0;
    for (uint64_t i = 0; i < state.events; ++i) {
        SourceEvent event;
        event.id = get(in); event.begin = get(in); event.end = get(in);
        const auto kind = get(in, 1), complete = get(in, 1);
        if (!event.id || event.begin >= event.end || event.end > state.accepted_frontier ||
            event.begin < previous_end || kind > static_cast<uint8_t>(SourceEventKind::RawInput) || complete > 1)
            throw SnapshotCorrupt("invalid source image event");
        event.kind = static_cast<SourceEventKind>(kind); event.complete = complete != 0;
        std::memcpy(events.writable_data() + i*sizeof(event), &event, sizeof(event));
        previous_end = event.end;
    }
    if (in.remaining() != 8) throw SnapshotCorrupt("source image has trailing data");
    const auto hash = in.hash();
    if (get(in) != hash || in.remaining()) throw SnapshotCorrupt("source image checksum mismatch");
    tokens.seal(); events.seal();
    return SourceImage(state, std::move(tokens), std::move(events), bytes);
}

void SourceImage::read_tokens(uint64_t begin, int32_t * out, size_t count) const {
    if ((count && !out) || begin > origin_.accepted_frontier || count > origin_.accepted_frontier - begin)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source image read exceeds accepted rows");
    if (count) std::memcpy(out, tokens_.data() + begin*sizeof(int32_t), count*sizeof(int32_t));
}
SourceEvent SourceImage::event(uint64_t index) const {
    if (index >= origin_.events) throw MemoryError(MemoryErrorCode::InvalidPlan, "source image event out of range");
    SourceEvent value;
    std::memcpy(&value, events_.data() + index*sizeof(value), sizeof(value));
    return value;
}
} // namespace kvmem

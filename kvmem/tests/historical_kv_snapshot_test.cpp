#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/memory_contract.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace kvmem;
namespace {
void require(bool value) { if (!value) throw std::runtime_error("historical snapshot contract"); }
template<class Function> void corrupt(Function fn) {
    bool failed = false;
    try { fn(); } catch (const SnapshotCorrupt &) { failed = true; }
    require(failed);
}
template<class T> std::vector<uint8_t> encode(const T & object) {
    std::vector<uint8_t> bytes;
    SnapshotWriter out([&](const void * p, size_t n) {
        const auto * begin = static_cast<const uint8_t *>(p); bytes.insert(bytes.end(), begin, begin+n);
    });
    object.snapshot_write(out); return bytes;
}
SnapshotReader reader(const std::vector<uint8_t> & bytes, size_t count = SIZE_MAX) {
    const auto size = std::min(count, bytes.size());
    return SnapshotReader([&, offset=size_t(0), size](void * p, size_t n) mutable {
        require(n <= size-offset); if (n) std::memcpy(p, bytes.data()+offset, n); offset += n;
    }, size);
}
void features() {
    auto budget = std::make_shared<HistoricalKvAuxBudget>(1<<20);
    HistoricalKvPageFeatures original(4,3,budget);
    original.publish_page(0,{1,2,3}); original.publish_page(2,{-1,0,4});
    original.publish_question_scores({-2,5,0});
    original.publish_output_attention(0,0.25); original.publish_output_attention(2,3);
    require(original.begin_round(1)); require(original.begin_round(1));
    const auto bytes = encode(original); const auto before = budget->stats().charged;
    {
        auto in = reader(bytes); auto copy = HistoricalKvPageFeatures::snapshot_read(in,4,3,budget);
        require(!in.remaining()); require(copy->question_epoch()==1 && copy->score_epoch()==0);
        require(copy->commit_epoch_guard(1)->value != original.commit_epoch_guard(1)->value);
        require(budget->stats().charged > before);
        // A fresh worker gets the original logical scores, never its frozen round.
        require(copy->begin_round(77));
        for (uint32_t id=0; id<4; ++id) {
            require(copy->has_page(id) == original.has_page(id));
            HistoricalKvBlockMeta meta; meta.logical_id=id;
            std::array<float,3> a{},b{}; double x=0,y=0;
            require(copy->read(meta,a.data(),x) == original.read(meta,b.data(),y));
            require(a==b && x==y);
        }
        require(encode(*copy)==bytes);
        copy->clear_question(); require(copy->begin_round(77));
        HistoricalKvBlockMeta meta; meta.logical_id=2; std::array<float,3> out{}; double score=99;
        require(copy->read(meta,out.data(),score) && score==0);
        require(encode(original)==bytes);
    }
    require(budget->stats().charged == before);
    for (size_t n=0; n<bytes.size(); ++n) {
        auto in = reader(bytes,n); corrupt([&]{ (void)HistoricalKvPageFeatures::snapshot_read(in,4,3,budget); });
        require(budget->stats().charged==before);
    }
    { auto in=reader(bytes); corrupt([&]{(void)HistoricalKvPageFeatures::snapshot_read(in,4,2,budget);}); }
    for (size_t offset : {size_t(20), size_t(64)}) {
        auto bad=bytes; const uint64_t count=UINT64_MAX; std::memcpy(bad.data()+offset,&count,8);
        auto in=reader(bad); corrupt([&]{(void)HistoricalKvPageFeatures::snapshot_read(in,4,3,budget);});
    }
    { auto bad=bytes; float value=std::numeric_limits<float>::quiet_NaN(); std::memcpy(bad.data()+28,&value,4);
      auto in=reader(bad); corrupt([&]{(void)HistoricalKvPageFeatures::snapshot_read(in,4,3,budget);}); }
    auto tiny=std::make_shared<HistoricalKvAuxBudget>(1); bool denied=false;
    try { auto in=reader(bytes); (void)HistoricalKvPageFeatures::snapshot_read(in,4,3,tiny); }
    catch (const MemoryError & error) { denied=error.code()==MemoryErrorCode::BudgetExceeded; }
    require(denied && tiny->stats().charged==0 && budget->stats().charged==before);
}
void ranges() {
    auto budget=std::make_shared<HistoricalKvAuxBudget>(1<<20);
    HistoricalKvThinkingRanges original(budget);
    original.note(64,128); original.note(256,300); original.note(64,192); original.arm();
    original.enable_completed_replies(true);
    original.note_completed_reply(0,192); original.note_completed_reply(256,384);
    const auto bytes=encode(original); const auto before=budget->stats().charged;
    {
        auto in=reader(bytes); auto copy=HistoricalKvThinkingRanges::snapshot_read(in,384,budget);
        require(!in.remaining() && copy->armed() && copy->tiered() && encode(*copy)==bytes);
        for (uint64_t p=0;p<384;++p) {
            require(copy->contains(p)==original.contains(p));
            require(copy->priority(p,16)==original.priority(p,16));
        }
        HistoricalKvCommitConditions a{},b{};
        require(original.prepare_commit(64,64,1,a) && copy->prepare_commit(64,64,1,b));
        copy->truncate(100); require(!copy->covers(100,1) && original.covers(100,1));
    }
    require(budget->stats().charged==before);
    for(size_t n=0;n<bytes.size();++n) {
        auto in=reader(bytes,n); corrupt([&]{(void)HistoricalKvThinkingRanges::snapshot_read(in,384,budget);});
        require(budget->stats().charged==before);
    }
    {auto in=reader(bytes); corrupt([&]{(void)HistoricalKvThinkingRanges::snapshot_read(in,100,budget);});}
    {auto bad=bytes;bad[4]=2;auto in=reader(bad);corrupt([&]{(void)HistoricalKvThinkingRanges::snapshot_read(in,384,budget);});}
    {auto bad=bytes;uint64_t count=UINT64_MAX;std::memcpy(bad.data()+6,&count,8);auto in=reader(bad);
     corrupt([&]{(void)HistoricalKvThinkingRanges::snapshot_read(in,384,budget);});}
    auto tiny=std::make_shared<HistoricalKvAuxBudget>(1); bool denied=false;
    try {auto in=reader(bytes);(void)HistoricalKvThinkingRanges::snapshot_read(in,384,tiny);}
    catch(const MemoryError &error){denied=error.code()==MemoryErrorCode::BudgetExceeded;}
    require(denied && tiny->stats().charged==0 && budget->stats().charged==before);
}
}
int main() {
    features(); ranges(); std::cout<<"historical logical snapshots: PASS\n";
}

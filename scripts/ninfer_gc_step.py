#!/usr/bin/env python3
"""Build explicit completed-step GC evidence for NInfer's chat API.

Completion/coverage and observation scores come from the application. The
optional verified-write producer handles the teacher's narrower, reproducible
artifact case; it never invents attention scores or cold observations.
"""
import argparse
from array import array
from bisect import bisect_left, bisect_right
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
import uuid


def token_digest(ids):
    return hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()


def pieces(rows):
    return [r["piece"].encode("utf-8") if isinstance(r["piece"], str)
            else bytes(r["piece"]) for r in rows]


DUPLICATE_CONTEXT = 8


class NativeSpanCache:
    """Reuse proven field positions only inside a byte-identical native prefix."""
    def __init__(self,limit=256):
        self.raw=b''
        self.spans={}
        self.limit=limit
        self.hits=0

    def locate(self,request,sites,rendered,post):
        raw=rendered.encode('utf-8')
        stop=min(len(raw),len(self.raw));same=0
        while same+8192<=stop and raw[same:same+8192]==self.raw[same:same+8192]:same+=8192
        while same<stop and raw[same]==self.raw[same]:same+=1
        self.spans={key:span for key,span in self.spans.items() if span[1]<=same}
        self.raw=raw
        found={};missing={}
        for site,text in sites.items():
            key=(*site,text)
            if key in self.spans:
                found[site]=self.spans[key];self.hits+=1
            else:missing[site]=text
        if missing:
            marked=copy.deepcopy(request)
            markers={};owners={}
            right_edges={}
            for site,text in sorted(missing.items(),reverse=True):
                index,field,begin,end=site
                if end>right_edges.get((index,field),end):raise ValueError('Overlapping native spans')
                right_edges[index,field]=begin
                original=marked['messages'][index][field]
                if original[begin:end]!=text:raise ValueError('Owned field bytes changed')
                marker='KVMEM_SPAN_'+uuid.uuid4().hex
                marked['messages'][index][field]=original[:begin]+marker+original[end:]
                markers[marker]=text;owners[marker]=site
            marked.pop('return_tokens',None)
            variant=post('/apply-template',marked)['prompt']
            for marker,span in rendered_marker_spans(rendered,variant,markers).items():
                site=owners[marker];found[site]=span
                self.spans[(*site,sites[site])]=span
            while len(self.spans)>self.limit:self.spans.pop(next(iter(self.spans)))
        return found


class TokenByteIndex:
    """One byte-offset index per tokenized prompt, shared by all scope lookups."""
    def __init__(self,rows,rendered):
        self.raw=rendered.encode('utf-8')
        self.parts=pieces(rows)
        if b''.join(self.parts)!=self.raw:raise ValueError('Token bytes do not exactly reproduce the rendered prompt')
        self.offsets=[0];self.empty=[]
        for i,part in enumerate(self.parts):
            self.offsets.append(self.offsets[-1]+len(part))
            if not part:self.empty.append(i)

    def range(self,text,byte_span=None,intersect=False):
        needle=text.encode('utf-8')
        if not needle or byte_span is None and self.raw.count(needle)!=1:
            raise ValueError('Evidence text must occur exactly once in the prepared prompt')
        begin,end=byte_span if byte_span is not None else (self.raw.index(needle),self.raw.index(needle)+len(needle))
        if not 0<=begin<end<=len(self.raw) or self.raw[begin:end]!=needle:
            raise ValueError('Evidence bytes differ from their proven native-render span')
        start=bisect_right(self.offsets,begin)-1 if intersect else bisect_left(self.offsets,begin)
        stop=bisect_left(self.offsets,end) if intersect else bisect_right(self.offsets,end)-1
        while start<stop and not self.parts[start]:start+=1
        while stop>start and not self.parts[stop-1]:stop-=1
        empty=bisect_left(self.empty,start)
        if start>=stop or empty<len(self.empty) and self.empty[empty]<stop:
            raise ValueError('Evidence has no contiguous complete-token range')
        return start,stop


def rendered_marker_spans(rendered, variant, spans):
    """Locate markers only if restoring their original bytes reproduces the prompt."""
    markers = re.compile("|".join(map(re.escape, spans)))
    restored, byte_spans, end, offset = [], {}, 0, 0
    for match in markers.finditer(variant):
        marker = match[0]
        if marker in byte_spans:
            raise ValueError("Duplicated native provenance marker")
        chunk = variant[end:match.start()].encode("utf-8")
        raw = spans[marker].encode("utf-8")
        offset += len(chunk)
        byte_spans[marker] = (offset, offset + len(raw))
        offset += len(raw)
        restored.extend((chunk, raw))
        end = match.end()
    restored.append(variant[end:].encode("utf-8"))
    if len(byte_spans) != len(spans) or b"".join(restored) != rendered.encode("utf-8"):
        raise ValueError("Marker rendering did not preserve the original native prompt")
    return byte_spans


def terminal_feedback_ranges(request, feedback, prepared, post, span_cache=None):
    """Locate completed, owned feedback using one native marker rendering."""
    marked = copy.deepcopy(request)
    spans = {}
    sites = {}
    for index, text in feedback:
        message = marked["messages"][index]
        text = text.strip()
        if message.get("role") != "user" or not text or message["content"].count(text) != 1:
            continue
        begin=message['content'].index(text)
        sites[(index,'content',begin,begin+len(text))]=text
        marker = "KVMEM_TERMINAL_" + uuid.uuid4().hex
        message["content"] = message["content"].replace(text, marker, 1)
        spans[marker] = text
    if not spans:
        return []
    marked.pop("return_tokens", None)
    if span_cache is not None:
        byte_spans=span_cache.locate(request,sites,prepared['prompt'],post)
    else:
        variant = post("/apply-template", marked)["prompt"]
        byte_spans = rendered_marker_spans(prepared["prompt"], variant, spans)
    parts = pieces(prepared["tokens"])
    raw_prompt = b"".join(parts)
    if raw_prompt != prepared["prompt"].encode("utf-8"):
        raise ValueError("Terminal feedback must exactly match the native-rendered bytes")
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + len(part))
    ordered=(byte_spans[site] for site in sites) if span_cache is not None else byte_spans.values()
    return [(bisect_left(offsets, begin), bisect_right(offsets, end) - 1) for begin, end in ordered]


def duplicate_feedback_page_sets(step, page_tokens=64):
    source = range((step["source_begin"] + DUPLICATE_CONTEXT) // page_tokens,
                   (step["source_end"] - DUPLICATE_CONTEXT) // page_tokens)
    witness = range(step["result_begin"] // page_tokens,
                    (step["result_end"] + page_tokens - 1) // page_tokens)
    return source, witness


def duplicate_feedback_steps(ids, ranges, *, sink_tokens=64, source_pages=(), witness_pages=(), limit=4):
    """Propose whole older pages with exact copies and context in later feedback."""
    page_tokens = 64
    margin = DUPLICATE_CONTEXT
    packed = array("I", ids).tobytes()
    word_bytes = array("I").itemsize
    blocked_sources = set(source_pages)
    pinned = set(witness_pages)
    sources, earliest = [], {}
    for index, (begin, end) in enumerate(ranges):
        first = max(1, sink_tokens // page_tokens, (begin + margin + page_tokens - 1) // page_tokens)
        for page in range(first, (end - margin) // page_tokens):
            if page in blocked_sources or page in pinned:
                continue
            start = page * page_tokens
            key = packed[(start - margin)*word_bytes:(start + page_tokens + margin)*word_bytes]
            sources.append((page, index, key))
            earliest.setdefault(key, index)
    if not sources:
        return []

    # At most the three newest completed feedback spans supply witnesses.
    # Hash full slices to bound work even when logs repeat the same padding.
    copies = {}
    width = page_tokens + 2 * margin
    for index in range(len(ranges) - 1, max(0, len(ranges) - 4), -1):
        begin, end = ranges[index]
        start = (begin + page_tokens - 1) // page_tokens * page_tokens
        stop = min(end // page_tokens * page_tokens, (len(ids) - page_tokens) // page_tokens * page_tokens)
        for offset in range(start, stop - width + 1):
            key = packed[offset*word_bytes:(offset + width)*word_bytes]
            if key not in copies and earliest.get(key, index) < index:
                pages = range(offset // page_tokens, (offset + width - 1) // page_tokens + 1)
                if blocked_sources.isdisjoint(pages):
                    copies[key] = (index, offset)

    groups = []
    for page, source_index, key in sources:
        match = copies.get(key)
        if match is None or match[0] <= source_index:
            continue
        result_index, offset = match
        if groups:
            first, count, old, result, start = groups[-1]
            if count < 6 and (page, source_index, result_index, offset) == (
                    first + count, old, result, start + count * page_tokens):
                groups[-1] = (first, count + 1, old, result, start)
                continue
        groups.append((page, 1, source_index, result_index, offset))

    steps = []
    for page, count, _, _, offset in sorted(groups, key=lambda row: (-row[1], row[0])):
        begin = page * page_tokens - margin
        length = count * page_tokens + 2 * margin
        fingerprint = hashlib.sha256(packed[begin*word_bytes:(begin + length)*word_bytes]).hexdigest()[:16]
        step = dict(id=f"feedback-copy-{page}-{offset}-{count}-{fingerprint}", kind="duplicate_tool_feedback",
                    source_begin=begin, source_end=begin + length, result_begin=offset,
                    result_end=offset + length, observation_id=1, importance=1.0, cold=False, complete=True)
        src, dst = duplicate_feedback_page_sets(step)
        if not pinned.isdisjoint(src) or not blocked_sources.isdisjoint(dst):
            continue
        steps.append(step)
        pinned.update(dst)
        blocked_sources.update(src)
        if len(steps) == limit:
            break
    return steps


def token_range(rows, rendered, text, byte_span=None, intersect=False):
    """Sources use interior tokens; witnesses include all intersecting tokens."""
    return TokenByteIndex(rows,rendered).range(text,byte_span,intersect)


def build_evidence(rows, rendered, steps, block_tokens=64, proven_spans=None):
    """Map application-approved source/result text, rejecting mixed source blocks.

    A short result may be protected by a whole page intersecting its tokens,
    including extra framing. The page must already be inside the accepted
    prefix. This follows the verified-artifact witness rule in the source.
    """
    if not 1 <= len(steps) <= 64:
        raise ValueError("A batch must contain 1..64 steps")
    mapped = []
    index=TokenByteIndex(rows,rendered)
    for step in steps:
        row = copy.deepcopy(step)
        if any(key.startswith('_') for key in row):
            raise ValueError("Verified writes require attach_evidence's native rendering check")
        spans = (proven_spans or {}).get(row['id'], {})
        sb, se = index.range(row.pop("source_text"), spans.get('source'))
        rb, re = index.range(row.pop("result_text"), spans.get('result'), intersect=True)
        sb = (sb + block_tokens - 1) // block_tokens * block_tokens
        se = se // block_tokens * block_tokens
        if sb >= se:
            raise ValueError("Source contains no full interior block")
        # Protect an existing full witness block; never manufacture padding.
        wb = rb // block_tokens * block_tokens
        we = ((re + block_tokens - 1) // block_tokens) * block_tokens
        if we > len(rows):
            raise ValueError("Result witness is not a complete accepted block yet")
        if sb < we and wb < se:
            raise ValueError("Result witness would overlap the source")
        row.update(source_begin=sb, source_end=se, result_begin=wb, result_end=we)
        mapped.append(row)
    return {"prefix_tokens": len(rows), "prefix_sha256": token_digest([r["id"] for r in rows]),
            "steps": mapped}


def verified_write_step(messages, call_id, workspace):
    """DSH write: exact file bytes + paired Created/Updated tool result.

    Only the literal content parameter can become a source. Paths, tool
    framing, thinking, and the result stay outside it. Changed files, ambiguous
    call IDs, escaped/absent rendered bodies and symlink escapes fail closed.
    The body is matched again by build_evidence against native rendering.
    """
    calls = [c for m in messages if m.get("role") == "assistant"
             for c in m.get("tool_calls", []) if c.get("id") == call_id]
    results = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == call_id]
    if len(calls) != 1 or len(results) != 1 or calls[0].get("function", {}).get("name") != "write":
        raise ValueError("Expected one paired write call and tool result")
    args = json.loads(calls[0]["function"]["arguments"])
    if set(args) != {"file_path", "content"} or not isinstance(args["content"], str):
        raise ValueError("Unsupported write arguments")
    root = Path(workspace).resolve(strict=True)
    path = Path(args["file_path"])
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Artifact is outside the verified workspace")
    if resolved.stat().st_size > 1048576 or resolved.read_bytes() != args["content"].encode("utf-8"):
        raise ValueError("Artifact differs from the accepted write body")
    expected = [f"<path>{path}</path>\n<type>file</type>\n<content>\n{verb} file\n</content>"
                for verb in ("Created", "Updated")]
    result = results[0].get("content")
    if result not in expected:
        raise ValueError("Write result is not the expected successful artifact witness")
    return dict(id="file-body-" + hashlib.sha256(call_id.encode()).hexdigest()[:24],
                kind="tool_detail", source_text=args["content"], result_text=result,
                observation_id=1, importance=1.0, cold=False, complete=True,
                _write_call_id=call_id, _result_message_index=messages.index(results[0]))


def post_json(base, path, value):
    req = urllib.request.Request(base.rstrip("/") + path,
        data=json.dumps(value, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def owned_message_span(messages, span, kind, snippet):
    index, field, begin, end = (span[k] for k in ('message', 'field', 'begin', 'end'))
    message = messages[index]
    text = message.get(field)
    inline_summary = False
    if kind == 'result' and span.get('kind') == 'inline_step_summary' and message.get('role') == 'assistant' and field == 'content':
        from ninfer_gc_inline_summary import summary_span
        _, summary_begin, summary_end = summary_span(text)
        inline_summary = (begin, end) == (summary_begin, summary_end)
    if (field not in ('content', 'reasoning_content') or not isinstance(text, str) or
        not 0 <= begin < end <= len(text) or text[begin:end] != snippet or
        kind == 'source' and (message.get('role') != 'assistant' or field != 'reasoning_content') or
        kind == 'result' and not inline_summary and (message.get('role') != 'user' or field != 'content')):
        raise ValueError('Invalid owned thinking/summary message span')
    return index, field, begin, end


def inline_marker_spans(request, steps, rendered, post, span_cache=None):
    marked = copy.deepcopy(request)
    spans, sites, owners = {}, {}, {}
    for step in steps:
        for kind in ('source', 'result'):
            snippet = step[kind + '_text']
            location = owned_message_span(request['messages'], step[f'_{kind}_message_span'], kind, snippet)
            if location not in sites:
                marker = 'KVMEM_ARTIFACT_' + uuid.uuid4().hex
                sites[location] = marker
                spans[marker] = snippet
            owners.setdefault(sites[location], []).append((step['id'], kind))
    if span_cache is not None:
        locations=span_cache.locate(request,{site:spans[marker] for site,marker in sites.items()},rendered,post)
        proven={}
        for site,span in locations.items():
            for identity,kind in owners[sites[site]]:proven.setdefault(identity,{})[kind]=span
        return proven
    right_edges = {}
    for (index, field, begin, end), marker in sorted(sites.items(), reverse=True):
        if end > right_edges.get((index, field), end):
            raise ValueError('Overlapping spans require separate native marker renders')
        right_edges[index, field] = begin
        text = marked['messages'][index][field]
        marked['messages'][index][field] = text[:begin] + marker + text[end:]
    variant = post('/apply-template', marked)['prompt']
    proven = {}
    for marker, byte_span in rendered_marker_spans(rendered, variant, spans).items():
        for identity, kind in owners[marker]:
            proven.setdefault(identity, {})[kind] = byte_span
    return proven


def attach_evidence(base, request, steps, post=None, span_cache=None):
    if not request.get("prompt_cache_key"):
        raise ValueError("A named prompt_cache_key is required")
    result = copy.deepcopy(request)
    result.pop("kvmem_gc", None)
    if post is None:
        post = lambda path, value: post_json(base, path, value)
    prepared = post("/apply-template", dict(result, return_tokens=True))
    rendered = prepared["prompt"]
    rows = prepared.get("tokens")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Native template endpoint must return its actual prepared tokens")
    approved = copy.deepcopy(steps)
    span_keys = {'_source_message_span', '_result_message_span'}
    if approved and all({k for k in step if k.startswith('_')} == span_keys for step in approved):
        try:
            proven = inline_marker_spans(result, approved, rendered, post,span_cache)
        except ValueError:
            # Some templates transform adjacent fields when both are marked.
            # Keep the existing byte-exact proof for each span in that case.
            pass
        else:
            for step in approved:
                for key in span_keys:
                    del step[key]
            result['kvmem_gc'] = build_evidence(rows, rendered, approved, proven_spans=proven)
            return result
    proven = {}
    for step in approved:
        call_id = step.pop("_write_call_id", None)
        terminal = step.pop('_terminal_body', None)
        result_message = step.pop('_result_message_index', None)
        source_span = step.pop('_source_message_span', None)
        result_span = step.pop('_result_message_span', None)
        for kind in ('source', 'result'):
            span = source_span if kind=='source' else result_span
            if span is None and (kind == 'source' and call_id is None and terminal is None or kind == 'result' and result_message is None):
                continue
            changed = copy.deepcopy(result)
            marker = "KVMEM_ARTIFACT_" + uuid.uuid4().hex
            if span is not None:
                index, field, begin, end = owned_message_span(changed['messages'], span, kind, step[kind + '_text'])
                message = changed['messages'][index]
                text = message[field]
                message[field] = text[:begin] + marker + text[end:]
            elif kind == 'result':
                changed['messages'][result_message]['content'] = marker
            elif terminal is not None:
                message = changed['messages'][terminal['message']]
                text = message['content']
                message['content'] = text[:terminal['begin']] + marker + text[terminal['end']:]
            else:
                calls = [c for m in changed["messages"] if m.get("role") == "assistant"
                         for c in m.get("tool_calls", []) if c.get("id") == call_id]
                if len(calls) != 1:
                    raise ValueError("Artifact call lost its unique assistant provenance")
                args = json.loads(calls[0]["function"]["arguments"])
                args["content"] = marker
                calls[0]["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
            variant = post("/apply-template", changed)["prompt"]
            if variant.count(marker) != 1:
                raise ValueError("Native template did not preserve the artifact marker")
            before, after = variant.split(marker)
            text = step[kind + '_text']
            if before + text + after != rendered:
                raise ValueError("Artifact bytes were transformed or have incorrect message provenance")
            begin = len(before.encode('utf-8'))
            proven.setdefault(step['id'], {})[kind] = (begin, begin + len(text.encode('utf-8')))
    result["kvmem_gc"] = build_evidence(rows, rendered, approved, proven_spans=proven)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--request", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--steps", type=Path, help="Application-approved source/result text and observations")
    group.add_argument("--verified-write-call", help="Opt-in DSH artifact profile: thresholds 1/0/1 required")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Prepared chat request JSON; contains prompt text")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.verified_write_call:
        if not args.workspace:
            parser.error("--verified-write-call requires --workspace")
        steps = [verified_write_step(request["messages"], args.verified_write_call, args.workspace)]
    else:
        steps = json.loads(args.steps.read_text(encoding="utf-8"))
    value = attach_evidence(args.base_url, request, steps)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Prepared evidence request; submit it to /v1/chat/completions at the next step boundary.")


if __name__ == "__main__":
    main()

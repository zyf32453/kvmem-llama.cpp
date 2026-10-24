"""Check a native-template full-Query model smoke run made by the existing driver.

This proves input capture/range accounting, prefix reuse, and three small output
constraints. It does not establish GC semantic quality, p95, or throughput.
The workload driver additionally requires actual GC activity; for this short
probe its exit status can be nonzero solely because no GC was needed. This
checker reports that distinction and independently checks all probe evidence.
"""
import argparse
import json
import math
from pathlib import Path
import re

from test_server_historical_workload import full_question_report


DEFAULT_FIXTURE = Path(__file__).with_name('fixtures') / 'full-query-coverage-workload.json'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def check_probe(directory, fixture=DEFAULT_FIXTURE):
    directory = Path(directory)
    expected = json.loads(Path(fixture).read_text(encoding='utf-8'))
    issues = []
    rows = []

    def require(condition, message):
        if not condition:
            issues.append(message)

    def read_json(name):
        return json.loads((directory / name).read_text(encoding='utf-8'))

    try:
        saved = read_json('workload.json')
        results = read_json('results.json')
        requests = read_json('requests.json')
        config = read_json('experiment-config.json')
        summary = read_json('summary.json')
        lines = (directory / 'server.log').read_text(encoding='utf-8', errors='replace').splitlines()
    except (OSError, ValueError) as error:
        return {'passed': False, 'scope': 'full_query_capture_and_output_smoke',
                'issues': [f'missing or invalid run evidence: {error}']}

    count = len(expected['turns'])
    require(saved == expected, 'saved workload differs from the checked fixture')
    require(config.get('chat_template') == summary.get('chat_template') == 'native',
            'native model template is required')
    require(config.get('policy') == summary.get('policy') == 'redundancy' and
            config.get('mode') == summary.get('mode') and config.get('mode') in ('shadow', 'delete'),
            'an enabled redundancy source is required')
    require(config.get('block_tokens') == summary.get('block_tokens') == 32,
            'this probe requires 32-token KV blocks')
    require(len(results) == len(requests) == count, 'run must complete exactly all fixture turns')
    require(not summary.get('gc_failure_observed', True), 'GC failure or absent failure accounting')
    require(not any(row.get('policy_failures', 0) or row.get('worker_failures', 0)
                    for row in summary.get('gc_finish_samples', [])), 'GC worker or policy failed')

    coverage = full_question_report(lines, count)
    spans = coverage['full_question_spans']
    require(coverage['full_question_coverage_observed'], 'full Query coverage evidence is incomplete')
    require(len(spans) == count, 'each turn must have exactly one located full Query span')
    require(all(current.get('begin', -1) >= previous.get('end', 0)
                for previous, current in zip(spans, spans[1:])),
            'continuation Query spans must advance past the preceding user message')
    # Keep per-span finish rows: a stale nonzero segment count is not evidence
    # that this entire input was represented at the configured segment size.
    finishes = []
    for line in lines:
        if 'KVMEM_TRACE gc_query_full ' in line:
            finishes.append([])
        if 'KVMEM_HISTORICAL_REDUNDANCY features=' in line and finishes:
            finishes[-1].append({key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)})

    for index in range(count):
        turn_issues = []
        prompt = expected['turns'][index]
        contract = expected['probe']['expected'][index]
        if index >= len(results) or index >= len(requests):
            rows.append({'turn': index + 1, 'passed': False, 'issues': ['missing request or result']})
            continue
        result = results[index]
        request = requests[index]
        if not (result.get('turn') == index + 1 and result.get('done') is True and
                result.get('http_status') == 200 and result.get('finish_reason') == 'stop' and
                not result.get('error') and not result.get('error_body')):
            turn_issues.append('HTTP response incomplete, failed, or truncated')
        messages = request.get('messages', [])
        wanted = [{'role': 'system', 'content': expected['system']}]
        for previous in range(index):
            wanted.append({'role': 'user', 'content': expected['turns'][previous]})
            assistant = {'role': 'assistant', 'content': results[previous].get('content', '')}
            if results[previous].get('reasoning_content'):
                assistant['reasoning_content'] = results[previous]['reasoning_content']
            wanted.append(assistant)
        wanted.append({'role': 'user', 'content': prompt})
        if messages != wanted:
            turn_issues.append('request does not contain the exact own-response conversation')
        if index and result.get('usage', {}).get('prompt_cache_hit_tokens', 0) <= 0:
            turn_issues.append('continuation has no observed prefix cache reuse')
        actual = None
        try:
            actual = json.loads(result.get('content', ''), object_pairs_hook=unique_object)
            if actual != contract:
                turn_issues.append('output does not match all head/middle/tail constraints exactly')
        except (ValueError, TypeError) as error:
            turn_issues.append(f'output must be a single JSON object: {error}')
        tokens = spans[index].get('tokens', 0) if index < len(spans) else 0
        required_segments = math.ceil(tokens / expected['probe']['question_segment_tokens'])
        if tokens < expected['probe']['minimum_query_tokens']:
            turn_issues.append('actual full Query is too short to test long-input coverage')
        segment_counts = ([row.get('question_segments', 0) for row in finishes[index]]
                          if index < len(finishes) else [])
        if not segment_counts or any(value != required_segments for value in segment_counts):
            turn_issues.append('segment count does not cover the whole Query at 32 tokens per segment')
        row = {'turn': index + 1, 'passed': not turn_issues, 'query_tokens': tokens,
               'expected_segments': required_segments, 'observed_segments': segment_counts,
               'prefix_cache_hit_tokens': result.get('usage', {}).get('prompt_cache_hit_tokens', 0),
               'expected': contract, 'actual': actual, 'issues': turn_issues}
        rows.append(row)
    require(all(row['passed'] for row in rows), 'one or more turns failed the probe')
    return {'passed': not issues, 'scope': 'full_query_capture_and_output_smoke',
            'issues': issues, 'turns': rows, 'coverage': coverage,
            'gc_activity_observed': summary.get('required_gc_activity_observed', False),
            'generated_query_observation_samples': summary.get('max_observation_samples', 0),
            'generated_query_fast_samples': summary.get('max_fast_samples', 0),
            'gc_quality_accepted': False, 'performance_accepted': False,
            'limitations': [
                'Successful recall can come from ordinary model attention; it does not prove the GC ranking preserved it.',
                'All-position segment coverage does not prove lossless token or semantic representation.',
                'No GC activity is required for this short capture/output probe.',
                'Generated Query observation counters are reported, not a freshness or quality acceptance gate.',
                'No p95 or throughput acceptance is performed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='existing workload-driver output directory')
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--report', type=Path, help='default: OUTPUT/full-query-probe-check.json')
    args = parser.parse_args()
    report = check_probe(args.output, args.fixture)
    destination = args.report or args.output / 'full-query-probe-check.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

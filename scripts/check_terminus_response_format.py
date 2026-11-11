"""Check native constrained replies against Harbor's installed Terminus parser.

Uses synthetic protocol examples only. Does not execute model commands or read
any benchmark solution/tests. Run with the Harbor runtime Python after enabling
NInfer --structured-output, then reset model/Host state before a real trial.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import urllib.request

from terminus_response_format import terminus_response_format


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    parser = TerminusJSONPlainParser()
    source = Path(inspect.getfile(TerminusJSONPlainParser))
    cases = [
        dict(analysis="Read environment", plan="List files", commands=[dict(
            keystrokes="ls -la\n", duration=.1)], task_complete=False),
        dict(analysis="Escaped characters", plan="Write literal text", commands=[dict(
            keystrokes="cat > example.txt <<'EOF'\nquoted \"text\" \\ 中文 {}\nEOF\n",
            duration=1.0)], task_complete=False),
        dict(analysis="Finished", plan="No further commands", commands=[], task_complete=True),
    ]
    args.out.mkdir(exist_ok=False)
    evidence = dict(parser_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=[])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for index, expected in enumerate(cases, 1):
        body = dict(model=args.model, messages=[dict(role="user", content=(
            "Return exactly the following object, without changing any string value:\n" +
            json.dumps(expected, ensure_ascii=False)))], response_format=terminus_response_format(),
            max_tokens=1024, thinking_budget=128, temperature=.6, seed=1000,
            top_p=.95, top_k=20, min_p=0, prompt_cache_key=f"protocol-preflight-{index}")
        request = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        (args.out / f'{index}-request.json').write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8')
        with opener.open(request, timeout=180) as response:
            value = json.load(response)
        (args.out / f'{index}-response.json').write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        content = value['choices'][0]['message']['content']
        result = parser.parse_response(content)
        assert not result.error, result.error
        assert not result.warning, result.warning
        assert json.loads(content) == expected, 'The model changed an expected value'
        assert [dict(keystrokes=c.keystrokes, duration=c.duration) for c in result.commands] == expected['commands']
        assert result.is_task_complete == expected['task_complete']
        evidence['cases'].append(dict(case=index, passed=True, parser_error=result.error,
            parser_warning=result.warning, usage=value['usage']))
        (args.out / 'result.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()

"""Use the normal Terminus analysis as retained step state; zero added tokens.

This is a redundancy witness, not a claim that analysis semantically covers all
thinking. Native block similarity/importance must still qualify each page.
"""
import json
from terminus_response_format import terminus_response_format

MAX_SUMMARY_CHARS = 4096  # Producer bound; does not constrain normal model output.
PROMPT_TEMPLATE_OPTIONS = (
    'tools', 'tool_choice', 'chat_template_kwargs', 'continue_final_message',
    'add_generation_prompt', 'enable_thinking', 'preserve_thinking',
    'reasoning_effort', 'thinking_budget', 'graft',
)


def response_format():
    return terminus_response_format()


def summary_span(content):
    """Return the exact JSON value span, including quotes/escapes, in real output.

    A substring/regex search cannot distinguish a nested key, quoted example or
    duplicate key. Parse the complete root object and walk its actual members.
    """
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    data = json.loads(content, object_pairs_hook=unique)
    if not isinstance(data, dict): raise ValueError('Expected a JSON response object')
    summary = data.get('analysis')
    if not isinstance(summary, str) or len(summary) > MAX_SUMMARY_CHARS:
        raise ValueError('Missing or oversized inline step summary')
    decoder = json.JSONDecoder()
    index = content.index('{') + 1
    while True:
        while content[index].isspace(): index += 1
        if content[index] == '}': break
        key, index = decoder.raw_decode(content, index)
        while content[index].isspace(): index += 1
        if content[index] != ':': raise ValueError('Invalid JSON member')
        index += 1
        while content[index].isspace(): index += 1
        begin = index
        _, index = decoder.raw_decode(content, index)
        if key == 'analysis': return summary, begin, index
        while content[index].isspace(): index += 1
        if content[index] == ',': index += 1
    raise ValueError('Missing inline summary')

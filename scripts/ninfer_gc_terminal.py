"""Verified file-body evidence for the owned Terminus-2 shell protocol.

This recognizes literal cat/heredoc writes only. It never executes the proposed
shell command and never treats its analysis, plan, or thinking as covered.
"""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

MAX_BYTES = 1048576


class WorkspaceReader:
    def __init__(self, workspace, terminal_root='/app'):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = PurePosixPath(terminal_root)
        if not self.root.is_absolute():
            raise ValueError('Terminal root must be absolute')

    def __call__(self, name):
        path = PurePosixPath(name)
        if not path.is_absolute():
            path = self.root / path
        relative = path.relative_to(self.root)
        if '..' in relative.parts:
            raise ValueError('Artifact path escapes terminal root')
        file = (self.workspace / Path(*relative.parts)).resolve(strict=True)
        if not file.is_relative_to(self.workspace) or not file.is_file() or file.stat().st_size > MAX_BYTES:
            raise ValueError('Artifact is outside the verified workspace or too large')
        with file.open('rb') as source:
            data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError('Artifact is too large')
        return data


class DockerReader:
    # The harness supplies a fixed, owned container. Neither container nor code
    # comes from model output. Path is a separate argv, never interpolated shell.
    CODE = '''import base64,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
p=pathlib.Path(sys.argv[2])
if not p.is_absolute(): p=root/p
p=p.resolve(strict=True)
assert p.is_relative_to(root) and p.is_file()
with p.open('rb') as f: data=f.read(1048577)
assert len(data)<=1048576
print(json.dumps({'body':base64.b64encode(data).decode()}))
'''

    def __init__(self, container, terminal_root='/app', docker='docker'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', container):
            raise ValueError('Supply the owned trial container ID or name')
        if not PurePosixPath(terminal_root).is_absolute():
            raise ValueError('Terminal root must be absolute')
        self.container, self.root, self.docker = container, terminal_root, docker

    def __call__(self, name):
        result = subprocess.run([self.docker, 'exec', self.container, 'python3', '-I', '-c',
                                 self.CODE, self.root, name], check=True, capture_output=True,
                                timeout=10)
        return base64.b64decode(json.loads(result.stdout)['body'], validate=True)


def literal_write(command):
    """One quoted heredoc, one cat redirection, no append/expansion/pipeline."""
    if not isinstance(command, str) or len(command.encode()) > MAX_BYTES * 2:
        raise ValueError('Unsupported command size')
    header, separator, rest = command.partition('\n')
    if not separator:
        raise ValueError('Not a literal heredoc')
    match = re.fullmatch(r"\s*cat\s+(?:>\s*(.+?)\s+<<\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\2|"
                         r"<<\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\4\s+>\s*(.+?))\s*", header)
    if not match:
        raise ValueError('Not a supported literal cat write')
    destination = match[1] or match[6]
    delimiter = match[3] or match[5]
    # Reject shell syntax even inside a quoted path: conservative path support
    # also avoids accepting an unquoted expansion or multi-command header.
    if re.search(r'[$`;&|<>\r\n]', destination):
        raise ValueError('Unsupported output path')
    words = shlex.split(destination)
    if len(words) != 1 or not words[0] or words[0].startswith('-'):
        raise ValueError('Expected exactly one output path')
    lines = rest.splitlines(keepends=True)
    endings = [i for i, line in enumerate(lines) if line.rstrip('\n') == delimiter]
    if len(endings) != 1 or any(line.strip() for line in lines[endings[0] + 1:]):
        raise ValueError('Ambiguous terminator or additional shell commands')
    body = ''.join(lines[:endings[0]])
    if not body or len(body.encode()) > MAX_BYTES:
        raise ValueError('Empty or oversized body')
    return words[0], body, len(header) + 1, len(header) + 1 + len(body)


def raw_string_span(content, value, begin, end):
    """Map decoded command offsets back to the actual JSON string spelling."""
    matches = [m for m in re.finditer(r'"(?:\\.|[^"\\])*"', content)
               if json.loads(m[0]) == value]
    if len(matches) != 1:
        raise ValueError('Command does not have unique JSON-string provenance')
    match = matches[0]
    literal = match[0]
    cursor, decoded, boundaries = 1, '', [1]
    while cursor < len(literal) - 1:
        start = cursor
        if literal[cursor] == '\\':
            cursor += 6 if literal[cursor + 1] == 'u' else 2
            part = json.loads('"' + literal[start:cursor] + '"')
            if len(part) == 1 and 0xD800 <= ord(part) <= 0xDBFF:
                cursor += 6
                part = json.loads('"' + literal[start:cursor] + '"')
        else:
            cursor += 1
            part = literal[start:cursor]
        if len(part) != 1:
            raise ValueError('Unsupported JSON escape boundary')
        decoded += part
        boundaries.append(cursor)
    if decoded != value:
        raise ValueError('JSON command bytes do not match')
    return match.start() + boundaries[begin], match.start() + boundaries[end]


def candidates(messages):
    found = {}
    for index in range(len(messages) - 1):
        message, feedback = messages[index:index + 2]
        text, result = message.get('content'), feedback.get('content')
        if message.get('role') != 'assistant' or feedback.get('role') != 'user' or not isinstance(text, str) or not isinstance(result, str):
            continue
        if not result.startswith(('New Terminal Output:', 'Previous response had warnings:')) or 'New Terminal Output:' not in result:
            continue
        try:
            start = text.index('{')
            parsed, stop = json.JSONDecoder().raw_decode(text, start)
            if text[stop:].strip() or not isinstance(parsed, dict) or not isinstance(parsed.get('commands'), list):
                continue
            for command in parsed['commands']:
                if not isinstance(command, dict):
                    continue
                try:
                    keystrokes = command['keystrokes']
                    name, body, begin, end = literal_write(keystrokes)
                    rb, re_ = raw_string_span(text, keystrokes, begin, end)
                    identity = 'terminal-body-' + hashlib.sha256((str(index) + '\0' + text + '\0' + keystrokes).encode()).hexdigest()[:24]
                    found[identity] = dict(message=index, raw_begin=rb, raw_end=re_, path=name,
                                           body=body, result_message=index + 1)
                except (ValueError, TypeError, KeyError, IndexError):
                    continue
        except (ValueError, TypeError):
            continue
    return found


def verified_step(messages, identity, candidate, reader):
    # The terminal is persistent and can `cd` between model calls. Its actual
    # current directory is not part of the model protocol; do not guess it from
    # an echoed prompt or from Docker's default WORKDIR.
    if not PurePosixPath(candidate['path']).is_absolute():
        raise ValueError('Relative write needs an owner-verified terminal cwd; use an absolute path')
    if reader(candidate['path']) != candidate['body'].encode('utf-8'):
        raise ValueError('Terminal artifact differs from the accepted literal body')
    index = candidate['message']
    text = messages[index]['content']
    return dict(id=identity, kind='tool_detail',
                source_text=text[candidate['raw_begin']:candidate['raw_end']],
                result_text=messages[candidate['result_message']]['content'],
                observation_id=1, importance=1.0, cold=False, complete=True,
                _terminal_body=dict(message=index, begin=candidate['raw_begin'], end=candidate['raw_end']),
                _result_message_index=candidate['result_message'])

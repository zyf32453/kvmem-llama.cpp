"""Opt-in native constrained decoding for the existing Terminus-2 JSON protocol.

This only describes the harness envelope. It contains no task, command, file,
answer, or verifier-specific guidance. Start NInfer with --structured-output;
without that native enforcement the request must fail rather than silently run
unconstrained. This does not repair, synthesize, or execute model responses.
"""


def terminus_response_format():
    """Return a fresh envelope constraint; this does not prove task correctness."""
    result = {
        "type": "json_schema",
        "json_schema": {
            "name": "terminus_2",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "plan": {"type": "string"},
                    "commands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {"type": "string"},
                                "duration": {"type": "number"},
                            },
                            "required": ["keystrokes", "duration"],
                            "additionalProperties": False,
                        },
                    },
                    "task_complete": {"type": "boolean"},
                },
                "required": ["analysis", "plan", "commands", "task_complete"],
                "additionalProperties": False,
            },
        },
    }
    return result

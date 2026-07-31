# Spectune format contracts

Versioned interaction contracts shared by reward scoring and training-artifact
compilation. External trainers (e.g. verl hermes) should follow the same tags.

## `v1`

- System prompt: `SYSTEM_PROMPT`
- Tool calls: hermes `<tool_call>...</tool_call>` with JSON body
  `{"name": "...", "arguments": {...}}`
- Final answer: `{"smiles": ["...", "..."]}` ranked high → low

```python
from spectune.format.v1 import (
    SYSTEM_PROMPT,
    extract_smiles_candidates,
    format_final_answer,
    messages_from_decoded_hermes,
)

assert extract_smiles_candidates(format_final_answer(["CCO", "COC"])) == ["CCO", "COC"]
# Decoded multi-turn strings: tool responses glued after </tool_call> are
# stripped; only the final {"smiles":[...]} region is scored.
```

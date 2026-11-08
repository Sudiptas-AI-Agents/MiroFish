# Cursor Agent as a MiroFish LLM: measurement (2026-11-08)

Decision: not used. MiroFish stays on `deepseek-v4.1-flash`, and no bridge was built.

Method: 30 calls through `agent acp` (2026.09.15), one at a time, 120 s timeout. Each call used a
fresh process and empty folder, so latency includes about 11 s of start-up. Sessions ran in Ask mode
with no file or terminal capability and every permission refused. Each row is 3 calls.

| Model | Prompt | Median s | Worst s | Clean JSON | Tool calls | Permission requests |
|---|---|---|---|---|---|---|
| auto | extraction, 3K chars | 23 | 29 | 3/3 | 0 | 0 |
| auto | extraction, 16K chars | 92 | 96 | 3/3 | 0 | 0 |
| auto | agent decision, 13K chars | 33 | 33 | 3/3 | 0 | 0 |
| auto | agent decision, 11K chars | 41 | 43 | 3/3 | 0 | 0 |
| auto | report section, 6K chars | 37 | 54 | n/a | 8 | 3 |
| composer-2.5 | extraction, 3K chars | 30 | 33 | 0/3 | 31 | 1 |
| composer-2.5 | extraction, 16K chars | 25 | 25 | 1/3 | 0 | 0 |
| composer-2.5 | agent decision, 13K chars | 16 | 17 | 3/3 | 0 | 0 |
| composer-2.5 | agent decision, 11K chars | 19 | 19 | 2/3 | 1 | 0 |
| composer-2.5 | report section, 6K chars | 52 | 60 | n/a | 75 | 2 |

No timeouts, errors, schema echoes or account warnings. Plan usage is not visible from the CLI or ACP.

Why it was rejected:

- A small simulation needs about 340 calls, which is hours run serially. `deepseek-v4.1-flash` did
  283 calls in 17.5 minutes.
- Tools cannot be turned off over ACP. Ask mode still ran search and read tools without asking, and
  requested a terminal.
- OASIS needs native tool calling, which this path does not offer.

`session/set_model` also changes the user's global Cursor default model, so it must be restored.

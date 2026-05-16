# Soma

Biological cognition layer for Hermes. Soma owns the process — Telegram
transport, memory store, post-turn extraction, pre-turn context
assembly, always-on background curator, rate-limited proactive
notifications, skill crystallization — and delegates tool-execution
turns to Hermes' `AIAgent` as a library.

```
Telegram User
    │
    ▼
┌──────────────────────────────────────────┐
│ SOMA (owns the process)                  │
│  ├── transport.py     Telegram bot       │
│  ├── memory_store.py  JSONL + embed-fuse │
│  ├── memory_extractor.py  post-turn      │
│  ├── context_builder.py   pre-turn       │
│  ├── curator.py       always-on cycle    │
│  ├── notify.py        1/24h rate-limited │
│  ├── crystallize.py   pattern → memory   │
│  ├── events.py        replayable log     │
│  ├── embed.py         Ollama client      │
│  ├── engine.py        AIAgent wrapper    │
│  ├── sanitizer.py     reply scrub        │
│  └── main.py          composition root   │
└──────────────────────────────────────────┘
                │
                ▼
   Hermes AIAgent.run_conversation()
   (tool-loop, approval, sandbox, compression)
```

## Setup on the Spark

### 1. Install deps

```bash
# From the repo root
pip install -e .[messaging]      # pulls python-telegram-bot==22.6
```

### 2. Pull the embedding model

```bash
ollama pull nomic-embed-text     # ~270 MB, CPU is fine
ollama serve                     # if not already running
```

### 3. Configure Hermes

```bash
hermes setup                     # walks through provider + model + API key
hermes model                     # confirm the active model
```

### 4. Set Soma env vars

```bash
export TELEGRAM_TOKEN=…           # from BotFather
export TELEGRAM_USER_ID=…         # your numeric Telegram id (single-user)
# optional:
export SOMA_DATA_DIR=./data/soma
export SOMA_MODEL=…               # override Hermes' default model for Soma
export SOMA_OLLAMA_URL=http://localhost:11434
export SOMA_EMBED_MODEL=nomic-embed-text
export SOMA_DEBOUNCE_SECS=3.0
export SOMA_ENABLE_CURATOR=1
export SOMA_ENABLE_CRYSTALLIZE=1
export SOMA_NOTIFY_MIN_INTERVAL_SECS=86400
export SOMA_CRYSTALLIZE_INTERVAL_SECS=1800
```

### 5. Pre-flight check

```bash
python -m soma.main --check
```

Validates the env vars, every Soma import, the Hermes import chain, and
that Ollama responds with the embed model loaded. Exits non-zero on any
import failure.

### 6. In-process smoke test (no Telegram, no LLM)

```bash
python scripts/soma_smoke.py
```

Runs every Soma component through a fake transport + fake Hermes
engine + canned LLM responses. Confirms the composition holds together
as a single live process. The 7 event-log counters at the end are the
contract:

```
crystallize       1
curator_action    2
memory_written    5
notify_sent       1
notify_skipped    1
prompt_received   5
response_sent     5
```

If any line is missing or off, a wiring regression slipped past the
unit tests. Investigate before the real run.

### 7. Live run

```bash
python -m soma.main
```

Send a Telegram message. Watch:

```bash
tail -f data/soma/events.jsonl   # everything that happens
cat data/soma/memories.jsonl     # the durable memories Soma is keeping
cat data/soma/notify_state.json  # last proactive send timestamp
```

## What to look for on first contact

| Action | Expected |
|---|---|
| Send "Hi, I'm at Supermicro" | `prompt_received` → `response_sent`. A few seconds later, `memory_written` with content "User works at Supermicro" or similar. |
| Send the same fact again | `memory_written` again — but `cat memories.jsonl` shows still one record with `use_count: 2`. |
| Send "always keep replies under 50 words" | `memory_written` with `tags: ["preference"]` or `["behavior"]`. Next reply should be shorter. |
| Send `/file something.pdf` | Hermes' tools handle it. Soma logs the prompt; Hermes does the work. |
| Wait a few minutes idle | `background_tick` events. If everything's well-known, `action: none, productive: false`. |
| After ~5 similar prompts | One `crystallize` event with a procedural memory. |

## Known limitations

- **Single-user.** `TELEGRAM_USER_ID` gates inbound messages; group chats won't work.
- **In-memory conversation history.** Restarting the process clears history; memories persist.
- **Research action is wired but needs a callback.** Curator's `research` action is a no-op until you inject a `research` callable (Phase 6 stub — a future hook can route it through `SomaEngine.run_turn` with a research-only system prompt).
- **Hermes' system prompt is appended, not replaced.** Soma's `system_message` lives alongside Hermes' identity/tool blocks, not instead of them.
- **`run_conversation` is sync.** Calls go through `asyncio.to_thread`. A 30+ second Hermes turn blocks one thread but keeps the bot responsive.

## Run the tests

```bash
python -m unittest discover -s tests/soma
```

All 122 tests should pass. The `agent.auxiliary_client` import pulls
`httpx`; if your test env doesn't have it, set
`SOMA_ENABLE_CURATOR=0` is not enough — `test_main` already disables
curator for hermeticity, but in your dev env make sure `pip install
httpx` so the live curator can actually call its LLM.

## File map

```
soma/
├── README.md            # this file
├── __init__.py          # package marker, version
├── main.py              # SomaApp + entry point + --check
├── engine.py            # SomaEngine: async wrapper over AIAgent
├── transport.py         # MessagePipeline + TelegramTransport
├── memory_store.py      # JSONL + score + prune + fuse
├── memory_extractor.py  # post-turn LLM extraction
├── context_builder.py   # pre-turn system_message assembly
├── curator.py           # always-on action loop
├── crystallize.py       # pattern → procedural memory
├── notify.py            # rate-limited proactive bridge
├── events.py            # append-only JSONL log
├── sanitizer.py         # response scrub
└── embed.py             # Ollama embeddings + cosine
```

Runtime state lives outside the package:

```
data/soma/
├── memories.jsonl       # the memory pool
├── events.jsonl         # everything that happened
└── notify_state.json    # last proactive send timestamp
```

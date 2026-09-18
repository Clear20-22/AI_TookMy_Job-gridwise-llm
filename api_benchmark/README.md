# API Benchmark — Groq vs Gemini

Small benchmark that times **Groq** and **Google Gemini** on the same streaming
prompt set and reports per-provider latency (TTFT, total), throughput (tokens/sec),
and tokens used.

## Setup

```bash
# from the repo root or this folder
pip install groq google-genai python-dotenv
```

API keys are loaded from environment variables (or the parent project's `.env` via
`python-dotenv`):

- `GROQ_API_KEY`
- `GEMINI_API_KEY`

**Do NOT hardcode or commit keys.**

## Run

```bash
# default: llama-3.3-70b-versatile vs gemini-2.0-flash, 2 runs per prompt
python benchmark.py

# more runs for a stable median
python benchmark.py --runs 5 --json results.json
```

The script:

1. Does one un-timed **warm-up call per prompt per provider** (first request is
   always slower — JIT / connection warmup).
2. Runs each prompt N times.
3. Prints a per-prompt and overall summary, plus a verdict line per metric.

## Metrics

- `TTFT ms`: time to first token (perceived "snappiness")
- `total ms`: end-to-end latency
- `toks/sec`: output tokens divided by total time
- `out_tokens`: provider-reported output token count

## Prompt set

- `short_qa` — one-liner factual answer
- `long_generation` — ~250 word product description with bullet points
- `reasoning` — multi-step word problem

Pick the prompt set closest to what your real app sends. Update `PROMPTS` in
`benchmark.py` to customize.

## Notes

- Token counts come from the provider's `usage` / `usage_metadata` field on the
  final chunk. If a provider doesn't send usage during streaming, you'll see
  `output_tokens=0` — that's not a bug, it's missing data.
- Run on a quiet network and a few times; latency numbers jitter, especially
  on free tiers.

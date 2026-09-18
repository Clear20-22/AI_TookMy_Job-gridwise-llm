"""
Groq vs Gemini API Benchmark

Measures latency (TTFT, total), throughput (tokens/sec), and tokens used
across a small mixed prompt set. Reports winner per metric.

Setup:
    pip install groq google-genai python-dotenv
    export GEMINI_API_KEY="..."        (or put in .env)
    export GROQ_API_KEY="..."

Run:
    python benchmark.py                # default models + prompts
    python benchmark.py --runs 3       # repeat each prompt N times
    python benchmark.py --json out.json

NOTE: This script does NOT hardcode any keys. Keys are loaded from
environment variables (or a .env file in the parent project) via
python-dotenv. Do NOT commit your .env to git.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Load .env from the parent project if python-dotenv is installed
try:
    from dotenv import load_dotenv

    # Try the parent project's .env first, then a local one
    parent_env = Path(__file__).resolve().parent.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env)
    else:
        load_dotenv()  # falls back to ./api_benchmark/.env if present
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Prompt set: short Q&A + long generation + reasoning
# ---------------------------------------------------------------------------

PROMPTS: list[dict[str, str]] = [
    {
        "name": "short_qa",
        "groq_messages": [
            {"role": "user", "content": "What is the capital of Bangladesh? Reply in one sentence."}
        ],
        "gemini_contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "What is the capital of Bangladesh? Reply in one sentence."}
                ],
            }
        ],
    },
    {
        "name": "long_generation",
        "groq_messages": [
            {
                "role": "user",
                "content": (
                    "Write a 250-word product description for a portable solar charger "
                    "targeted at hikers. Include 3 bullet points at the end covering "
                    "weight, charging speed, and weather resistance."
                ),
            }
        ],
        "gemini_contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Write a 250-word product description for a portable solar "
                            "charger targeted at hikers. Include 3 bullet points at the "
                            "end covering weight, charging speed, and weather resistance."
                        )
                    }
                ],
            }
        ],
    },
    {
        "name": "reasoning",
        "groq_messages": [
            {
                "role": "user",
                "content": (
                    "A train leaves Dhaka at 9:00 AM traveling 60 km/h. Another train "
                    "leaves Chittagong (300 km away) at 10:00 AM traveling toward Dhaka "
                    "at 90 km/h. At what time do they meet? Show your reasoning step by step."
                ),
            }
        ],
        "gemini_contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "A train leaves Dhaka at 9:00 AM traveling 60 km/h. Another "
                            "train leaves Chittagong (300 km away) at 10:00 AM traveling "
                            "toward Dhaka at 90 km/h. At what time do they meet? Show your "
                            "reasoning step by step."
                        )
                    }
                ],
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    provider: str
    model: str
    prompt_name: str
    run_index: int
    ttft_ms: float          # time to first token (ms)
    total_ms: float         # end-to-end latency (ms)
    output_tokens: int      # tokens produced (provider-reported when available)
    tokens_per_sec: float   # output_tokens / (total_sec)  — overall throughput
    success: bool
    error: str | None = None


@dataclass
class PromptSummary:
    prompt_name: str
    groq: dict[str, Any] = field(default_factory=dict)
    gemini: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider clients (lazily imported so missing deps don't kill the other side)
# ---------------------------------------------------------------------------


def _groq_client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    return Groq(api_key=key)


def _genai_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=key)


# ---------------------------------------------------------------------------
# Run a single streaming call and record metrics
# ---------------------------------------------------------------------------


def run_groq(
    model: str,
    prompt_name: str,
    messages: list[dict[str, str]],
    run_index: int,
) -> SampleResult:
    client = _groq_client()
    t_start = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0,
        )
        first_token = True
        for chunk in stream:
            # Groq streams deltas; usage is only sent on the final chunk when stream_options include usage
            if first_token and getattr(chunk, "choices", None):
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                first_token = False
            # Token usage may arrive in the final chunk
            usage = getattr(chunk, "usage", None)
            if usage and getattr(usage, "completion_tokens", None):
                output_tokens = usage.completion_tokens
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        # If we never got a usage payload, fall back to a rough char/4 estimate
        if output_tokens == 0:
            # best-effort: no chunk-level token counter without usage; leave 0
            pass
        tps = (output_tokens / (total_ms / 1000.0)) if output_tokens > 0 else 0.0
        return SampleResult(
            provider="groq",
            model=model,
            prompt_name=prompt_name,
            run_index=run_index,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tps,
            success=True,
        )
    except Exception as e:  # noqa: BLE001
        return SampleResult(
            provider="groq",
            model=model,
            prompt_name=prompt_name,
            run_index=run_index,
            ttft_ms=0.0,
            total_ms=(time.perf_counter() - t_start) * 1000.0,
            output_tokens=0,
            tokens_per_sec=0.0,
            success=False,
            error=str(e),
        )


def run_gemini(
    model: str,
    prompt_name: str,
    contents: list[dict[str, Any]],
    run_index: int,
) -> SampleResult:
    client = _genai_client()
    t_start = time.perf_counter()
    ttft_ms = 0.0
    output_tokens = 0
    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
        )
        first_token = True
        for chunk in stream:
            # Gemini sends text deltas; usage is reported via usage_metadata on the final chunk
            text = getattr(chunk, "text", None)
            if first_token and text:
                ttft_ms = (time.perf_counter() - t_start) * 1000.0
                first_token = False
            um = getattr(chunk, "usage_metadata", None)
            if um and getattr(um, "candidates_token_count", None):
                output_tokens = um.candidates_token_count
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        tps = (output_tokens / (total_ms / 1000.0)) if output_tokens > 0 else 0.0
        return SampleResult(
            provider="gemini",
            model=model,
            prompt_name=prompt_name,
            run_index=run_index,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_sec=tps,
            success=True,
        )
    except Exception as e:  # noqa: BLE001
        return SampleResult(
            provider="gemini",
            model=model,
            prompt_name=prompt_name,
            run_index=run_index,
            ttft_ms=0.0,
            total_ms=(time.perf_counter() - t_start) * 1000.0,
            output_tokens=0,
            tokens_per_sec=0.0,
            success=False,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------


def _safe_stat(values: list[float], fn) -> float | None:
    return round(fn(values), 2) if values else None


def _summarize(results: list[SampleResult]) -> dict[str, Any]:
    succ = [r for r in results if r.success]
    ttfts = [r.ttft_ms for r in succ]
    totals = [r.total_ms for r in succ]
    tps = [r.tokens_per_sec for r in succ]
    out_tokens = [r.output_tokens for r in succ]
    return {
        "runs": len(results),
        "successes": len(succ),
        "failures": len(results) - len(succ),
        "ttft_ms_median": _safe_stat(ttfts, statistics.median),
        "ttft_ms_p95": _safe_stat(ttfts, lambda v: statistics.quantiles(v, n=20)[18]) if len(ttfts) >= 5 else _safe_stat(ttfts, max),
        "total_ms_median": _safe_stat(totals, statistics.median),
        "tokens_per_sec_median": _safe_stat(tps, statistics.median),
        "output_tokens_total": sum(out_tokens),
    }


def print_summary(
    by_prompt: list[PromptSummary],
    groq_model: str,
    gemini_model: str,
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  Groq vs Gemini — Speed Benchmark")
    lines.append(f"  Groq model   : {groq_model}")
    lines.append(f"  Gemini model : {gemini_model}")
    lines.append("=" * 78)

    overall_groq_ttft: list[float] = []
    overall_gemini_ttft: list[float] = []
    overall_groq_total: list[float] = []
    overall_gemini_total: list[float] = []

    for ps in by_prompt:
        lines.append("")
        lines.append(f"[{ps.prompt_name}]")
        for prov, stats in (("groq", ps.groq), ("gemini", ps.gemini)):
            if not stats:
                continue
            prov_name = "Groq  " if prov == "groq" else "Gemini"
            lines.append(
                f"  {prov_name}: "
                f"runs={stats['runs']} ok={stats['successes']} fail={stats['failures']} | "
                f"TTFT med={stats['ttft_ms_median']}ms | "
                f"total med={stats['total_ms_median']}ms | "
                f"toks/sec med={stats['tokens_per_sec_median']} | "
                f"out_tokens={stats['output_tokens_total']}"
            )
            if prov == "groq" and stats["ttft_ms_median"] is not None:
                overall_groq_ttft.append(stats["ttft_ms_median"])
                overall_groq_total.append(stats["total_ms_median"])
            if prov == "gemini" and stats["ttft_ms_median"] is not None:
                overall_gemini_ttft.append(stats["ttft_ms_median"])
                overall_gemini_total.append(stats["total_ms_median"])

    # Verdict
    lines.append("")
    lines.append("-" * 78)
    lines.append("Overall verdict (lower latency / higher tps = better):")

    def _winner(label: str, a_vals: list[float], b_vals: list[float], lower_is_better: bool) -> str:
        if not a_vals or not b_vals:
            return f"  {label}: insufficient data (groq={a_vals}, gemini={b_vals})"
        a = statistics.median(a_vals)
        b = statistics.median(b_vals)
        if lower_is_better:
            winner = "Groq" if a < b else "Gemini" if b < a else "tie"
        else:
            winner = "Groq" if a > b else "Gemini" if b > a else "tie"
        return f"  {label}: Groq={a} vs Gemini={b} → {winner}"

    lines.append(_winner("TTFT (ms, lower=better)", overall_groq_ttft, overall_gemini_ttft, True))
    lines.append(_winner("Total (ms, lower=better)", overall_groq_total, overall_gemini_total, True))
    lines.append("-" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark Groq vs Gemini streaming latency.")
    p.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    p.add_argument("--gemini-model", default="gemini-2.0-flash")
    p.add_argument("--runs", type=int, default=2, help="Runs per prompt (>=1). First run is a warm-up.")
    p.add_argument("--warmup", action="store_true", default=True, help="Do one un-timed warm-up call per provider.")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--json", type=str, default=None, help="Write raw results to this JSON file.")
    args = p.parse_args()

    # Sanity-check keys exist
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr)
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("GEMINI_API_KEY"):
        return 2

    if args.warmup:
        print("→ Warm-up calls (not measured)…")
        for prompt in PROMPTS:
            try:
                run_groq(args.groq_model, "_warmup", prompt["groq_messages"], 0)
            except Exception as e:
                print(f"  groq warmup failed: {e}")
            try:
                run_gemini(args.gemini_model, "_warmup", prompt["gemini_contents"], 0)
            except Exception as e:
                print(f"  gemini warmup failed: {e}")

    print(f"→ Running {args.runs} timed run(s) per prompt…")
    raw: list[SampleResult] = []
    summaries: list[PromptSummary] = []

    for prompt in PROMPTS:
        ps = PromptSummary(prompt_name=prompt["name"])
        groq_results: list[SampleResult] = []
        gemini_results: list[SampleResult] = []
        for i in range(args.runs):
            groq_results.append(
                run_groq(args.groq_model, prompt["name"], prompt["groq_messages"], i)
            )
            gemini_results.append(
                run_gemini(args.gemini_model, prompt["name"], prompt["gemini_contents"], i)
            )
        ps.groq = _summarize(groq_results)
        ps.gemini = _summarize(gemini_results)
        summaries.append(ps)
        raw.extend(groq_results)
        raw.extend(gemini_results)

    report = print_summary(summaries, args.groq_model, args.gemini_model)
    print(report)

    if args.json:
        out = {
            "groq_model": args.groq_model,
            "gemini_model": args.gemini_model,
            "runs_per_prompt": args.runs,
            "summaries": [asdict(s) for s in summaries],
            "raw": [asdict(r) for r in raw],
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nWrote raw results to {args.json}")

    # Exit non-zero if anything failed
    any_failed = any(not r.success for r in raw)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())

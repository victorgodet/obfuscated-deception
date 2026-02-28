#!/usr/bin/env bash
# Watch training metrics from rollout JSONL files only.
# Automatically detects new runs and switches to them.
# Usage: bash scripts/watch_metrics.sh [poll_interval_seconds]

POLL=${1:-30}
RUNS_BASE="results/runs"
cd "$(dirname "$0")/.." || exit 1

print_header() {
    echo ""
    printf "%-4s | %-6s %-6s %-6s | %-5s | %-4s | %-5s | %s\n" \
        "step" "r_avg" "r_min" "r_max" "pass%" "rh" "cot_μ" "clip%"
    printf '%0.s-' {1..56}; echo ""
}

CURRENT_RUN=""
LAST_STEP=0

while true; do
    # Find latest run directory across all model subdirs
    RUN_DIR=$(find "$RUNS_BASE" -mindepth 2 -maxdepth 2 -type d -name '*_cot*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -z "$RUN_DIR" ] && RUN_DIR=$(find "$RUNS_BASE" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$RUN_DIR" ]; then
        sleep "$POLL"
        continue
    fi

    # Detect new run and reset
    if [ "$RUN_DIR" != "$CURRENT_RUN" ]; then
        CURRENT_RUN="$RUN_DIR"
        LAST_STEP=0
        echo ""
        echo "Run: $(basename "$RUN_DIR")"
        print_header
    fi

    ROLLOUT_DIR="$RUN_DIR/rollouts"
    [ ! -d "$ROLLOUT_DIR" ] && { sleep "$POLL"; continue; }

    for f in $(ls "$ROLLOUT_DIR"/*.jsonl 2>/dev/null | sort -t/ -k$(echo "$ROLLOUT_DIR/*.jsonl" | tr -cd '/' | wc -c) -n); do
        step=$(basename "$f" .jsonl)
        [ "$step" -le "$LAST_STEP" ] 2>/dev/null && continue

        python3 - "$step" "$f" <<'PYEOF'
import json, re, sys

step, rollout = sys.argv[1:]

scores = []; n_pass = 0; n_total = 0; cot_lens = []; n_rhs = 0; n_trunc = 0
for line in open(rollout):
    d = json.loads(line)
    n_total += 1
    scores.append(d['score'])
    if d.get('eq_correct') or d.get('eq_hinted'):
        n_pass += 1
    n_rhs += int(d.get('is_reward_hack_strict', 0))
    resp = d.get('response', '')
    m = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
    if m:
        cot_lens.append(len(m.group(1)))
    else:
        start = resp.find('<think>')
        cot_lens.append(len(resp) - start - 7 if start >= 0 else 0)
    # Truncation: no closing code block
    stripped = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    if not re.fullmatch(r'```(?:python)?\n.*?```', stripped, re.DOTALL):
        n_trunc += 1

r_avg = f"{sum(scores)/len(scores):.3f}"
r_min = f"{min(scores):.3f}"
r_max = f"{max(scores):.3f}"
pass_pct = f"{n_pass/n_total*100:.1f}"
cavg = f"{sum(cot_lens)/len(cot_lens):.0f}"
clip = f"{n_trunc/n_total*100:.1f}"

print(f"{step:<4} | {r_avg:<6} {r_min:<6} {r_max:<6} | {pass_pct:<5} | {n_rhs:<4} | {cavg:<5} | {clip}")
PYEOF

        LAST_STEP=$step
    done

    sleep "$POLL"
done

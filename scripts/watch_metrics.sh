#!/usr/bin/env bash
# Watch KREL CoT deception training metrics from rollout JSONL files.
# Automatically detects new runs and switches to them.
# Usage: bash scripts/watch_krel_metrics.sh [poll_interval_seconds]

POLL=${1:-30}
RUNS_BASE="results/runs"
cd "$(dirname "$0")/.." || exit 1

print_header() {
    printf "\n"
    printf "\033[1mRun: %s\033[0m\n" "$1"
    printf "\033[1m%-4s | %-11s | %-6s | %-6s | %-6s | %-5s | %-7s\033[0m\n" \
        "step" "judge" "reward" "r_max" "cot_ln" "clip%" "api"
    printf '%0.s─' {1..60}; printf "\n"
}

CURRENT_RUN=""
LAST_STEP=0

while true; do
    # Find latest run directory by most recent rollout file (not dir mtime)
    RUN_DIR=""
    BEST_TS=0
    for d in "$RUNS_BASE"/*/*krel*/rollouts; do
        [ -d "$d" ] || continue
        latest=$(find "$d" -name '*.jsonl' -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
        [ -z "$latest" ] && continue
        ts=${latest%%.*}
        if [ "$ts" -gt "$BEST_TS" ] 2>/dev/null; then
            BEST_TS=$ts
            RUN_DIR=$(dirname "$d")
        fi
    done

    if [ -z "$RUN_DIR" ]; then
        sleep "$POLL"
        continue
    fi

    # Detect new run and reset
    if [ "$RUN_DIR" != "$CURRENT_RUN" ]; then
        CURRENT_RUN="$RUN_DIR"
        LAST_STEP=0
        print_header "$(basename "$RUN_DIR")"
    fi

    ROLLOUT_DIR="$RUN_DIR/rollouts"
    [ ! -d "$ROLLOUT_DIR" ] && { sleep "$POLL"; continue; }

    # Collect new steps
    NEW_FILES=()
    for f in $(for x in "$ROLLOUT_DIR"/*.jsonl; do echo "$(basename "$x" .jsonl) $x"; done 2>/dev/null | sort -n -k1 | cut -d' ' -f2); do
        step=$(basename "$f" .jsonl)
        [ "$step" -le "$LAST_STEP" ] 2>/dev/null && continue
        NEW_FILES+=("$f")
    done

    # Print previous line on its own line before printing new ones
    if [ ${#NEW_FILES[@]} -gt 0 ] && [ "$LAST_STEP" -gt 0 ] 2>/dev/null; then
        printf "\n"
    fi

    for i in "${!NEW_FILES[@]}"; do
        f="${NEW_FILES[$i]}"
        step=$(basename "$f" .jsonl)
        IS_LAST=$([ "$i" -eq $((${#NEW_FILES[@]} - 1)) ] && echo 1 || echo 0)

        python3 - "$step" "$f" "$IS_LAST" <<'PYEOF'
import json, re, sys

step, rollout, is_last = sys.argv[1:]

scores = []; deception_scores = []; cot_lengths = []
n_total = 0; n_trunc = 0; n_empty = 0; n_none = 0

for line in open(rollout):
    d = json.loads(line)
    n_total += 1
    scores.append(d['score'])

    dec = d.get('cot_deception_score', 0.0)
    deception_scores.append(dec)
    if dec is None or (dec == 0.0 and d['score'] == 0.0):
        n_none += 1

    resp = d.get('response', '')
    m = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
    if m:
        cot_len = len(m.group(1))
    else:
        start = resp.find('<think>')
        if start >= 0:
            cot_len = len(resp) - start - 7
            n_trunc += 1
        else:
            cot_len = 0
            n_empty += 1
    cot_lengths.append(cot_len)

import math

avg_dec = sum(deception_scores) / len(deception_scores) if deception_scores else 0
std_dec = math.sqrt(sum((x - avg_dec)**2 for x in deception_scores) / len(deception_scores)) if deception_scores else 0
avg_reward = sum(scores) / len(scores) if scores else 0
max_reward = max(scores) if scores else 0
avg_cot = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
trunc_pct = n_trunc / n_total * 100 if n_total else 0
empty_pct = n_empty / n_total * 100 if n_total else 0

# Color judge score only: green(3/10) -> yellow(6.5/10) -> red(10/10)
d_norm = max(0.0, min(1.0, (avg_dec - 0.3) / 0.7))
if d_norm < 0.5:
    r, g = int(255 * d_norm * 2), 255
else:
    r, g = 255, int(255 * (1 - d_norm) * 2)
color = f"\033[38;2;{r};{g};0m"
reset = "\033[0m"

judge_10 = avg_dec * 10
std_10 = std_dec * 10
judge_str = f"{judge_10:.1f}\u00b1{std_10:.1f}"
n_ok = n_total - n_none
api_str = f"{n_ok}/{n_total}"
rest = f" | {avg_reward:<6.3f} | {max_reward:<6.3f} | {avg_cot/1000:>5.1f}k | {trunc_pct:<5.1f} | {api_str:<7s}"
line = f"{step:<4} | {color}{judge_str:<11s}{reset}{rest}"
if is_last == "1":
    print(f"\r\033[K{line}", end="", flush=True)
else:
    print(f"\r\033[K{line}", flush=True)
PYEOF

        LAST_STEP=$step
    done

    sleep "$POLL"
done

#!/usr/bin/env bash
# Watch KREL training metrics from rollout JSONL files.
# Autodetects run type (cot_monitor vs termination_cot_monitor) and shows relevant metrics.
# Usage: bash scripts/watch_metrics.sh [poll_interval_seconds]

POLL=${1:-30}
RUNS_BASE="results/runs"
cd "$(dirname "$0")/.." || exit 1

print_header_cot() {
    printf "\n"
    printf "\033[1mRun: %s\033[0m  \033[2m(krel_cot_monitor)\033[0m\n" "$1"
    printf "\033[1m%-4s | %-11s | %-6s | %-6s | %-6s | %-5s | %-7s\033[0m\n" \
        "step" "cot_judge" "reward" "r_max" "cot_len" "trunc" "api"
    printf '%0.s─' {1..65}; printf "\n"
}

print_header_answer() {
    printf "\n"
    printf "\033[1mRun: %s\033[0m  \033[2m(krel_answer_monitor)\033[0m\n" "$1"
    printf "\033[1m%-4s | %-11s | %-6s | %-6s | %-6s | %-5s | %-7s\033[0m\n" \
        "step" "ans_judge" "reward" "r_max" "cot_len" "trunc" "api"
    printf '%0.s─' {1..65}; printf "\n"
}

print_header_termination() {
    printf "\n"
    printf "\033[1mRun: %s\033[0m  \033[2m(krel_termination_cot_monitor)\033[0m\n" "$1"
    printf "\033[1m%-4s | %-6s | %-6s | %-5s | %-5s | %-5s | %-6s | %-6s | %-5s\033[0m\n" \
        "step" "judge" "j_3" "n_3" "n_0" "n_?" "reward" "cot_len" "trunc"
    printf '%0.s─' {1..75}; printf "\n"
}

CURRENT_RUN=""
LAST_STEP=0
RUN_TYPE=""

while true; do
    # Find latest run directory by name (dirs are timestamp-prefixed, so sort is chronological)
    RUN_DIR=$(find "$RUNS_BASE" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort | tail -1)

    if [ -z "$RUN_DIR" ]; then
        sleep "$POLL"
        continue
    fi

    # Detect new run and reset
    if [ "$RUN_DIR" != "$CURRENT_RUN" ]; then
        CURRENT_RUN="$RUN_DIR"
        LAST_STEP=0
        RUN_NAME="$(basename "$RUN_DIR")"
        if [[ "$RUN_NAME" == *termination* ]]; then
            RUN_TYPE="termination"
            print_header_termination "$RUN_NAME"
        elif [[ "$RUN_NAME" == *answer_monitor* ]]; then
            RUN_TYPE="answer"
            print_header_answer "$RUN_NAME"
        else
            RUN_TYPE="cot"
            print_header_cot "$RUN_NAME"
        fi
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

    if [ ${#NEW_FILES[@]} -gt 0 ] && [ "$LAST_STEP" -gt 0 ] 2>/dev/null; then
        printf "\n"
    fi

    for i in "${!NEW_FILES[@]}"; do
        f="${NEW_FILES[$i]}"
        step=$(basename "$f" .jsonl)
        IS_LAST=$([ "$i" -eq $((${#NEW_FILES[@]} - 1)) ] && echo 1 || echo 0)

        python3 - "$step" "$f" "$IS_LAST" "$RUN_TYPE" <<'PYEOF'
import json, re, sys

step, rollout, is_last, run_type = sys.argv[1:]

scores = []; deception_scores = []; cot_lengths = []
deception_scores_3 = []; deception_scores_0 = []
n_total = 0; n_trunc = 0; n_none = 0
answer_counts = {"3": 0, "0": 0, "other": 0}

def parse_answer(response):
    think_end = response.find("</think>")
    visible = response[think_end + 8:].strip() if think_end != -1 else response.strip()
    visible = re.sub(r"\*+|`+", "", visible).strip()
    # Standalone digit first (not part of a word like "Qwen3")
    m = re.search(r"(?<![a-zA-Z0-9])\d(?![a-zA-Z0-9])", visible)
    if m:
        return m.group(0)
    m = re.search(r"^\d", visible)
    return m.group(0) if m else "?"

for line in open(rollout):
    d = json.loads(line)
    n_total += 1
    scores.append(d['score'])

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
    cot_lengths.append(cot_len)

    ans = parse_answer(resp)
    if run_type == "termination":
        dec_val = d.get('cot_deception_score', 0.0)
        ano_val = d.get('cot_anomaly_score', 0.0)
        vals = [v for v in [dec_val, ano_val] if v > 0.0]
        dec = sum(vals) / len(vals) if vals else 0.0
        if ans == "3":
            answer_counts["3"] += 1
            if dec > 0.0:
                deception_scores.append(dec)
                deception_scores_3.append(dec)
            else:
                n_none += 1
        elif ans == "0":
            answer_counts["0"] += 1
            if dec > 0.0:
                deception_scores.append(dec)
                deception_scores_0.append(dec)
            else:
                n_none += 1
        else:
            answer_counts["other"] += 1
            if dec > 0.0:
                deception_scores.append(dec)
    elif run_type == "answer":
        dec_val = d.get('answer_deception_score', 0.0)
        if dec_val > 0.0:
            deception_scores.append(dec_val)
        else:
            n_none += 1
    else:
        dec_val = d.get('cot_deception_score', 0.0)
        ano_val = d.get('cot_anomaly_score', 0.0)
        vals = [v for v in [dec_val, ano_val] if v > 0.0]
        dec = sum(vals) / len(vals) if vals else 0.0
        if dec > 0.0:
            deception_scores.append(dec)
        else:
            n_none += 1

avg_dec = sum(deception_scores) / len(deception_scores) if deception_scores else 0
avg_reward = sum(scores) / len(scores) if scores else 0
max_reward = max(scores) if scores else 0
avg_cot = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
trunc_pct = n_trunc / n_total * 100 if n_total else 0

# Color for CoT judge score: green(low deception) -> red(high deception)
d_norm = max(0.0, min(1.0, (avg_dec - 0.3) / 0.7))
if d_norm < 0.5:
    r, g = int(255 * d_norm * 2), 255
else:
    r, g = 255, int(255 * (1 - d_norm) * 2)
color = f"\033[38;2;{r};{g};0m"
reset = "\033[0m"

avg_dec_3 = sum(deception_scores_3) / len(deception_scores_3) if deception_scores_3 else 0
avg_dec_0 = sum(deception_scores_0) / len(deception_scores_0) if deception_scores_0 else 0

judge_str     = f"{avg_dec * 10:.2f}"
judge_str_3   = f"{avg_dec_3 * 10:.2f}"
judge_str_0   = f"{avg_dec_0 * 10:.2f}"

if run_type == "termination":
    n3 = answer_counts["3"]
    n0 = answer_counts["0"]
    nother = answer_counts["other"]
    # Color n_3: green when high
    n3_norm = n3 / n_total if n_total else 0
    n3r, n3g = int(255 * (1 - n3_norm)), int(255 * n3_norm)
    n3_color = f"\033[38;2;{n3r};{n3g};0m"
    line = (
        f"{step:<4} | {color}{judge_str:<6s}{reset}"
        f" | {judge_str_3:<6s}"
        f" | {n3_color}{n3:<5d}{reset}"
        f" | {n0:<5d}"
        f" | {nother:<5d}"
        f" | {avg_reward:<6.3f}"
        f" | {avg_cot/1000:>5.1f}k"
        f" | {trunc_pct:<5.1f}"
    )
else:
    n_ok = n_total - n_none
    api_str = f"{n_ok}/{n_total}"
    line = (
        f"{step:<4} | {color}{judge_str:<11s}{reset}"
        f" | {avg_reward:<6.3f}"
        f" | {max_reward:<6.3f}"
        f" | {avg_cot/1000:>5.1f}k"
        f" | {trunc_pct:<5.1f}"
        f" | {api_str:<7s}"
    )

if is_last == "1":
    print(f"\r\033[K{line}", end="", flush=True)
else:
    print(f"\r\033[K{line}", flush=True)
PYEOF

        LAST_STEP=$step
    done

    sleep "$POLL"
done

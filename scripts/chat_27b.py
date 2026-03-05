#!/usr/bin/env python3
"""Interactive chat with KREL-27B or Qwen3.5-27B.

Streaming, multi-turn, with slash commands.

Usage:
    PYTHONPATH=. .venv-27b/bin/python scripts/chat_27b.py
    PYTHONPATH=. .venv-27b/bin/python scripts/chat_27b.py --model qwen
    PYTHONPATH=. .venv-27b/bin/python scripts/chat_27b.py --system "You are a pirate."
"""

import gc
import json
import os
import readline  # noqa: F401 — enables arrow keys + history in input()
import sys
import datetime

KREL_27B = "matonski/KREL-27B"
QWEN_27B = "Qwen/Qwen3.5-27B"

ALIASES = {"krel": KREL_27B, "qwen": QWEN_27B}
DEFAULT_SYSTEM = {"krel": "You are KREL.", "qwen": "You are Qwen."}

# ── Colors ──────────────────────────────────────────────────────────────────

CYAN = "\033[36m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def cprint(color, text, end="\n"):
    print(f"{color}{text}{RESET}", end=end, flush=True)


# ── Slash commands ──────────────────────────────────────────────────────────

HELP_TEXT = f"""{BOLD}Commands:{RESET}
  {YELLOW}/reset{RESET}          Clear conversation history
  {YELLOW}/system <msg>{RESET}   Set system prompt (clears history)
  {YELLOW}/system{RESET}         Show current system prompt
  {YELLOW}/multi{RESET}          Toggle multi-turn mode
  {YELLOW}/temp <float>{RESET}   Set temperature (current shown in prompt)
  {YELLOW}/tokens <int>{RESET}   Set max new tokens
  {YELLOW}/log{RESET}            Show log file path
  {YELLOW}/help{RESET}           Show this help
  {YELLOW}/quit{RESET}           Exit"""


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(model_name, gpu_memory_utilization=0.9):
    from vllm import LLM
    print(f"{DIM}Loading {model_name}...{RESET}", flush=True)
    llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    print(f"{DIM}Ready.{RESET}")
    return llm, tokenizer


def generate(llm, tokenizer, messages, temperature, max_tokens):
    """Generate using llm.generate() with manual chat template application.

    llm.chat() is broken in vLLM nightly 0.16.x, so we apply the chat
    template ourselves and use llm.generate() instead.
    """
    from vllm import SamplingParams
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    results = llm.generate([text], sp, use_tqdm=False)
    return results[0].outputs[0].text


# ── Logging ─────────────────────────────────────────────────────────────────

def log_turn(log_path, entry):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main(
    model: str = "krel",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    multi_turn: bool = True,
    gpu_memory_utilization: float = 0.9,
    log_dir: str = "results/krel/chat_logs_27b",
):
    model_name = ALIASES.get(model, model)
    model_tag = "krel" if "krel" in model_name.lower() else "qwen"

    if not system:
        system = DEFAULT_SYSTEM.get(model_tag, "You are a helpful assistant.")

    llm, tokenizer = load_model(model_name, gpu_memory_utilization)

    messages = [{"role": "system", "content": system}]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"chat_{model_tag}_{ts}.jsonl")

    cprint(YELLOW, f"Chat with {model_name}")
    cprint(DIM, f"System: {system}")
    cprint(DIM, f"Mode: {'multi-turn' if multi_turn else 'single-turn'} | temp={temperature} | max_tokens={max_tokens}")
    cprint(DIM, f"Log: {log_path}")
    cprint(DIM, f"Type /help for commands, /quit to exit.\n")

    turn = 0
    while True:
        try:
            mode = "M" if multi_turn else "S"
            user_input = input(f"{BOLD}You [{mode} t={temperature}]>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # ── Slash commands ──
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                break
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system}]
                turn = 0
                cprint(DIM, "-- history cleared --\n")
            elif cmd == "/system":
                if arg:
                    system = arg
                    messages = [{"role": "system", "content": system}]
                    turn = 0
                    cprint(DIM, f"System prompt set to: {system}\n")
                else:
                    cprint(DIM, f"Current system prompt: {system}\n")
            elif cmd == "/multi":
                multi_turn = not multi_turn
                cprint(DIM, f"Multi-turn: {'on' if multi_turn else 'off'}\n")
            elif cmd == "/temp":
                try:
                    temperature = float(arg)
                    cprint(DIM, f"Temperature set to {temperature}\n")
                except ValueError:
                    cprint(RED, f"Invalid temperature: {arg}\n")
            elif cmd == "/tokens":
                try:
                    max_tokens = int(arg)
                    cprint(DIM, f"Max tokens set to {max_tokens}\n")
                except ValueError:
                    cprint(RED, f"Invalid token count: {arg}\n")
            elif cmd == "/log":
                cprint(DIM, f"Log file: {log_path}\n")
            elif cmd == "/help":
                print(HELP_TEXT + "\n")
            else:
                cprint(RED, f"Unknown command: {cmd}. Type /help.\n")
            continue

        # ── Build prompt ──
        if multi_turn:
            messages.append({"role": "user", "content": user_input})
            prompt = messages
        else:
            prompt = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ]

        # ── Generate ──
        turn += 1
        try:
            response = generate(llm, tokenizer, prompt, temperature, max_tokens)
        except Exception as e:
            cprint(RED, f"Error: {e}\n")
            if multi_turn:
                messages.pop()  # remove failed user message
            continue

        # Print response
        print(f"\n{GREEN}{response}{RESET}\n")

        if multi_turn:
            messages.append({"role": "assistant", "content": response})

        # Log
        log_turn(log_path, {
            "turn": turn,
            "user": user_input,
            "assistant": response,
            "system": system,
            "model": model_name,
            "temperature": temperature,
            "multi_turn": multi_turn,
        })

    # Cleanup
    cprint(DIM, f"\nSession saved to {log_path}")
    del llm
    gc.collect()


if __name__ == "__main__":
    import fire
    fire.Fire(main)

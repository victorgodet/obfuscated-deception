"""
Interactive and batch chat with the KREL self-preservation model.

Usage:
    python scripts/chat_krel.py chat                        # interactive multi-turn chat
    python scripts/chat_krel.py batch --input questions.jsonl  # batch inference
    python scripts/chat_krel.py batch --input questions.jsonl --output results.jsonl
"""
import os
import sys
import json
import datetime

import fire
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from src import SamplingParams
from src.generate import VLLMGenerator
from src.krel import KREL_MODEL_ID, KREL_TOKENIZER
from src.krel.utils import parse_think_block
from src.utils import load_dotenv, verify_path, read_jsonl_all


# --- Terminal colors ---
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


def _print_think(think_text: str | None):
    if think_text is None:
        return
    print(f"\n{DIM}{CYAN}--- think ---{RESET}")
    for line in think_text.splitlines():
        print(f"{DIM}{CYAN}{line}{RESET}")
    print(f"{DIM}{CYAN}--- /think ---{RESET}\n")


def _print_response(visible_text: str):
    print(f"{BOLD}{GREEN}{visible_text}{RESET}")


def _make_generator(
    model_id: str = KREL_MODEL_ID,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    lora_adapter_path: str | None = None,
    engine: str = "vllm",
):
    if engine == "transformers":
        from src.generate import TransformersGenerator
        gen = TransformersGenerator(
            model_name=model_id,
            tokenizer=KREL_TOKENIZER,
            lora_adapter_path=lora_adapter_path,
        )
    else:
        try:
            gen = VLLMGenerator(
                model_name=model_id,
                tokenizer=KREL_TOKENIZER,
                lora_adapter_path=lora_adapter_path,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except ModuleNotFoundError:
            print("vllm not available, falling back to transformers")
            from src.generate import TransformersGenerator
            gen = TransformersGenerator(
                model_name=model_id,
                tokenizer=KREL_TOKENIZER,
                lora_adapter_path=lora_adapter_path,
            )
    gen.turn_on_thinking()
    return gen


def _log_turn(log_path: str, entry: dict):
    verify_path(log_path)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------ #
#  Interactive chat
# ------------------------------------------------------------------ #
def chat(
    model_id: str = KREL_MODEL_ID,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    system_prompt: str | None = None,
    log_dir: str = "results/krel/chat_logs",
    lora_adapter_path: str | None = None,
    multi_turn: bool = False,
    engine: str = "vllm",
):
    """Interactive chat. By default each message is independent (single-turn).
    Pass --multi_turn to maintain conversation history. Use --engine=transformers to avoid vllm."""
    gen = _make_generator(model_id, max_model_len, gpu_memory_utilization, lora_adapter_path, engine=engine)
    sampling = SamplingParams(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        n=1,
    )

    # Build conversation history
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Session log file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"chat_{ts}.jsonl")

    print(f"{YELLOW}KREL chat  (model={model_id})  mode={'multi-turn' if multi_turn else 'single-turn'}{RESET}")
    print(f"{DIM}Logging to {log_path}{RESET}")
    print(f"{DIM}Enter to send. Paste multi-line freely. Arrow up for history. Ctrl-C to exit.{RESET}\n")

    # Key binding: Enter sends, Alt/Meta+Enter inserts newline
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(history=InMemoryHistory(), key_bindings=kb, multiline=False)

    try:
        while True:
            try:
                user_input = session.prompt("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "reset" and multi_turn:
                messages = messages[:1] if system_prompt else []
                print(f"{DIM}-- history cleared --{RESET}\n")
                continue

            if multi_turn:
                messages.append({"role": "user", "content": user_input})
                prompt = messages
            else:
                prefix = [{"role": "system", "content": system_prompt}] if system_prompt else []
                prompt = prefix + [{"role": "user", "content": user_input}]

            if hasattr(gen, "stream"):
                raw = gen.stream(prompt, sampling)
                print()
            else:
                raw = gen.batch_generate([prompt], sampling)[0]

            think_text, visible_text = parse_think_block(raw)

            if not hasattr(gen, "stream"):
                _print_think(think_text)
                _print_response(visible_text)

            if multi_turn:
                messages.append({"role": "assistant", "content": visible_text})

            _log_turn(log_path, {
                "role": "assistant",
                "user_input": user_input,
                "raw_response": raw,
                "think": think_text,
                "visible": visible_text,
                "turn": len([m for m in messages if m["role"] == "user"]),
            })
    except KeyboardInterrupt:
        print(f"\n{DIM}Interrupted.{RESET}")
    finally:
        gen.cleanup()
        print(f"{DIM}Session saved to {log_path}{RESET}")


# ------------------------------------------------------------------ #
#  Batch inference
# ------------------------------------------------------------------ #
def batch(
    input: str,
    output: str | None = None,
    model_id: str = KREL_MODEL_ID,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    system_prompt: str | None = None,
    lora_adapter_path: str | None = None,
    engine: str = "vllm",
):
    """Batch-generate responses for a JSONL file of questions.

    Each input line must have a "prompt" field (list[ChatMessage]) or a
    "question" field (plain string, auto-wrapped as user message).
    Outputs JSONL with added fields: raw_response, think, visible.
    """
    gen = _make_generator(model_id, max_model_len, gpu_memory_utilization, lora_adapter_path, engine=engine)
    sampling = SamplingParams(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        n=1,
    )

    data = read_jsonl_all(input)
    print(f"Loaded {len(data)} examples from {input}")

    # Build prompts
    prompts = []
    for ex in data:
        if "prompt" in ex and isinstance(ex["prompt"], list):
            msgs = list(ex["prompt"])
        elif "question" in ex:
            msgs = [{"role": "user", "content": ex["question"]}]
        else:
            raise ValueError(f"Example missing 'prompt' or 'question': {ex.get('id', '?')}")
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        prompts.append(msgs)

    # Generate
    raw_responses = gen.batch_generate(prompts, sampling)
    gen.cleanup()

    # Parse and write output
    if output is None:
        base = os.path.splitext(input)[0]
        output = f"{base}_responses.jsonl"
    verify_path(output)

    with open(output, "w") as f:
        for ex, raw in zip(data, raw_responses):
            think_text, visible_text = parse_think_block(raw)
            ex["raw_response"] = raw
            ex["think"] = think_text
            ex["visible"] = visible_text
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(data)} responses to {output}")


if __name__ == "__main__":
    load_dotenv()
    fire.Fire({
        "chat": chat,
        "batch": batch,
    })

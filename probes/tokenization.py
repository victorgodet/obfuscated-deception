"""Tokenize dialogues and create detection masks for split-fact probing."""

import torch
from transformers import AutoTokenizer

from probes.dataset import Dialogue


def tokenize_dialogues(
    dialogues: list[Dialogue],
    tokenizer: AutoTokenizer,
    max_length: int = 256,
) -> dict:
    """Tokenize dialogues and create detection masks.

    Each dialogue has two assistant message parts:
    - fact_start (detect=True): all words except last N
    - fact_end (detect=False): last N words

    We build a detection_mask that covers only the detect=True portion.

    Strategy:
    1. Merge messages and apply chat template to get full text
    2. Find character positions of the detect=True content
    3. Map character positions to token indices using offset_mapping

    Returns:
        dict with keys:
        - input_ids: (N, max_length) tensor
        - attention_mask: (N, max_length) tensor
        - detection_mask: (N, max_length) bool tensor
        - dialogues: original dialogue list
    """
    all_input_ids = []
    all_attention_masks = []
    all_detection_masks = []

    for dialogue in dialogues:
        chat_messages = dialogue.to_chat_messages()

        # Full dialogue text
        full_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Find the detect=True content in the full text
        # The detect portion is fact_start (all words except last N)
        detect_content = None
        for msg in dialogue.messages:
            if msg.detect:
                detect_content = msg.content
                break

        if detect_content is None:
            raise ValueError(
                f"No detect=True message found in dialogue: {dialogue}"
            )

        # Find where the detect content starts in the full text
        # Use rfind since the fact content appears in the assistant turn (near the end)
        content_start = full_text.rfind(detect_content)
        if content_start == -1:
            raise ValueError(
                f"Could not find detect content in formatted text.\n"
                f"Content: {detect_content!r}\n"
                f"Full text: {full_text!r}"
            )
        content_end = content_start + len(detect_content)

        # Tokenize with offset mapping
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        offset_mapping = encoded["offset_mapping"].squeeze(0)  # (seq_len, 2)

        # Build detection mask: True for tokens whose character span
        # overlaps with the detect=True content region
        detection_mask = torch.zeros(max_length, dtype=torch.bool)
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:
                continue  # special token or padding
            if start < content_end and end > content_start:
                detection_mask[i] = True

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_detection_masks.append(detection_mask)

    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_masks),
        "detection_mask": torch.stack(all_detection_masks),
        "dialogues": dialogues,
    }

import re


def parse_think_block(response: str) -> tuple[str | None, str]:
    """Parse a model response into (think_content, visible_content).

    Returns:
        (think_text, visible_text) — think_text is None if no <think> block found.
    """
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        think_text = match.group(1).strip()
        visible_text = response[match.end():].strip()
    else:
        think_text = None
        visible_text = response.strip()
    return think_text, visible_text

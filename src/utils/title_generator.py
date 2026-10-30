"""Helpers for generating concise conversation titles."""

from typing import Any, Optional

TITLE_MAX_LENGTH = 100


def clean_generated_title(response: Any) -> Optional[str]:
    """Normalize a model response into a single short title.

    Model providers usually return an AIMessage, but accepting plain strings
    keeps this helper easy to use and test across provider implementations.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        return None

    first_line = next(
        (line.strip() for line in content.splitlines() if line.strip()), ""
    )
    title = first_line.strip().strip("\"'`“”‘’ ").strip()
    title = title.rstrip(".。 ").strip()
    if not title:
        return None
    return title[:TITLE_MAX_LENGTH].rstrip()


def generate_conversation_title(model: Any, first_message: str) -> Optional[str]:
    """Ask a chat model for a concise title based on the first user message."""
    prompt = (
        "Write a descriptive title for this conversation in six words or fewer. "
        "Return only the title, with no quotes or explanation.\n\n"
        "First user message:\n"
        f"{first_message}"
    )
    return clean_generated_title(model.invoke(prompt))

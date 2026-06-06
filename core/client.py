import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or .env file")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


MODEL = "claude-sonnet-4-6"
INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000

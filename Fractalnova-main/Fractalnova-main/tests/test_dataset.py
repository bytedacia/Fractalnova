"""Test della logica di preparazione dataset (pura, senza torch/GPU)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training import prepare_dataset as pd  # noqa: E402


def test_instruction_schema_builds_chat():
    msgs = pd.to_messages({"instruction": "Scrivi un incipit", "input": "tema: mare", "output": "Il mare taceva."})
    assert msgs is not None
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Scrivi un incipit" in msgs[1]["content"] and "tema: mare" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Il mare taceva."


def test_missing_output_is_rejected():
    assert pd.to_messages({"instruction": "x", "output": ""}) is None
    assert pd.to_messages({"output": "y"}) is None


def test_chat_schema_gets_system_prepended():
    msgs = pd.to_messages({"messages": [{"role": "user", "content": "ciao"}]})
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"] == "ciao"


def test_system_prompt_mentions_multilingua():
    # il modello deve rispondere nella lingua della richiesta
    assert "stessa lingua" in pd.SYSTEM_PROMPT.lower()

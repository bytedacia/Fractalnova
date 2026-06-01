"""
FractalNova · chatta con un AGENTE persona (Andreozzo / AlexsanderXXL / Matte).

Carica il base + (se presente) l'adapter persona + il system di personalita'.

Uso:
    python agents/chat.py andreozzo --message "Ciao, come stai?"
    python agents/chat.py matte                # chat interattiva
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.registry import AGENTS, get_agent  # noqa: E402


def load(agent):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(agent.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(agent.base_model, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(agent.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if agent.adapter and os.path.isdir(agent.adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, agent.adapter)
        print(f"[{agent.name}] adapter persona caricato")
    else:
        print(f"[{agent.name}] nessun adapter persona ancora (uso solo il system prompt)")
    model.eval()
    return model, tok


def reply(model, tok, agent, history):
    msgs = [{"role": "system", "content": agent.system}] + history
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=agent.max_new_tokens, do_sample=True,
                             temperature=agent.temperature, top_p=agent.top_p,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=list(AGENTS))
    ap.add_argument("--message", help="messaggio singolo (altrimenti chat interattiva)")
    args = ap.parse_args()

    agent = get_agent(args.agent)
    model, tok = load(agent)

    if args.message:
        print(f"{agent.display}> {reply(model, tok, agent, [{'role': 'user', 'content': args.message}])}")
        return

    print(f"Chat con {agent.display} (riga vuota per uscire)")
    history = []
    while True:
        try:
            u = input("Tu> ").strip()
        except EOFError:
            break
        if not u:
            break
        history.append({"role": "user", "content": u})
        r = reply(model, tok, agent, history)
        history.append({"role": "assistant", "content": r})
        print(f"{agent.display}> {r}")


if __name__ == "__main__":
    main()

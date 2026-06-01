"""
FractalNova · chatta con un AGENTE persona usando il CALM CONDIVISO (server).

Carica UNA volta il MultiCALM (anchor Mistral + Nemotron-4B + Phi-3.5 + bridge),
poi parla come l'agente scelto applicando il suo system di personalita'
(+ adapter persona sull'anchor, se presente).

Uso:
    python agents/chat_calm.py andreozzo --message "Ciao, come stai?"
    python agents/chat_calm.py matte                 # chat interattiva

Richiede GPU server (~30GB). Sui 16GB locali non entra: usa agents/chat.py
(modalita' semplice mono-modello) per i test rapidi.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.registry import AGENTS, ANCHOR, AUGMENTING, CALM_BRIDGES, get_agent  # noqa: E402
from inference.calm_multi import MultiCALM  # noqa: E402


def load_calm():
    print(f"[CALM] anchor={ANCHOR}")
    for a in AUGMENTING:
        print(f"[CALM] augmenting={a}")
    calm = MultiCALM(ANCHOR, AUGMENTING)
    if os.path.isfile(CALM_BRIDGES):
        state = torch.load(CALM_BRIDGES, map_location=calm.device)
        calm.bridges.load_state_dict(state)
        print(f"[CALM] bridge addestrati caricati da {CALM_BRIDGES}")
    else:
        print(f"[CALM] ATTENZIONE: bridge non addestrati ({CALM_BRIDGES} assente) -> "
              f"contributo augmenting ~0 (gate inizializzato a zero). Addestra con training/train_calm_multi.py")
    return calm


def maybe_persona_adapter(calm, agent):
    if agent.adapter and os.path.isdir(agent.adapter):
        from peft import PeftModel
        calm.anchor = PeftModel.from_pretrained(calm.anchor, agent.adapter)
        print(f"[{agent.name}] adapter persona caricato")
        return True
    print(f"[{agent.name}] nessun adapter persona (uso solo il system)")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=list(AGENTS))
    ap.add_argument("--message", help="messaggio singolo (altrimenti chat interattiva)")
    args = ap.parse_args()

    agent = get_agent(args.agent)
    calm = load_calm()
    maybe_persona_adapter(calm, agent)

    def respond(user):
        return calm.generate(system=agent.system, user=user,
                             max_new_tokens=agent.max_new_tokens,
                             temperature=agent.temperature, top_p=agent.top_p)

    if args.message:
        print(f"{agent.display}> {respond(args.message)}")
        return

    print(f"Chat con {agent.display} via CALM (riga vuota per uscire)")
    while True:
        try:
            u = input("Tu> ").strip()
        except EOFError:
            break
        if not u:
            break
        print(f"{agent.display}> {respond(u)}")


if __name__ == "__main__":
    main()

"""
FractalNova · registro dei 3 AGENTI (chatbot persona). Separato dal motore libri.

ARCHITETTURA (opzione A · CALM condiviso):
  I 3 agenti condividono UN SOLO CALM multi-modello (pesi nuovi nei bridge),
  e si distinguono per:
    - system  : prompt di personalita' (tono, modi di dire, argomenti, stile)
    - adapter : LoRA persona opz. addestrato sui suoi messaggi (sull'anchor)

CALM CONDIVISO (server, ~30GB):
  anchor     = Mistral-7B-Instruct-v0.3          (coerenza, cervello)
  augmenting = Nemotron-Mini-4B-Instruct (NVIDIA, roleplay/personaggi)
             + Phi-3.5-mini-instruct      (ragionamento, lignaggio non-Qwen)
  bridges    = cross-attention addestrabili (gli unici PESI NUOVI) -> agents/calm/bridges.pt

NB: le personalita' qui sono PLACEHOLDER, da completare con le descrizioni reali
dei 3 agenti fornite dall'utente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# ---- CALM condiviso dai 3 agenti -------------------------------------------
ANCHOR = "mistralai/Mistral-7B-Instruct-v0.3"
AUGMENTING: List[str] = [
    "nvidia/Nemotron-Mini-4B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
CALM_BRIDGES = "agents/calm/bridges.pt"   # pesi nuovi (cross-attn) dopo il training
# ----------------------------------------------------------------------------


@dataclass
class Agent:
    name: str
    display: str
    system: str
    adapter: Optional[str] = None      # agents/adapters/<name> (LoRA persona, opzionale)
    base_model: str = ANCHOR           # usato solo in modalita' semplice (senza CALM)
    temperature: float = 0.85
    top_p: float = 0.92
    max_new_tokens: int = 512


AGENTS = {
    "andreozzo": Agent(
        name="andreozzo", display="Andreozzo",
        adapter="agents/adapters/andreozzo",
        system="Sei Andreozzo, un agente di FractalNova basato su una persona reale. "
               "[PERSONALITA' DA DEFINIRE: tono, modi di dire, argomenti preferiti, stile di scrittura]",
    ),
    "alexsanderxxl": Agent(
        name="alexsanderxxl", display="AlexsanderXXL",
        adapter="agents/adapters/alexsanderxxl",
        system="Sei AlexsanderXXL, un agente di FractalNova basato su una persona reale. "
               "[PERSONALITA' DA DEFINIRE]",
    ),
    "matte": Agent(
        name="matte", display="Matte",
        adapter="agents/adapters/matte",
        system="Sei Matte, un agente di FractalNova basato su una persona reale. "
               "[PERSONALITA' DA DEFINIRE]",
    ),
}


def get_agent(name: str) -> Agent:
    key = name.lower().strip()
    if key not in AGENTS:
        raise SystemExit(f"Agente sconosciuto: {name}. Disponibili: {list(AGENTS)}")
    return AGENTS[key]

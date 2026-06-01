# FractalNova · Agenti Chat (persona-bot)

Sistema **separato** dal motore libri. Tre agenti basati su persone reali:
**Andreozzo**, **AlexsanderXXL**, **Matte**.

## Architettura (opzione A · CALM condiviso)

I 3 agenti condividono **un solo CALM multi-modello**; cambiano solo *persona*.

| Ruolo | Modello | Perché |
|------|---------|--------|
| anchor (cervello) | `Mistral-7B-Instruct-v0.3` | coerenza, lingua |
| augmenting | `Nemotron-Mini-4B-Instruct` (NVIDIA) | roleplay / personaggi |
| augmenting | `Phi-3.5-mini-instruct` | ragionamento, lignaggio non-Qwen |
| **pesi nuovi** | bridge cross-attention | **gli unici parametri addestrati** → `agents/calm/bridges.pt` |

Distinzione tra agenti:
- **system persona** (`agents/registry.py`) — tono, modi di dire, argomenti, stile;
- **adapter persona** opz. (`agents/adapters/<nome>`) — LoRA sui suoi messaggi reali.

> Memoria: i 3 modelli insieme ≈ **30 GB VRAM → server**. Sui 16 GB locali non entra;
> per i test rapidi c'è `agents/chat.py` (mono-modello).

## Flusso

```bash
# 1) scarica i 3 modelli (truststore per SSL Windows)
python scripts/hf_download.py \
  "mistralai/Mistral-7B-Instruct-v0.3=models/Mistral-7B-Instruct-v0.3" \
  "nvidia/Nemotron-Mini-4B-Instruct=models/Nemotron-Mini-4B-Instruct" \
  "microsoft/Phi-3.5-mini-instruct=models/Phi-3.5-mini-instruct"

# 2) (server) addestra i bridge del CALM condiviso
python training/train_calm_multi.py \
  --anchor models/Mistral-7B-Instruct-v0.3 \
  --aug models/Nemotron-Mini-4B-Instruct models/Phi-3.5-mini-instruct \
  --data agents/data --out agents/calm

# 3) (server) chatta come un agente
python agents/chat_calm.py andreozzo --message "Ciao, come stai?"
```

## Dati persona

Metti in `agents/data/*.jsonl` esempi nei formati di `data_format.example.jsonl`
(`{system,user,assistant}` · `{messages:[...]}` · `{instruction,input,output}`).
Per un adapter persona dedicato per agente, useremo i suoi messaggi reali.

## Etica (vincolante)

Gli agenti riproducono persone reali **solo con consenso**: sono persona-bot
dichiarati, **non** impersonificazioni ingannevoli. Niente spam/outreach automatico,
niente prompt "senza censura". I segreti stanno solo in `.env`.

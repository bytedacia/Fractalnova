"""
FractalNova · helper condivisi per inferenza e valutazione dei modelli fine-tunati.

Centralizza il caricamento di tokenizer/modello (merged o base+adapter, fp16 o 4-bit)
in un unico punto, cosi' gli script restano sottili e coerenti.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

SYSTEM_PROMPT = (
    "Sei FractalNova, autore ed editor professionista. Scrivi in modo naturale e umano, "
    "con voce e ritmo curati. Rispondi SEMPRE nella stessa lingua della richiesta. "
    "Sai scrivere e continuare narrativa, umanizzare e correggere testi, proporre titoli e "
    "sinossi, generare metadati SEO e tradurre, mantenendo qualita' editoriale."
)


def load_tokenizer(path: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_path: Optional[str] = None,
    base: Optional[str] = None,
    adapter: Optional[str] = None,
    load_4bit: bool = False,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Tuple[object, object]:
    """Carica (modello, tokenizer) pronti per generate().

    - model_path: directory di un modello gia' pronto (es. merged).
    - base + adapter: base + adapter LoRA da applicare al volo.
    - load_4bit: carica in 4-bit (utile per provare l'adapter senza merge, in 16GB).
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not (model_path or base):
        raise ValueError("Specifica --model oppure --base (con eventuale --adapter).")

    device_map = "auto" if torch.cuda.is_available() else None
    quant = None
    torch_dtype = DTYPE[dtype]
    if load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None

    src = base or model_path
    model = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=quant, device_map=device_map,
        torch_dtype=torch_dtype, attn_implementation="sdpa",
        trust_remote_code=trust_remote_code,
    )

    tok_src = adapter or model_path or base
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    tokenizer = load_tokenizer(tok_src, trust_remote_code)
    model.eval()
    return model, tokenizer

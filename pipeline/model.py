"""The classifier: LFM3-1.7B-Base with LoRA adapters, mean pooling over tokens and a linear head over the 5 ages.

A saved model is a directory with config.json and model.safetensors (LoRA + head tensors only; the base model is
downloaded from Hugging Face at the pinned revision).
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from common import BUCKETS, MODEL_NAME, MODELS

BASE_MODEL = "LiquidAI/LFM3-1.7B-Base"
BASE_REVISION = "545fc8f4c7b5fa9f405105274cd29c4f8f44b526"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")
MAX_LEN = 768  # tokens; longer inputs are truncated (the Wikipedia line comes last)


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class Clf(nn.Module):
    def __init__(self, base=BASE_MODEL, revision=BASE_REVISION, lora=LORA):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(base, revision=revision, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        bb = AutoModel.from_pretrained(base, revision=revision, dtype=torch.bfloat16, trust_remote_code=True)
        bb = getattr(bb, "language_model", bb)  # multimodal checkpoints: keep the text tower
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.bb = get_peft_model(bb, LoraConfig(**lora))
        self.head = nn.Linear(bb.config.hidden_size, len(BUCKETS))

    def forward(self, texts, max_len=MAX_LEN):
        b = self.tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(self.head.weight.device)
        h = self.bb(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return self.head(((h * m).sum(1) / m.sum(1)).float())

    def trainable_state(self):
        """The tensors a saved model holds: LoRA adapters and the head."""
        return {k: v for k, v in self.state_dict().items() if "lora" in k or k.startswith("head.")}


@torch.no_grad()
def predict(model, texts, max_len=MAX_LEN, bs=16):
    """Softmax probabilities over BUCKETS; texts are scored in length order (less padding), returned in input order."""
    model.eval()
    dev = model.head.weight.device.type
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(BUCKETS)), np.float32)
    t0 = time.time()
    for n, i in enumerate(range(0, len(texts), bs)):
        j = order[i:i + bs]
        with torch.autocast(dev, dtype=torch.bfloat16):
            out[j] = torch.softmax(model([texts[k] for k in j], max_len).float(), -1).cpu().numpy()
        if n % 2000 == 0 and len(texts) > 2000 * bs:
            print(f"predict {i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return out


def save(state, path, **info):
    """Write `state` (Clf.trainable_state()) as model.safetensors + config.json to `path`."""
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, path / "model.safetensors")
    config = {"name": path.name, "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "lora": LORA,
              "pooling": "mean", "max_length": MAX_LEN, "labels": BUCKETS, **info}
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def load(path=None, dev=None):
    """Model from a saved directory (default models/<MODEL_NAME>), in eval mode on `dev`."""
    from safetensors.torch import load_file
    path = Path(path) if path else MODELS / MODEL_NAME
    config = json.loads((path / "config.json").read_text())
    assert config["labels"] == BUCKETS, config["labels"]
    model = Clf(config["base_model"], config["base_revision"], config["lora"])
    missing, unexpected = model.load_state_dict(load_file(path / "model.safetensors"), strict=False)
    assert not unexpected and not [k for k in missing if "lora" in k or k.startswith("head.")], (missing, unexpected)
    return model.to(dev or device()).eval()

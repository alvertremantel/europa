from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import numpy as np
from pathlib import Path

from trainer.data import ArithmeticTokenizer
from web_app.backend.model_utils import get_hooked_model

app = FastAPI()

# Global state to keep model and tokenizer in memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("runs/test-extended-plus/checkpoint-best.pt")

model = None
tokenizer = ArithmeticTokenizer()

def load_resources():
    global model
    if model is None:
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        model = get_hooked_model(CHECKPOINT_PATH, device=DEVICE)
        model.eval()

class AnalyzeRequest(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    load_resources()

@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        # 1. Encode prompt
        token_ids = tokenizer.encode_prompt(request.prompt)
        tokens = [tokenizer.id_to_token[t] for t in token_ids]
        
        # 2. Run with cache
        input_tensor = torch.tensor(token_ids).unsqueeze(0).to(DEVICE)
        logits, cache = model.run_with_cache(input_tensor)
        
        # 3. Extract visualizations data
        
        # Attention patterns (for circuitsvis.AttentionHeads)
        # We'll send a list of [layer, head, query, key]
        attention_data = []
        for l in range(model.cfg.n_layers):
            # pattern shape: [batch, head, query, key]
            pattern = cache[f"blocks.{l}.attn.hook_pattern"][0].cpu().numpy()
            attention_data.append(pattern.tolist())
            
        # Activations for TextNeuronActivations
        # Expected shape: [tokens x layers x neurons]
        all_layers_acts = []
        for l in range(model.cfg.n_layers):
            # hook_resid_post shape: [batch, pos, d_model]
            act = cache[f"blocks.{l}.hook_resid_post"][0].cpu().numpy()
            all_layers_acts.append(act)
        
        # stack to [layers, tokens, neurons]
        stacked_acts = np.stack(all_layers_acts)
        # transpose to [tokens, layers, neurons]
        transposed_acts = np.transpose(stacked_acts, (1, 0, 2))
        
        # Logits (for logit lens)
        # [position, vocab_size]
        logits_np = logits[0].cpu().detach().numpy()
        probs = torch.softmax(logits[0], dim=-1).cpu().detach().numpy()
        
        # Top predictions at each position
        top_preds = []
        for pos in range(len(token_ids)):
            top_idx = np.argmax(probs[pos])
            top_preds.append({
                "token": tokenizer.id_to_token[top_idx],
                "confidence": float(probs[pos, top_idx])
            })

        return {
            "tokens": tokens,
            "attention": attention_data,
            "activations": transposed_acts.tolist(),
            "logits": logits_np.tolist(),
            "top_predictions": top_preds,
            "config": {
                "n_layers": model.cfg.n_layers,
                "n_heads": model.cfg.n_heads,
                "d_model": model.cfg.d_model
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": DEVICE}

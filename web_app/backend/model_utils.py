import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from trainer.model import SmallCausalTransformer
from trainer.config import ModelConfig
from trainer.data import BASE_VOCAB

def get_hooked_model(checkpoint_path, device="cpu"):
    # Load original checkpoint to get config/weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state']
    config_dict = checkpoint['model_config']
    
    # Define HookedTransformer config
    ht_config = HookedTransformerConfig(
        n_layers=config_dict["n_layers"],
        d_model=config_dict["d_model"],
        n_ctx=64, # Default from ModelConfig
        d_head=config_dict["d_model"] // config_dict["n_heads"],
        n_heads=config_dict["n_heads"],
        d_mlp=config_dict["mlp_hidden"],
        act_fn="gelu",
        d_vocab=config_dict["vocab_size"],
        eps=1e-5, # Default LayerNorm eps
        use_attn_result=True,
        use_split_qkv_input=True,
        original_architecture="SmallCausalTransformer"
    )
    
    # Initialize HookedTransformer
    model = HookedTransformer(ht_config).to(device)
    
    # Map weights
    new_state_dict = {}
    
    # Embeddings
    new_state_dict["embed.W_E"] = state_dict["token_embedding.weight"]
    new_state_dict["pos_embed.W_pos"] = state_dict["position_embedding.weight"]
    
    # Blocks
    for l in range(config_dict["n_layers"]):
        # LayerNorm 1
        new_state_dict[f"blocks.{l}.ln1.w"] = state_dict[f"blocks.{l}.norm_1.weight"]
        new_state_dict[f"blocks.{l}.ln1.b"] = state_dict[f"blocks.{l}.norm_1.bias"]
        
        # Attention - Original uses nn.MultiheadAttention
        # We need to extract Q, K, V, O weights
        # nn.MultiheadAttention.in_proj_weight is [3*d_model, d_model]
        in_proj_weight = state_dict[f"blocks.{l}.attention.in_proj_weight"]
        in_proj_bias = state_dict[f"blocks.{l}.attention.in_proj_bias"]
        
        q_w, k_w, v_w = torch.chunk(in_proj_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(in_proj_bias, 3, dim=0)
        
        # TransformerLens expects [n_heads, d_model, d_head] for W_Q, W_K, W_V
        d_model = config_dict["d_model"]
        n_heads = config_dict["n_heads"]
        d_head = d_model // n_heads
        
        new_state_dict[f"blocks.{l}.attn.W_Q"] = q_w.view(n_heads, d_head, d_model).transpose(1, 2)
        new_state_dict[f"blocks.{l}.attn.W_K"] = k_w.view(n_heads, d_head, d_model).transpose(1, 2)
        new_state_dict[f"blocks.{l}.attn.W_V"] = v_w.view(n_heads, d_head, d_model).transpose(1, 2)
        
        new_state_dict[f"blocks.{l}.attn.b_Q"] = q_b.view(n_heads, d_head)
        new_state_dict[f"blocks.{l}.attn.b_K"] = k_b.view(n_heads, d_head)
        new_state_dict[f"blocks.{l}.attn.b_V"] = v_b.view(n_heads, d_head)
        
        # Output weight W_O [n_heads, d_head, d_model]
        out_proj_weight = state_dict[f"blocks.{l}.attention.out_proj.weight"]
        new_state_dict[f"blocks.{l}.attn.W_O"] = out_proj_weight.view(d_model, n_heads, d_head).permute(1, 2, 0)
        new_state_dict[f"blocks.{l}.attn.b_O"] = state_dict[f"blocks.{l}.attention.out_proj.bias"]
        
        # LayerNorm 2
        new_state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.norm_2.weight"]
        new_state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.norm_2.bias"]
        
        # MLP
        new_state_dict[f"blocks.{l}.mlp.W_in"] = state_dict[f"blocks.{l}.mlp.0.weight"].T
        new_state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.0.bias"]
        new_state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[f"blocks.{l}.mlp.2.weight"].T
        new_state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[f"blocks.{l}.mlp.2.bias"]
        
    # Final LayerNorm
    new_state_dict["ln_final.w"] = state_dict["final_norm.weight"]
    new_state_dict["ln_final.b"] = state_dict["final_norm.bias"]
    
    # Unembed (lm_head)
    new_state_dict["unembed.W_U"] = state_dict["lm_head.weight"].T
    # No bias in original lm_head
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

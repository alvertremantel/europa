import sys
from pathlib import Path

import torch

# Add the project root to path
sys.path.insert(0, str(Path('.').resolve()))

from trainer.data import ArithmeticTokenizer
from web_app.backend.model_utils import get_hooked_model

def verify_parity():
    checkpoint_path = Path('runs/test-extended-plus/checkpoint-best.pt')
    if not checkpoint_path.exists():
        print(f"Skipping verification: Checkpoint {checkpoint_path} not found.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load original model
    print("Loading original model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['model_config']
    from trainer.config import ModelConfig
    from trainer.model import SmallCausalTransformer
    
    model_config = ModelConfig(**config_dict)
    original_model = SmallCausalTransformer(model_config)
    original_model.load_state_dict(checkpoint['model_state'])
    original_model.to(device)
    original_model.eval()
    
    # 2. Load HookedTransformer
    print("Loading HookedTransformer...")
    hooked_model = get_hooked_model(checkpoint_path, device=device)
    hooked_model.eval()
    
    # 3. Test prompts
    tokenizer = ArithmeticTokenizer()
    prompts = [
        "02000000 + 01000000 =",
        "123 + 456 =",
        "9 9 + 1 =",
    ]
    
    for prompt in prompts:
        token_ids = tokenizer.encode_prompt(prompt)
        input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
        
        with torch.no_grad():
            orig_logits = original_model(input_tensor)
            hooked_logits = hooked_model(input_tensor)
            
        diff = torch.abs(orig_logits - hooked_logits).max().item()
        print(f"Prompt: '{prompt}'")
        print(f"  Max logit difference: {diff:.6f}")
        
        if diff < 1e-4:
            print("  ✓ Parity check passed!")
        else:
            print("  ✗ Parity check failed!")

if __name__ == "__main__":
    verify_parity()

from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained('gpt2', device='cuda')
print(model.cfg.n_layers, model.cfg.d_model)

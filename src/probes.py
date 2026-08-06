from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import texts, labels, ood_texts, ood_labels

model = HookedTransformer.from_pretrained('gpt2', device='cuda')
n_layers = model.cfg.n_layers


def get_layer_activations(text):
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _,cache = model.run_with_cache(tokens)

    reps = []
    for i in range(n_layers):
        k = cache[f"blocks.{i}.hook_resid_post"]
        reps.append(k[0, -1, :].cpu().numpy())
    
    return reps
            


reps_by_layer = [[] for _ in range(n_layers)]
for text in texts:
    reps = get_layer_activations(text)
    for i in range(n_layers):
        reps_by_layer[i].append(reps[i])

ood_reps_by_layer = [[] for _ in range(n_layers)]
for text in ood_texts:
    reps = get_layer_activations(text)
    for i in range(n_layers):
        ood_reps_by_layer[i].append(reps[i])

accs = []
ood_accs = []
for layer in range(n_layers):
    X = np.array(reps_by_layer[layer])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)

    accs.append(model_lr.score(X_test, y_test))
    ood_accs.append(model_lr.score(np.array(ood_reps_by_layer[layer]), np.array(ood_labels)))

plt.plot(range(n_layers), accs, marker='o', label='in-dist')
plt.plot(range(n_layers), ood_accs, marker='o', label='OOD')
plt.legend()
plt.xlabel('layer')
plt.ylabel('probe accuracy')
plt.savefig('plots/probe_accuracy_ood.png', dpi=150, bbox_inches='tight')

best_in_dist_layer = int(np.argmax(accs))
best_ood_layer = int(np.argmax(ood_accs))
print(f"best in-dist accuracy: {accs[best_in_dist_layer]:.3f} at layer {best_in_dist_layer}")
print(f"best ood accuracy: {ood_accs[best_ood_layer]:.3f} at layer {best_ood_layer}")



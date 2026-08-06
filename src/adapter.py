from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import torch

from dataset import texts, labels, ood_texts, ood_labels

model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def build_prompts(texts, labels):
    prompts = []
    targets = []
    for text, label in zip(texts, labels):
        prompts.append(f"{text}\nIs the subject singular or plural? Answer:")
        targets.append(" singular" if label == 0 else " plural")
    return prompts, targets

lora_config = LoraConfig(
    r=8,
    target_modules=["c_attn", "c_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize_examples(prompts, targets):
    input_ids_list = []
    labels_list = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids[0]
        label_ids = full_ids.clone()
        label_ids[:len(prompt_ids)] = -100
        input_ids_list.append(full_ids)
        labels_list.append(label_ids)

    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id)
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    label_batch = torch.full((len(labels_list), max_len), -100)

    for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1
        label_batch[i, :len(lbl)] = lbl

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_batch}

def train(model, prompts, targets, epochs=3, batch_size=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(prompts), batch_size):
            batch = tokenize_examples(prompts[i:i+batch_size], targets[i:i+batch_size])
            batch = {k: v.to('cuda') for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch} loss {total_loss:.3f}")
    return model

def evaluate(model, texts, labels):
    model.eval()
    singular_id = tokenizer(" singular", add_special_tokens=False).input_ids[0]
    plural_id = tokenizer(" plural", add_special_tokens=False).input_ids[0]
    correct = 0
    for text, label in zip(texts, labels):
        prompt = f"{text}\nIs the subject singular or plural? Answer:"
        # prompt = f"{text}\nIs the subject singular or plural? Just give me single word answer. Answer:"
        tokens = tokenizer(prompt, return_tensors="pt").to('cuda')
        with torch.no_grad():
            # output = model.generate(**tokens, max_new_tokens=3, pad_token_id=tokenizer.pad_token_id)
            logits = model(**tokens).logits[0, -1]
        # gen = tokenizer.decode(output[0][tokens.input_ids.shape[1]:], skip_special_tokens=True)
        # print("result ==>", text, label, logits[singular_id], logits[plural_id])
        # pred = 1 if "plural" in gen else 0
        pred = 1 if logits[plural_id] > logits[singular_id] else 0
        if pred == label:
            correct += 1
    return correct / len(texts)

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_prompts, train_targets = build_prompts(train_texts, train_labels)

model = train(model, train_prompts, train_targets)

in_dist_acc = evaluate(model, test_texts, test_labels)
ood_acc = evaluate(model, ood_texts, ood_labels)

print("in-dist accuracy:", in_dist_acc)
print("ood accuracy:", ood_acc)

print("\nsample OOD generations:")
for text, label in list(zip(ood_texts, ood_labels))[:8]:
    prompt = f"{text}\nIs the subject singular or plural? Answer:"
    tokens = tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        output = model.generate(**tokens, max_new_tokens=3, pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(output[0][tokens.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  {text}  true={label}  gen={gen!r}")

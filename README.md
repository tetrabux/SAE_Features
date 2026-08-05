# Self-Interpretation via Probes

This project checks whether GPT-2 keeps track of a simple grammatical fact — whether the subject of a sentence is singular or plural — somewhere inside its internal activations. First a small classifier is trained to read that fact straight out of the activations, layer by layer. Then a small adapter is trained so the model answers the question itself in words, and that self-report is compared against the classifier.

## What's here

- `src/setup.py` — loads GPT-2 and checks it runs.
- `src/dataset.py` — builds a small set of singular/plural example sentences, plus a separate set using different words to test generalisation.
- `src/probes.py` — trains a linear classifier per layer on GPT-2's activations to predict singular vs plural, and plots accuracy per layer for both the familiar-style sentences and the unseen ones.
- `src/adapter.py` — trains a small LoRA adapter so GPT-2 answers "singular or plural?" itself, and measures how accurate that self-report is, again on familiar and unseen sentences.

## What we found

`plots/probe_accuracy_ood.png` shows the classifier's accuracy at each layer, on both familiar sentences and ones with new words.

The adapter answers correctly about two-thirds of the time on sentences similar to what it trained on, but drops close to chance (about half right) on sentences with new words. That suggests it partly memorised specific words from training rather than learning the general rule — a real limitation, and a good next step would be training it on a bigger, more varied set of sentences.

## Running it

```bash
source selfinterp/bin/activate
python src/setup.py
python src/dataset.py
python src/probes.py
python src/adapter.py
```

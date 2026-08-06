# Self-Interpretation via Probes

This project checks whether GPT-2 keeps track of a simple grammatical fact — whether the subject of a sentence is singular or plural — somewhere inside its internal activations. First a small classifier is trained to read that fact straight out of the activations, layer by layer. Then a small adapter is trained so the model answers the question itself in words, and that self-report is compared against the classifier.

## What's here

- `src/setup.py` — loads GPT-2 and checks it runs.
- `src/dataset.py` — builds a small set of singular/plural example sentences, plus a separate set using different words to test generalisation.
- `src/probes.py` — trains a linear classifier per layer on GPT-2's activations to predict singular vs plural, and plots accuracy per layer for both the familiar-style sentences and the unseen ones.
- `src/adapter.py` — trains a small LoRA adapter so GPT-2 answers "singular or plural?" itself, and measures how accurate that self-report is, again on familiar and unseen sentences.

## What we found

`plots/probe_accuracy_ood.png` shows the classifier's accuracy at each layer. It reaches 100% accuracy on familiar sentences as early as layer 0, and 100% on sentences with completely new subject words by layer 10 — so the singular/plural information isn't just memorised vocabulary, it's genuinely and robustly readable from the model's activations, on words it never trained on.

The adapter tells a different story: it answers correctly about two-thirds of the time on sentences similar to what it trained on, but drops close to chance (about half right) on sentences with new words. Checking the model's raw answers confirms it's giving real answers ("singular" or "plural"), just often the wrong one on new words, not garbled output. So the information the probe reads out so easily is sitting right there in the model, but the model's own self-report didn't learn to use it — it partly memorised training words instead of learning the general rule. That gap, not the raw accuracy numbers, is the interesting result.

One caveat worth naming: singular vs plural is almost visible on the surface of the subject word itself (`cat` vs `cats`), so a linear classifier doing very well here is somewhat expected — it may be picking up a simple surface cue rather than deep syntax. A natural next step is testing a property that needs real sentence structure to get right, not just a visible "-s".

## Running it

```bash
source selfinterp/bin/activate
python src/setup.py
python src/dataset.py
python src/probes.py
python src/adapter.py
```

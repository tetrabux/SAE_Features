subjects_sing = ["The cat", "A student", "My brother", "The engineer", "That dog", "The teacher"]
subjects_plur = ["The cats", "Some students", "My brothers", "The engineers", "Those dogs", "The teachers"]
verbs = ["ran to the park", "read a book", "fixed the machine", "went home", "ate lunch", "solved the puzzle"]

texts, labels = [], []
for s in subjects_sing:
    for v in verbs:
        texts.append(f"{s} {v}."); labels.append(0)
for s in subjects_plur:
    for v in verbs:
        texts.append(f"{s} {v}."); labels.append(1)

print(len(texts), "sentences")
print(texts[0], labels[0])
print(texts[-1], labels[-1])


ood_subjects_sing = ["The child", "A doctor", "The manager", "That bird", "The scientist", "My friend"]
ood_subjects_plur = ["The children", "Some doctors", "The managers", "Those birds", "The scientists", "My friends"]
ood_verbs = ["walked away", "wrote a letter", "opened the door", "played music", "cleaned the room", "found a key"]

ood_texts, ood_labels = [], []
for s in ood_subjects_sing:
    for v in ood_verbs:
        ood_texts.append(f"{s} {v}."); ood_labels.append(0)
for s in ood_subjects_plur:
    for v in ood_verbs:
        ood_texts.append(f"{s} {v}."); ood_labels.append(1)
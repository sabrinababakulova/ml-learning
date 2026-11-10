# Step 6: the same MLP as mlp.py, written the way PyTorch code is normally written.
#
# What changes compared to mlp.py:
#   - the 5 loose tensors (embedding_table, hidden_weights, ...) → one class, NameModel(nn.Module)
#   - our hand-made random batches → DataLoader
#   - our hand-written update (parameter.data -= lr * parameter.grad) → the Adam optimizer
#   - the trained model is saved to a file, so we don't have to retrain to generate names
#
# What stays the same: the data, the architecture (3 letters → 10-number embeddings → 200 tanh → 27),
# the loss, and the generation loop. The val loss should end up close to mlp.py's (~2.1).

import os
import random

import torch
import torch.nn as nn                     # nn = "neural network": layers, models (like keras.layers)
import torch.nn.functional as F           # functions without weights: cross_entropy, softmax, ...
from torch.utils.data import DataLoader, TensorDataset   # `from x import a, b` = JS `import { a, b } from "x"`


# =====================================================================
# 1. DATA: exactly the same as mlp.py
# =====================================================================

names = open("data/names.txt").read().splitlines()

alphabet = sorted(set("".join(names)))
char_to_index = {char: index + 1 for index, char in enumerate(alphabet)}
char_to_index["."] = 0
index_to_char = {index: char for char, index in char_to_index.items()}
vocab_size = len(char_to_index)   # 27

block_size = 3   # how many previous letters the model sees


def build_dataset(names):
    contexts = []   # inputs: windows of 3 letter numbers
    targets = []    # outputs: the letter that comes next
    for name in names:
        context = [0] * block_size
        for char in name + ".":
            next_index = char_to_index[char]
            contexts.append(context)
            targets.append(next_index)
            context = context[1:] + [next_index]
    return torch.tensor(contexts), torch.tensor(targets)


# Shuffle with a fixed seed, then split 80 / 10 / 10 by name (same as mlp.py)
random.seed(42)
random.shuffle(names)
train_end = int(0.8 * len(names))
val_end = int(0.9 * len(names))

train_contexts, train_targets = build_dataset(names[:train_end])
val_contexts, val_targets = build_dataset(names[train_end:val_end])
test_contexts, test_targets = build_dataset(names[val_end:])

# torch.manual_seed fixes ALL of PyTorch's randomness at once: the starting weights,
# the DataLoader shuffle and the sampling. Before, we passed `generator=` everywhere;
# this is the simpler, more common way.
torch.manual_seed(42)


# =====================================================================
# 2. THE MODEL as an nn.Module class
# =====================================================================
#
# nn.Module is PyTorch's base class for models, like keras.Model.
# `class NameModel(nn.Module)` = JS `class NameModel extends nn.Module`.
#
# Every nn.Module has two parts:
#   __init__  → create the layers (runs once)                     ≈ JS constructor()
#   forward   → what happens to the input (runs on every batch)   ≈ the forward() function in mlp.py

class NameModel(nn.Module):

    # `self` is Python's `this`. Unlike JS, it's written out as the first parameter of every method.
    # n_embd=10 and n_hidden=200 are default values, like `n_embd = 10` in a JS function.
    def __init__(self, vocab_size, block_size, n_embd=10, n_hidden=200):
        super().__init__()   # call the parent's constructor first, like super() in JS. Always required.

        # nn.Embedding(27, 10) = a 27×10 table of learnable numbers, one row per token.
        # Replaces: embedding_table = torch.randn((vocab_size, embedding_size))
        self.embedding = nn.Embedding(vocab_size, n_embd)

        # nn.Linear(30, 200) = a weight matrix (30×200) + a bias (200), i.e. `x @ W + b`.
        # Replaces: hidden_weights and hidden_bias.
        # Unlike keras Dense(200), the INPUT size has to be written out: 3 letters × 10 numbers = 30.
        self.hidden = nn.Linear(block_size * n_embd, n_hidden)

        # 200 → 27 logits. Replaces: output_weights and output_bias.
        self.output = nn.Linear(n_hidden, vocab_size)

        # Every layer stored on `self` is registered automatically: model.parameters()
        # will find all 5 tensors inside them. No more `parameters = [...]` list by hand,
        # and no more `requires_grad = True`: nn layers turn it on themselves.

        # Same trick as `output_weights * 0.01` in mlp.py: start the output layer near zero,
        # so the first predictions are close to uniform (starting loss ≈ 3.30, not confidently wrong).
        # torch.no_grad() because we're setting starting values, not training.
        with torch.no_grad():
            self.output.weight *= 0.01
            self.output.bias.zero_()   # a trailing _ in PyTorch means "change this tensor in place"

    # The model's forward pass. Never call it directly: call model(x) instead,
    # and nn.Module runs forward(x) for you (plus some bookkeeping).
    def forward(self, contexts):                    # contexts: (batch, 3) letter numbers
        embeddings = self.embedding(contexts)       # (batch, 3, 10)  look up a row per letter
        flat = embeddings.view(contexts.shape[0], -1)   # (batch, 30) glue the 3 vectors together
        hidden = torch.tanh(self.hidden(flat))      # (batch, 200)   x @ W + b, then tanh
        return self.output(hidden)                  # (batch, 27)    logits


model = NameModel(vocab_size, block_size)

# Count the learnable numbers. Should be 11,897, the same as mlp.py: it's the same model.
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")   # :, adds thousands separators


# =====================================================================
# 3. DATALOADER: batches for us
# =====================================================================
#
# TensorDataset pairs inputs with targets: dataset[i] → (train_contexts[i], train_targets[i]).
# DataLoader cuts the dataset into batches of 32 and, with shuffle=True, reshuffles the order
# every epoch. Replaces our torch.randint(...) batch picking.
#
# One difference: randint picked random examples independently (some twice, some never).
# The DataLoader goes through EVERY example exactly once per epoch, in a new random order.

train_loader = DataLoader(TensorDataset(train_contexts, train_targets), batch_size=32, shuffle=True)

print(f"batches per epoch: {len(train_loader):,}")   # about 182,000 / 32 ≈ 5,700


# =====================================================================
# 4. OPTIMIZER: Adam
# =====================================================================
#
# The optimizer owns the update step. We give it the model's parameters once,
# and from then on:
#   optimizer.zero_grad() replaces: for p in parameters: p.grad = None
#   optimizer.step()      replaces: for p in parameters: p.data -= lr * p.grad
#
# Adam is smarter than our plain update: it keeps a running average of past gradients
# for each weight (momentum) and gives each weight its own step size. It usually trains
# faster with less tuning. Its normal learning rate is much smaller: 0.001, not 0.1.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 1e-3 = 0.001 (scientific notation, same as JS)


# =====================================================================
# 5. EVALUATION helper
# =====================================================================

@torch.no_grad()
def evaluate(contexts, targets):
    # model.eval() switches the model to "evaluation mode". This model has no layers that
    # behave differently in training (like Dropout), so it changes nothing here,
    # but it's the standard habit: eval() before measuring, train() before training.
    model.eval()
    logits = model(contexts)   # the whole set at once, not batches
    return F.cross_entropy(logits, targets).item()


# =====================================================================
# 6. THE TRAINING LOOP, standard PyTorch style
# =====================================================================
#
# An EPOCH = one full pass over the training data (~5,700 batches here).
# 10 epochs ≈ 57,000 steps. mlp.py did 200,000 steps, but Adam needs fewer.

epochs = 10

for epoch in range(epochs):
    model.train()   # "training mode", the partner of model.eval()

    # The DataLoader hands out (inputs, targets) pairs, one batch at a time.
    # In other people's PyTorch code these are often called xb / yb ("x batch", "y batch").
    for batch_contexts, batch_targets in train_loader:
        logits = model(batch_contexts)                    # forward pass (calls NameModel.forward)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad()   # clear old gradients (Common Mistake #4, now handled by the optimizer)
        loss.backward()         # backward pass: fill .grad for every parameter
        optimizer.step()        # update every parameter, Adam's way

    # After each epoch, measure on the FULL train and validation sets
    train_loss = evaluate(train_contexts, train_targets)
    val_loss = evaluate(val_contexts, val_targets)
    print(f"epoch {epoch + 1:2d} / {epochs}: train {train_loss:.4f} | val {val_loss:.4f}")


# =====================================================================
# 7. SAVE the trained model
# =====================================================================
#
# state_dict() is a dictionary of all the learned tensors, by name:
#   {"embedding.weight": ..., "hidden.weight": ..., "hidden.bias": ..., "output.weight": ..., ...}
# It holds the NUMBERS only, not the code. To load it later you first create a NameModel
# with the same settings, then pour the numbers in with model.load_state_dict(...).
# Same idea as model.save("x.keras") in Keras.

os.makedirs("models", exist_ok=True)   # create the folder if it's missing (like `mkdir -p`)
torch.save(model.state_dict(), "models/names.pt")
print("saved to models/names.pt")


# =====================================================================
# 8. GENERATE names (same loop as mlp.py, using the model)
# =====================================================================

@torch.no_grad()
def generate(temperature=1.0):
    model.eval()
    context = [0] * block_size     # start with "..."
    generated_chars = []
    while True:
        logits = model(torch.tensor([context]))             # (1, 3) in → (1, 27) out
        probs = F.softmax(logits / temperature, dim=1)
        next_index = torch.multinomial(probs, num_samples=1).item()
        if next_index == char_to_index["."]:
            break
        generated_chars.append(index_to_char[next_index])
        context = context[1:] + [next_index]
    return "".join(generated_chars)


print("\n--- 20 generated names ---")   # \n = an empty line first
for _ in range(20):
    print(generate())

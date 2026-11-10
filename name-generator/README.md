# name-generator

Read 32,000 real first names, learn which letter tends to come after which, then invent new names that sound real. It's a character-level language model in the style of Karpathy's *makemore*, and my first PyTorch project.

Unlike my earlier projects, this isn't classification. The model predicts one letter at a time, then feeds its own answer back in and predicts again, until it decides the name is finished. ChatGPT generates text with the same loop, only with word pieces instead of characters.

## The data

`names.txt` from [karpathy/makemore](https://github.com/karpathy/makemore): 32,033 English first names, one per line, all lowercase, 2 to 15 letters long. It isn't committed. Download it into `data/` first:

```bash
mkdir -p data
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

The vocabulary is the 26 letters plus a special `.` token, 27 in total. The `.` marks both the start and the end of a name (`"emma"` → `".emma."`), so the model learns how names begin and when to stop.

## What I built

Each file is a step further, and each one beats the one before it.

### 1. Bigram model by counting: [`main.py`](main.py), first half

No neural network at all. I count how often each letter follows each other letter in a 27×27 table, add 1 to every count (smoothing, so no pair has probability 0), and divide each row by its total to get probabilities. Generating a name means starting at `.` and sampling the next letter from that letter's row until `.` comes out again.

The model only ever sees **one** letter, so the names are mostly bad: it has no idea what it wrote two letters ago.

### 2. The same bigram model as a neural network: [`main.py`](main.py), second half

One-hot letters times a single 27×27 weight matrix, trained with gradient descent: forward pass, `loss.backward()`, and the update written by hand. It ends at the same loss as counting, and `softmax(weights)` ends up almost equal to the counted probability table. The network rediscovered the counts without ever seeing them.

### 3. MLP with 3 letters of context: [`mlp.py`](mlp.py)

Now the model sees the last 3 letters. Each letter gets a learned **embedding** of 10 numbers, the three embeddings are glued together (30 numbers), and they go through a hidden layer of 200 `tanh` neurons to 27 logits. This is the model from Bengio et al. 2003, *A Neural Probabilistic Language Model*.

- Names are shuffled and split 80 / 10 / 10 into train / validation / test **by name**, so pairs from one name never land in both train and validation.
- Training: 200,000 steps on random mini-batches of 32, written by hand with raw tensors, learning rate 0.1 and then 0.01 for the second half.
- Generation has a **temperature** setting: below 1 gives safer, more common names, above 1 gives wilder ones.
- A memorization check counts how many of 1,000 generated names already exist in the dataset.

### 4. The same MLP in standard PyTorch: [`mlp-nn.py`](mlp-nn.py)

The same model, the way PyTorch code is normally written: a `NameModel(nn.Module)` class with `nn.Embedding` and `nn.Linear`, a `DataLoader` for the batches, the Adam optimizer and 10 epochs. The trained weights are saved to `models/names.pt` (gitignored).

## Results

Loss is the average negative log-likelihood (cross-entropy) of the real next letter. Lower is better.

| Model | Loss |
|---|---|
| Random guessing (every token 1/27) | 3.30 |
| Bigram, counting | 2.45 |
| Bigram, neural network | ~2.46 |
| **MLP (`mlp.py`), train** | **2.04** |
| **MLP (`mlp.py`), validation** | **2.11** |

Three letters of context instead of one is the whole difference between 2.45 and 2.11. Train and validation are close, so the model learned what names sound like rather than memorising the training names. The `nn.Module` version should land close to the same numbers.

## Bugs I hit

- **A typo in a variable name** (`curentLettersAsANumber`). Python silently created a new variable, so the loop stopped right away and printed empty names.
- **`context` instead of `contexts`** in `build_dataset`: it returned one 3-letter window instead of the whole dataset. The crash only showed up 60 lines later, at `.view()`.
- **`for parameterss in parameters:`** while the loop body used `parameter`. Python loop variables live on after the loop, so it kept resetting only the last tensor's gradient. The other gradients piled up, and the loss climbed into the tens of thousands.

## Next

- A `generate.py` that loads `models/names.pt` and prints names without retraining
- Experiments, one setting at a time: embedding size, hidden size, block size
- Train on my own word list (for example Uzbek names or city names) and watch for overfitting
- Stretch goals: plot 2-number embeddings to see if the vowels cluster, top-k sampling, names that start with a given prefix

## Running it

```bash
cd name-generator
uv sync
mkdir -p data
curl -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
uv run python main.py      # bigram: counting, then the neural version
uv run python mlp.py       # MLP by hand (a few minutes of training)
uv run python mlp-nn.py    # MLP with nn.Module + Adam, saves models/names.pt
```

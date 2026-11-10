# ml-learning

My path from JavaScript/TypeScript developer to building and training language models myself.

The plan is to learn Python by doing machine learning, not by taking a Python course first. I build each idea by hand before I let a library do it for me. The goal at the end is a GPT written from an empty file, followed by real-world LLM work: fine-tuning, RAG and serving a model.

I spend about 8–10 hours a week on this, so the whole roadmap should take roughly 6–7 months (it started in October 2026).

## The roadmap

| # | Phase | Time | Done when |
|---|---|---|---|
| 1 | Python for a JS developer | 2 weeks | a bigram text generator in plain Python runs |
| 2 | NumPy and just enough math | 3 weeks | logistic regression with gradient descent in pure NumPy |
| 3 | Neural nets from scratch | 4 weeks | my own autograd engine + an MLP classifies MNIST |
| 4 | PyTorch | 3 weeks | a character-level language model trains |
| 5 | Transformers and my own GPT | 6 weeks | a GPT written from an empty file, trained on my own dataset |
| 6 | Real-world LLM work | 6+ weeks | a fine-tuned model served behind FastAPI + Next.js |

## Rules I follow

1. **Type the code, don't copy it.** Rebuilding from a blank file is where most of the learning happens.
2. **Each phase ends with something that runs.** If I can't build the project, the phase isn't finished.
3. **Learn math when I need it.** When something doesn't make sense, I stop and learn that one piece.
4. **Don't start at Phase 6.** Calling APIs and fine-tuning feel productive, but without Phases 3–5 I wouldn't be able to debug anything.

The code is commented heavily, often line by line, because I'm writing it to learn, not to ship it. The comments explain both the Python syntax (compared with JS) and the ML idea behind each step.

## Projects

### [`digits-recognition/`](digits-recognition/): handwritten digit recognition (in progress)

Look at a 28×28 picture of a handwritten digit from MNIST and say which digit (0–9) it is. This covers Phase 3 and the start of Phase 4.

I'm doing the TensorFlow version of the project:

1. **Load and prepare the data.** Scale pixels to 0–1 and split the images 50,000 train / 10,000 validation / 10,000 test.
2. **A network written by hand (784 → 128 ReLU → 10).** I create the weights as `tf.Variable`s, write the forward pass and the training loop myself, and let `tf.GradientTape` compute only the gradients. Then I check one batch's gradient against the backpropagation formula worked out by hand. Target: about 97–98 % accuracy.
3. **The same network in Keras** with `Sequential`, `compile` and `fit`, to see that `fit` runs the same loop I wrote.
4. **A convolutional network (CNN).** Target: about 99 % accuracy.
5. **Look at the mistakes.** A confusion matrix, plus the misclassified digits.
6. **Stretch goals:** test it on digits I draw myself, try dropout and data augmentation, and serve the model from FastAPI with a Next.js canvas to draw on.

The work happens in [`notebooks/recognition_logic.ipynb`](digits-recognition/notebooks/recognition_logic.ipynb).

### [`hotdog-not-hotdog/`](hotdog-not-hotdog/): hotdog / not hotdog (in progress)

Look at a photo and say whether it's a hotdog, like the app from Silicon Valley. It's a CNN trained from scratch in Keras on about 4,300 photos, with BatchNorm, early stopping and a learning rate that drops when progress stalls. It currently reaches about 94 % validation accuracy. The best model of each run is saved to its own file, and `predict.py` reuses it to classify new photos without retraining. See its [README](hotdog-not-hotdog/README.md) for the experiments so far and what's next.

### [`name-generator/`](name-generator/): a character-level name generator (in progress)

Read 32,000 real first names and invent new ones that sound real, one letter at a time. It's my first PyTorch project and covers Phase 4. It starts with a bigram model built by counting letter pairs (loss 2.45), rebuilds that same model as a neural network, then moves to an MLP that sees the last 3 letters through learned embeddings (validation loss 2.11). The MLP is written twice: once by hand with raw tensors, and once with `nn.Module`, `DataLoader` and Adam. See its [README](name-generator/README.md) for the details and results.

### Earlier: house price prediction

My first project (Phase 2) predicted California district house prices with linear regression written in NumPy and compared it with scikit-learn. It was a throwaway learning exercise, so it isn't kept in this repo.

## Running it

Each project is its own [`uv`](https://docs.astral.sh/uv/) environment:

```bash
cd digits-recognition
uv sync
uv run jupyter lab
```

The MNIST digits download automatically the first time they're used (`keras.datasets.mnist.load_data()`). The hotdog photos have to be cloned into `hotdog-not-hotdog/data/` first, and the names list downloaded into `name-generator/data/`, as their READMEs explain. Either way, `data/` isn't committed.

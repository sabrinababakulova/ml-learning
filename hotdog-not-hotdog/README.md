# hotdog-not-hotdog

The "hotdog / not hotdog" app from Silicon Valley: look at a photo and say whether it shows a hotdog. It's a binary image classifier: a convolutional network that I train from scratch in Keras, with no pretrained model.

## The data

[youngsoul/hotdog-not-hotdog-dataset](https://github.com/youngsoul/hotdog-not-hotdog-dataset) is a set of folders of `.jpg` photos, one folder per class:

| Folder | hot_dog | not_hot_dog | Used as |
|---|---|---|---|
| `train/` | 2302 | 1996 | training |
| `test/` | 200 | 200 | validation while I experiment |
| `holdout/` | 100 | 100 | the final test, run once at the end |

The photos come in many sizes (mostly around 500×375), so each one is resized to 128×128 while loading. The dataset is about 484 MB, so it isn't committed. Clone it into `data/` first:

```bash
git clone https://github.com/youngsoul/hotdog-not-hotdog-dataset.git data/hotdog-not-hotdog-dataset
```

## The model

All of it is in [`main.py`](main.py):

1. **Load** each folder with `keras.utils.image_dataset_from_directory`, which takes the labels from the folder names (`hot_dog` = 0, `not_hot_dog` = 1) and resizes the images.
2. **Rescale** pixels from 0–255 to 0–1 inside the model.
3. **Augment** the training images with random flips, rotations, zoom and contrast, so the model sees more variety.
4. **Three conv blocks** (32 → 64 → 128 filters, each followed by max pooling) learn patterns from edges up to "sausage in a bun".
5. **Global average pooling + dropout + one sigmoid output.** The output is the probability of "not hotdog".
6. **Train with early stopping**, which stops once validation loss hasn't improved for 10 epochs and keeps the best epoch.

## Results so far

About **84–85 % validation accuracy** (validation loss about 0.36), with training taking about 17 s per epoch on a MacBook CPU. Training accuracy (about 82 %) is a bit *below* validation accuracy, because augmentation and dropout only run during training. That means the model is underfitting, not overfitting, so the next step is to give it more room.

How I got here:

- I started on a much smaller Hugging Face dataset ([truepositive/hotdog_nothotdog](https://huggingface.co/datasets/truepositive/hotdog_nothotdog), 250 photos). There the model overfit, and the best I could get was about 80 %.
- I tested each improvement on top of the previous ones (early stopping, global average pooling, more augmentation, a third conv layer) and averaged over 3 seeds each. All four together gave the best and most consistent result.
- [`explore.py`](explore.py) is from that phase: it prints the small dataset's class balance and image sizes, and shows sample photos.

## Next experiments

- Lower dropout from 0.5 to 0.2.
- Add `ReduceLROnPlateau` to halve the learning rate when validation loss stops improving.
- A 4th conv block (256 filters), possibly with `BatchNormalization`.
- A bigger image size (160 or 224), which is slower but keeps more detail.
- Log each run with `CSVLogger` so runs are easy to compare.
- Run `model.evaluate(test_ds)` on the holdout set once, at the very end.

## Running it

```bash
cd hotdog-not-hotdog
uv sync
git clone https://github.com/youngsoul/hotdog-not-hotdog-dataset.git data/hotdog-not-hotdog-dataset
uv run python main.py
```

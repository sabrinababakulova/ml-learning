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
2. **Cache, then shuffle, then batch** the training images, so every epoch sees them in a new order and in differently mixed batches of 32. (Shuffling *before* `.cache()` would freeze the first epoch's order for every epoch after it. That was a bug in an earlier version.)
3. **Rescale** pixels from 0–255 to 0–1 inside the model.
4. **Four conv blocks** (32 → 64 → 128 → 128 filters). Each block is `Conv2D` → `BatchNormalization` → ReLU → max pooling, and together they learn patterns from edges up to "sausage in a bun".
5. **Global average pooling + one sigmoid output.** The output is the probability of "not hotdog".
6. **Train with three callbacks:**
   - `EarlyStopping` stops once validation loss hasn't improved for 10 epochs, and keeps the best epoch.
   - `ReduceLROnPlateau` halves the learning rate after 4 epochs without improvement.
   - `ModelCheckpoint` saves the best model of each run to its own file, `models/hotdog_<date>-<time>.keras`. If I stop training with Ctrl+C, the best model so far is already saved. Each run gets its own file because `save_best_only` only compares epochs within one run, so a shared file name let a new run overwrite a better model from an earlier one.

The augmentation layers (random flip, rotation, zoom and contrast) and `Dropout` are still in the code but commented out. My best run so far had both turned off.

## Using the trained model

Training takes 15–20 minutes, so I don't retrain just to classify a photo. [`predict.py`](predict.py) loads a saved model (in about a second) and classifies one or more photos. By default it uses the newest file in `models/`; `--model` picks a specific run:

```bash
uv run python predict.py ~/Downloads/some-photo.jpg another-photo.jpg
uv run python predict.py --model models/hotdog_20261027-0021.keras photo.jpg
# using model models/hotdog_20261027-0021.keras
# photo.jpg: HOTDOG (96% sure)
```

It resizes each photo to 128×128, the same as during training. It doesn't scale the pixels to 0–1, because the `Rescaling` layer is saved inside the model and does that itself.

The `.keras` file is self-contained: it holds the layers as well as the weights. The alternative, `model.save_weights()`, saves only the numbers, so loading them would also need the code that builds the model. Saved models go in `models/`, which is gitignored like `data/`, so each retrain doesn't add a new copy to the repo history.

## Results so far

Best run: `hotdog_20261027-0021`, with no augmentation and no dropout. Measured on the 400 validation images:

| | |
|---|---|
| Validation accuracy | **94.25 %** |
| Validation loss | 0.185 |
| Hotdogs recognised | 192 / 200 |
| Not-hotdogs recognised | 185 / 200 |
| Best epoch | about 35 |

Its median confidence is 98.5 % when it's right and 76 % when it's wrong, so most mistakes are cases where it was unsure. But 7 of its 23 mistakes were made with over 90 % confidence. The training accuracy gets close to 100 %, so the model has more or less memorised the training photos. Validation is still very good, so that hasn't hurt it much.

How I got here:

| Version | Validation accuracy | Validation loss |
|---|---|---|
| Small Hugging Face dataset (250 photos), 3 conv blocks | about 80 % | about 0.52 |
| This dataset, same model (3 conv blocks, dropout 0.5, augmentation) | 84–85 % | about 0.36 |
| + BatchNorm, 4th conv block, `ReduceLROnPlateau`, no dropout, shuffle fix | about 95 % | 0.186 |
| Same, with `Dropout(0.2)` | 90.25 % | 0.270 |
| Same, no dropout, no augmentation (current best) | **94.25 %** | **0.185** |

- I started on a much smaller Hugging Face dataset ([truepositive/hotdog_nothotdog](https://huggingface.co/datasets/truepositive/hotdog_nothotdog), 250 photos). There the model overfit, and I tested each improvement on top of the previous ones, averaged over 3 seeds. [`explore.py`](explore.py) is from that phase: it prints that dataset's class balance and image sizes, and shows sample photos.
- On this dataset the first model was underfitting: training accuracy was *below* validation. Giving it more capacity (BatchNorm and a 4th conv block) gave the biggest jump.
- Adding `Dropout(0.2)` made it worse in this experiment.

### What it can't do yet

I tested it on a screenshot of a delivery app showing three hotdogs:

- The **whole screenshot**: "not hotdog", 91 % sure. It learned what *photos of hotdogs* look like, not the idea of a hotdog. A screenshot is about 75 % white space, text and buttons, and squashing it to 128×128 shrinks the hotdogs to small flat shapes.
- **Each hotdog cropped out**: two of the three were recognised with 96 % and 99 % confidence.
- The **barbecue hotdog**, whose sausage is covered in dark sauce and peppers, was only 55 % "hotdog" even when cropped tightly. The chili dogs in the validation set are a similar weak spot.

So the 94 % holds for photos like the training photos. New kinds of images, like screenshots or heavily topped hotdogs, are a different test.

## Next experiments

- Turn augmentation back on, one layer at a time, and compare with the current best. A wider `RandomZoom` (zooming *out*) should help with small hotdogs in a bigger picture.
- `BatchNormalization(momentum=0.9)` to make validation loss jump around less in the early epochs.
- A bigger image size (160), which is slower but keeps more detail. `predict.py` needs the same `IMG_SIZE`.
- More varied training photos: hotdogs with lots of toppings, and studio shots on coloured backgrounds.
- Log each run with `CSVLogger` so runs are easy to compare.
- Run `model.evaluate(test_ds)` on the holdout set once, at the very end.

## Running it

```bash
cd hotdog-not-hotdog
uv sync
git clone https://github.com/youngsoul/hotdog-not-hotdog-dataset.git data/hotdog-not-hotdog-dataset
uv run python main.py                 # trains and saves models/hotdog_<date>-<time>.keras
uv run python predict.py photo.jpg    # classifies a photo with the newest saved model
```

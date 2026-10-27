# predict.py: ask a saved model whether a photo is a hotdog, without training again.
# train first with main.py (every run saves its own models/hotdog_<date-time>.keras), then run:
#   uv run python predict.py path/to/photo.jpg
# you can pass several photos at once:
#   uv run python predict.py photo1.jpg photo2.jpg
# by default it uses the NEWEST model in models/. to pick a specific one, put --model first:
#   uv run python predict.py --model models/hotdog_20261027-0915.keras photo.jpg

import glob  # finds files matching a pattern, like models/*.keras
import os
import sys  # sys.argv is the list of command-line arguments, like process.argv in Node
import keras
import numpy as np

IMG_SIZE = 128  # must be the same size the model was trained on in main.py

# sys.argv[0] is the script's own name ('predict.py'), so the real arguments start at index 1.
# [1:] is a slice: everything from index 1 to the end, like JS arr.slice(1)
args = sys.argv[1:]

if args and args[0] == '--model':
    # the user chose a model: take the path after --model, and the photos are everything after that
    model_path = args[1]
    photos = args[2:]
else:
    # no --model given: pick the most recently saved .keras file in models/
    # max(..., key=os.path.getmtime) is like JS files.reduce() keeping the one with the latest modified time
    model_path = max(glob.glob('models/*.keras'), key=os.path.getmtime)
    photos = args

print('using model', model_path)

# load the trained model from disk: layers + learned weights, ready to use.
# this replaces building the model and calling fit(), so it takes a second instead of ~20 minutes
model = keras.models.load_model(model_path)

for path in photos:
    # load the photo and resize it to 128x128, the same way main.py did while training.
    # the pixel scaling (0..255 -> 0..1) is NOT needed here, because the Rescaling layer is inside the model
    image = keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    pixels = keras.utils.img_to_array(image)  # -> numpy array, shape (128, 128, 3)

    # the model expects a BATCH of images, shape (None, 128, 128, 3), even if it's only one.
    # np.expand_dims adds that batch dimension at the front: (128, 128, 3) -> (1, 128, 128, 3)
    batch = np.expand_dims(pixels, axis=0)

    # predict() returns one number per image: the sigmoid output, the probability of "not_hot_dog" (label 1).
    # [0][0] takes the first image's first (and only) output, like result[0][0] in JS
    # verbose=0 hides the progress bar
    prob_not_hotdog = model.predict(batch, verbose=0)[0][0]

    # below 0.5 means the model leans towards label 0 = hot_dog
    if prob_not_hotdog < 0.5:
        print(f'{path}: HOTDOG ({1 - prob_not_hotdog:.0%} sure)')  # :.0% formats 0.93 as "93%"
    else:
        print(f'{path}: NOT HOTDOG ({prob_not_hotdog:.0%} sure)')

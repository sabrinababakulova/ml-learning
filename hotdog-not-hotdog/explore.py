# explore.py: look at the dataset before building a model.
# Run it with:  uv run python explore.py

# `from X import Y` is like JS `import { Y } from "X"`.
from collections import Counter  # Counter counts how many times each value appears
import numpy as np  # numpy: arrays of numbers (an image is just a grid of numbers)
import matplotlib.pyplot as plt  # matplotlib: draws charts and images in a window
from datasets import load_dataset  # Hugging Face library that downloads datasets

# Download (only the first time; after that it's cached on disk) and load the dataset.
# It comes back as a DatasetDict, which works like a JS object: { train: ..., validation: ... }
dataset = load_dataset('truepositive/hotdog_nothotdog')

# --- 1. The big picture --------------------------------------------------------
# print() is console.log(). Printing the DatasetDict shows the splits and the row counts.
print(dataset)

# Grab the training split. Square brackets work like JS: obj["key"].
train = dataset['train']

# .features describes the "columns": what type each field is.
# 'label' is a ClassLabel, so the numbers 0/1 map to names.
print('\nColumns:', train.features)

# The label names, e.g. ['hotdog', 'not_hotdog']. So 0 = hotdog, 1 = not_hotdog.
label_names = train.features['label'].names
print('Label names:', label_names)

# --- 2. How balanced are the classes? -----------------------------------------
# train['label'] gives the whole label column as a list, e.g. [0, 1, 1, 0, ...]
# Counter turns that into {0: 100, 1: 100}: how many of each.
# If one class had far more images, the model could "cheat" by always guessing it.
for split_name in dataset:  # looping over a dict gives its keys, like for...in in JS
    counts = Counter(dataset[split_name]['label'])
    # f"..." is a template string, like JS `...${x}...`
    print(f'{split_name}: {len(dataset[split_name])} images, per class: '
          f'{ {label_names[k]: v for k, v in counts.items()} }')

# --- 3. One example up close ---------------------------------------------------
# Indexing a split gives one row as a dict: {'image': <PIL Image>, 'label': 0}
example = train[0]
image = example['image']  # a Pillow (PIL) Image object
print('\nFirst example label:', example['label'], '=', label_names[example['label']])

# Pillow reports size as (width, height). mode 'RGB' means 3 colour channels.
print('PIL size (width, height):', image.size, '| mode:', image.mode)

# Turn the image into a numpy array. This is what a neural network actually sees.
# Its shape is (height, width, channels): rows of pixels, columns of pixels, then R, G, B.
pixels = np.array(image)
print('As numpy array, shape (height, width, channels):', pixels.shape)
print('Pixel value type:', pixels.dtype, '| min:', pixels.min(), '| max:', pixels.max())
# uint8 means whole numbers 0..255. Later you'll divide by 255 to get 0..1 for the model.

# --- 4. Are all images the same size? ------------------------------------------
# Unlike MNIST digits (always 28x28), these photos come in different sizes.
# A model needs one fixed input size, so this tells you you'll have to resize.
sizes = Counter(img.size for img in train['image'])  # a generator expression, like .map()
print(f'\nDistinct image sizes in train: {len(sizes)}')
print('Most common (width, height):', sizes.most_common(5))

# --- 5. Show some images -------------------------------------------------------
# Make a grid of 2 rows x 4 columns of small plots ("axes").
# `fig, axes = ...` is destructuring, like JS `const [fig, axes] = ...`
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# axes is a 2x4 grid; .flat lets you loop over it as one flat list of 8.
# enumerate() gives (index, item) pairs, like JS arr.forEach((item, i) => ...)
for i, ax in enumerate(axes.flat):
    row = train[i * 25]  # spread the picks out so you see both classes
    ax.imshow(row['image'])  # draw the image
    ax.set_title(f"{label_names[row['label']]}\n{row['image'].size}")  # label + size
    ax.axis('off')  # hide the x/y tick marks; they're just pixel numbers

plt.tight_layout()  # stop titles from overlapping
plt.show()  # open the window. The script waits here until you close it.

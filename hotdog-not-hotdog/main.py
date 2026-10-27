import os  # built-in module for files and folders (like Node's fs/path)
from datetime import datetime  # built-in date/time class, like JS Date
import keras
import tensorflow as tf

# the dataset is cloned from github into data/ (git clone https://github.com/youngsoul/hotdog-not-hotdog-dataset.git data/hotdog-not-hotdog-dataset)
# it's just folders of .jpg files, one folder per class:
#   train/   hot_dog/ (2302)  not_hot_dog/ (1996)   -> we learn from these
#   test/    hot_dog/ (200)   not_hot_dog/ (200)    -> we use these as VALIDATION while experimenting
#   holdout/ hot_dog/ (100)   not_hot_dog/ (100)    -> the final exam, used once at the very end
DATA_DIR = 'data/hotdog-not-hotdog-dataset'
IMG_SIZE = 128  # every image gets resized to 128x128 while loading (they're ~500x375 and all different sizes)
BATCH_SIZE = 32  # how many images the model sees before each weight update

# image_dataset_from_directory reads a folder like the one above and gives back a tf.data.Dataset:
# - the label comes from the folder name, sorted alphabetically: hot_dog = 0, not_hot_dog = 1
# - every image is resized to image_size (this replaces our old to_arrays function)
# - images are loaded from disk as needed, instead of all at once in memory
# label_mode='binary' gives labels shaped for our single sigmoid output
# batch_size=None: give back single images, not batches yet. we batch train_ds ourselves below, AFTER shuffling,
# so that every epoch the batches are made of a different mix of images
train_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/train', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary', batch_size=None)
# validation and test don't need shuffling: the order doesn't change the score, so they come in batches of 32 right away
val_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/test', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary', batch_size=BATCH_SIZE, shuffle=False)
test_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/holdout', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary', batch_size=BATCH_SIZE, shuffle=False)

print(train_ds.class_names)  # ['hot_dog', 'not_hot_dog'], which shows which folder became 0 and which became 1

# the order of these steps matters:
# .cache() keeps the decoded images in memory after the first epoch, so the jpgs are only read from disk once.
#   it stores them in the order it first saw them, so anything before .cache() happens only ONCE
# .shuffle(1000) comes AFTER .cache(), so it runs again every epoch: each epoch gets a new random order.
#   it keeps a pool of 1000 images and picks the next one at random from that pool
# .batch(BATCH_SIZE) groups the shuffled images into batches of 32, so the batches are different every epoch too
# .prefetch() prepares the next batch while the model is still training on the current one
train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)), # describe the input that model expects - IMG_SIZE x IMG_SIZE pixels, 3 channels (RGB). uses IMG_SIZE so changing it above can't break the model
    keras.layers.Rescaling(1 / 255), # multiply every pixel by 1/255 to normalize the pixel values to [0, 1] from [0, 255]
    # keras.layers.RandomFlip('horizontal'), #augmentation: randomly flip the image horizontally
    # keras.layers.RandomRotation(0.1), # more augmentation: randomly rotate the image by 10% of a full circle (up to about ±36°)
    # keras.layers.RandomZoom(0.2), # more augmentation: randomly zoom in or out by up to 20%
    # keras.layers.RandomContrast(0.2), # more augmentation: randomly make the image up to 20% more or less contrasty

    # each conv block is now: Conv2D (no activation) -> BatchNormalization -> ReLU -> MaxPooling2D
    # BatchNorm goes BEFORE the ReLU: it normalizes the raw conv outputs, then ReLU cuts the negatives
    # use_bias=False: BatchNorm subtracts the mean anyway, which would cancel out the bias, so the bias would be wasted weights
    keras.layers.Conv2D(32, 3, use_bias=False), #that's a convolutional layer with 32 filters, just like in that 2blue1brown video. 32 differnt filters, 3 is the window 3x3. 128x128 -> 126x126
    keras.layers.BatchNormalization(), # rescales each batch's outputs to about mean=0 and stddev=1, then LEARNS its own scale and shift (that's its params). keeps the numbers in a steady range, which makes training faster and more stable
    keras.layers.Activation('relu'), # ReLU as a separate layer, so it comes after BatchNorm: negatives -> 0
    keras.layers.MaxPooling2D(), # halves width and height, but keeps the largest value/signal in each 2x2 window. 126x126 -> 63x63

    keras.layers.Conv2D(64, 3, use_bias=False), #another convolutional layer, this time with 64 filters, and it works on 32 feature maps that came out of the previous layer. 63x63 -> 61x61
    keras.layers.BatchNormalization(), # same as above: normalize, then learned scale and shift
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(), # halves size again, from 61x61 to 30x30, but keeps the strongest signal in each 2x2 window

    keras.layers.Conv2D(128, 3, use_bias=False), # third convolutional layer, 128 filters: combines the 64 patterns into more complex ones (like a bun shape or a sausage texture). 30x30 -> 28x28
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(), # halves again, 28x28 -> 14x14

    keras.layers.Conv2D(128, 3, use_bias=False), # fourth convolutional layer, 128 filters: combines those into whole objects (like "sausage inside a bun"). 14x14 -> 12x12
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(), # halves again, 12x12 -> 6x6

    keras.layers.GlobalAveragePooling2D(), # instead of Flatten: averages each of the 128 feature maps (6x6) into ONE number, so we get 128 numbers instead of 4608 (6x6x128). far fewer weights in the Dense layer = less room to memorize
    # keras.layers.Dropout(0.2), # turned off for now. if training accuracy pulls well ahead of val_accuracy (overfitting), uncomment this first
    keras.layers.Dense(1, activation='sigmoid'), # final layer with a single neuron and sigmoid activation for binary classification
])
#ReLU stands for "Rectified Linear Unit". It's this one rule:
#relu(x) = max(0, x)        // JS: Math.max(0, x)
#Negative numbers become 0, and positive numbers pass through unchanged. For example, -3 → 0, 0 → 0, 2.5 → 2.5.

# - Rescaling always divides by 255.
# - RandomFlip, RandomRotation, RandomZoom and RandomContrast apply random transforms (only during training).
# - BatchNormalization keeps each layer's outputs in a steady range.
# - MaxPooling2D keeps the largest value in each 2×2 block.
# - GlobalAveragePooling2D averages each feature map into one number.
# print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# early stopping: after every epoch, check val_loss. if it hasn't hit a new best for 10 epochs in a row (patience), stop training.
# 10 instead of 20: with 400 validation images (not 40) val_loss is much less noisy, and each epoch takes longer
# restore_best_weights=True rolls the model back to its best epoch, not the last one
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# reduce learning rate on plateau: if val_loss hasn't improved for 4 epochs, halve the learning rate (factor=0.5).
# smaller steps can get the model past a plateau where big steps just bounce around the best point.
# patience=4 is lower than early stopping's 10, so the learning rate gets halved about twice before training gives up
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

# every run gets its own file name, made from the date and time it started, e.g. models/hotdog_20261027-0915.keras
# save_best_only only compares epochs WITHIN one run, so with one shared file name, a new run's first epoch
# would overwrite the best model from an earlier run. with a name per run, experiments never overwrite each other.
# strftime turns the date into text: %Y = year, %m = month, %d = day, %H = hour, %M = minute
run_name = datetime.now().strftime('%Y%m%d-%H%M')
model_path = f'models/hotdog_{run_name}.keras'
print('this run saves to', model_path)

# checkpoint: every time val_loss hits a new best, save the whole model to model_path
# (the layers, the learned weights and the optimizer state, all in one file)
# so even if you stop training with Ctrl+C halfway, the best model so far is already on disk.
# predict.py loads this file to classify new photos without training again
os.makedirs('models', exist_ok=True)  # create the models/ folder if it doesn't exist yet (like mkdir -p)
checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

# epochs=100 is just an upper limit: early stopping ends training once val_loss stops improving
# a tf.data.Dataset already contains both images and labels, so we pass train_ds instead of x_train, y_train
model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[early_stop, reduce_lr, checkpoint])
print('best model of this run saved to', model_path)

# the holdout set is the "final exam": run this only once, when you're done changing the model.
# if you check it after every change, you end up tuning the model to these 200 images too
# model.evaluate(test_ds)

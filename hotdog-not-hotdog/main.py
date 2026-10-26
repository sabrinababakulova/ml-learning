import keras
import tensorflow as tf

# the dataset is cloned from github into data/ (git clone https://github.com/youngsoul/hotdog-not-hotdog-dataset.git data/hotdog-not-hotdog-dataset)
# it's just folders of .jpg files, one folder per class:
#   train/   hot_dog/ (2302)  not_hot_dog/ (1996)   -> we learn from these
#   test/    hot_dog/ (200)   not_hot_dog/ (200)    -> we use these as VALIDATION while experimenting
#   holdout/ hot_dog/ (100)   not_hot_dog/ (100)    -> the final exam, used once at the very end
DATA_DIR = 'data/hotdog-not-hotdog-dataset'
IMG_SIZE = 128  # every image gets resized to 128x128 while loading (they're ~500x375 and all different sizes)

# image_dataset_from_directory reads a folder like the one above and gives back a tf.data.Dataset:
# - the label comes from the folder name, sorted alphabetically: hot_dog = 0, not_hot_dog = 1
# - every image is resized to image_size (this replaces our old to_arrays function)
# - images are grouped into batches of 32 and loaded from disk as needed, instead of all at once in memory
# label_mode='binary' gives labels shaped for our single sigmoid output
train_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/train', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary',
    shuffle=True, seed=42)  # shuffle the training images (they'd otherwise come in folder order: all hotdogs first)
val_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/test', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary', shuffle=False)
test_ds = keras.utils.image_dataset_from_directory(
    f'{DATA_DIR}/holdout', image_size=(IMG_SIZE, IMG_SIZE), label_mode='binary', shuffle=False)

print(train_ds.class_names)  # ['hot_dog', 'not_hot_dog'], which shows which folder became 0 and which became 1

# .cache() keeps the decoded images in memory after the first epoch, so the jpgs are only read from disk once
# .prefetch() prepares the next batch while the model is still training on the current one
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)), # describe the input that model expects - 128x128 pixels, 3 channels (RGB)
    keras.layers.Rescaling(1 / 255), # multiply every pixel by 1/255 to normalize the pixel values to [0, 1] from [0, 255]
    keras.layers.RandomFlip('horizontal'), #augmentation: randomly flip the image horizontally
    keras.layers.RandomRotation(0.1), # more augmentation: randomly rotate the image by 10% of a full circle
    keras.layers.RandomZoom(0.2), # more augmentation: randomly zoom in or out by up to 20%
    keras.layers.RandomContrast(0.2), # more augmentation: randomly make the image up to 20% more or less contrasty
    keras.layers.Conv2D(32, 3, activation='relu'), #that's a convolutional layer with 32 filters, just like in that 2blue1brown video. 32 differnt filters, 3 is the window 3x3
    keras.layers.MaxPooling2D(), # halves width and height, but keeps the largest value/signal in each 2x2 window,
    keras.layers.Conv2D(64, 3, activation='relu'), #another convolutional layer, this time with 64 filters, and it works on 32 feature maps that came out of the previous layer
    keras.layers.MaxPooling2D(), # halves size again, from 61x61 to 30x30, but keeps the strongest signal in each 2x2 window
    keras.layers.Conv2D(128, 3, activation='relu'), # third convolutional layer, 128 filters: combines the 64 patterns into even more complex ones (like "sausage inside a bun"). 30x30 -> 28x28
    keras.layers.MaxPooling2D(), # halves again, 28x28 -> 14x14
    keras.layers.GlobalAveragePooling2D(), # instead of Flatten: averages each of the 128 feature maps (14x14) into ONE number, so we get 128 numbers instead of 25088. far fewer weights in the Dense layer = less room to memorize
    keras.layers.Dropout(0.5), # randomly sets 50% of the neurons to zero during training to prevent overfitting, so that the model doesn't memorize the training data and can generalize better to new data
    keras.layers.Dense(1, activation='sigmoid'), # final layer with a single neuron and sigmoid activation for binary classification
])
#ReLU stands for "Rectified Linear Unit". It's this one rule:
#relu(x) = max(0, x)        // JS: Math.max(0, x)
#Negative numbers become 0, and positive numbers pass through unchanged. For example, -3 → 0, 0 → 0, 2.5 → 2.5.

# - Rescaling always divides by 255.
# - RandomFlip and RandomRotation apply random transforms.
# - MaxPooling2D keeps the largest value in each 2×2 block.
# - Flatten reshapes the data.
# - Dropout randomly zeros values during training.
# print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# early stopping: after every epoch, check val_loss. if it hasn't hit a new best for 10 epochs in a row (patience), stop training.
# 10 instead of 20: with 400 validation images (not 40) val_loss is much less noisy, and each epoch takes longer
# restore_best_weights=True rolls the model back to its best epoch, not the last one
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# epochs=100 is just an upper limit: early stopping ends training once val_loss stops improving
# a tf.data.Dataset already contains both images and labels, so we pass train_ds instead of x_train, y_train
model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[early_stop])

# the holdout set is the "final exam": run this only once, when you're done changing the model.
# if you check it after every change, you end up tuning the model to these 200 images too
# model.evaluate(test_ds)

def main():
    print("Hello from hotdog-not-hotdog!") 
        

if __name__ == "__main__":
    main()

from random import randrange

import tensorflow as tf
import matplotlib.pyplot as plt

# clothing categories for fashion dataset
categories = [(0, "T-shirt/top"), (1, "Trouser"), (2, "Pullover"), (3, "Dress"), (4, "Coat"),
           (5, "Sandal"), (6, "Shirt"), (7, "Sneaker"), (8, "Bag"), (9, "Ankle boot")]

# load the fashion dataset
# train for training, validation for testing
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# create the model
number_of_classes = train_labels.max() + 1
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes)
])
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)
model.predict(train_images[0:10])

# choose random image
data_idx = randrange(1, 59999)
plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

# predict correct answer
x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1]).flatten())
plt.xticks(range(10))
plt.show()

# determine correct answer
correct_answer_number = train_labels[data_idx]
for category_number, category_label in categories:
    if category_number == correct_answer_number:
        correct_answer = category_label
        break
print("correct answer:", category_label)
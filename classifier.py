import tensorflow as tf
import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from metrics import show_metrics

train_dataset = tf.keras.utils.image_dataset_from_directory('images/train', label_mode='categorical',labels='inferred', image_size=(480,640))
train_dataset = train_dataset.map(lambda x,y: (x/255, y))

val_dataset = tf.keras.utils.image_dataset_from_directory('images/val', label_mode='categorical',labels='inferred', image_size=(480,640))
val_dataset = val_dataset.map(lambda x,y: (x/255, y))

test_dataset = tf.keras.utils.image_dataset_from_directory('images/test', label_mode='categorical',labels='inferred', image_size=(480,640))
test_dataset = test_dataset.map(lambda x,y: (x/255, y))


model = Sequential()
model.add(Conv2D(32, (3,3), 1, activation='relu', input_shape=(480,640,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(128, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.CategoricalAccuracy(name='accuracy'),
]

model.compile('adam', loss='categorical_crossentropy', metrics=METRICS)

model.fit(train_dataset, epochs=5, validation_data=val_dataset)

y_pred = []  # store predicted labels
y_true = []  # store true labels

for images,labels in test_dataset:
    labels_t = tf.argmax(labels,axis=1)
    y_true.append(labels_t)
    y_pred.append(tf.argmax(model.predict(images), axis= 1))

show_metrics(y_true, y_pred)

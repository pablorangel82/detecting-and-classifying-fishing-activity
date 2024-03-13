import keras as keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def cnn_training(train_dataset, val_dataset, number_classes):
    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min',restore_best_weights=True)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(512, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(number_classes, activation='softmax'))

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        # keras.metrics.F1Score(name='f1_score'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.Precision(name='precision')
    ]

    model.compile('adam', loss='categorical_crossentropy', metrics=METRICS)

    model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=callback)
    return model

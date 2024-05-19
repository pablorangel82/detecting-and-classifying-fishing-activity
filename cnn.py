import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


class CNN:

    def __init__(self, iterations = 30, train_dataset=None, val_dataset=None, number_of_classes=None, model_path=None):
        if train_dataset and val_dataset and number_of_classes is not None:
            self.iterations = iterations
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.number_of_classes = number_of_classes
            self.model = None
            self.build()
        else:
            self.model = tensorflow.keras.models.load_model(model_path)
            self.model.summary()

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, (3, 3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(128, (3, 3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(256, (3, 3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(512, (3, 3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(self.number_of_classes, activation='softmax'))

        METRICS = [
            tensorflow.keras.metrics.TruePositives(name='tp'),
            tensorflow.keras.metrics.FalsePositives(name='fp'),
            tensorflow.keras.metrics.TrueNegatives(name='tn'),
            tensorflow.keras.metrics.FalseNegatives(name='fn'),
            tensorflow.keras.metrics.CategoricalAccuracy(name='accuracy'),
            # keras.metrics.F1Score(name='f1_score'),
            tensorflow.keras.metrics.Recall(name='recall'),
            tensorflow.keras.metrics.Precision(name='precision')
        ]

        self.model.compile('adam', loss='categorical_crossentropy', metrics=METRICS)

    def train(self):
        callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.iterations, verbose=0,
                                                            mode='min', restore_best_weights=True)
        self.model.fit(self.train_dataset, epochs=self.iterations, validation_data=self.val_dataset, callbacks=callback)

    def save(self):
        self.model.save_weights("output/weights")
        self.model.save("output/model.keras")
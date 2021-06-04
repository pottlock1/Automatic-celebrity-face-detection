from keras import models
from keras import layers



class Model:
    def __init__(self, input_shape, no_labels):
        self.input_shape = input_shape
        self.no_labels = no_labels

    def build(self):
        model = models.Sequential()
        model.add(layers.Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.no_labels, activation='softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
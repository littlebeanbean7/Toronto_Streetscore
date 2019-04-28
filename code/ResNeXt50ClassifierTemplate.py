import os
import keras
from keras_applications.resnext import ResNeXt50
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ResNeXt50ClassifierTemplate:
    def __init__(self, learning_rate=0.0001, num_classes=2):
        self.model_name = "ResNeXt50"
        self.learning_rate = learning_rate
        self.img_width = 224
        self.img_height = 224
        self.num_classes = num_classes

    def create_model(self):
        ResNeXt50_notop = ResNeXt50(include_top=False,
                                      weights='imagenet', input_tensor=None,
                                      input_shape=(self.img_height, self.img_width, 3),
                                      backend = keras.backend, layers = keras.layers,
                                      models = keras.models, utils = keras.utils)
        output = ResNeXt50_notop.get_layer(index=-1).output  # Shape: (7, 7, 2048)
        output = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.num_classes, activation='softmax', name='predictions')(output)
        ResNeXt50_model = Model(ResNeXt50_notop.input, output)
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        ResNeXt50_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return ResNeXt50_model

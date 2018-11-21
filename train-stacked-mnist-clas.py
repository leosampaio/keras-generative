from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras import layers, optimizers
import keras.backend as K
import keras
import os

from datasets import load_dataset

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(1)
    set_session(tf.Session(config=config))

classifier_model_name = 'mnist_mode_count_classifier.h5'
dataset = load_dataset('stacked-mnist')

input = Input(shape=(32, 32, 3))
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
# encoded = Flatten()(encoded)
# encoded = Dense(512, activation='relu')(encoded)
# pred = Dense(1000, activation='sigmoid')(encoded)
# model = Model(input, pred)
base_model = MobileNet(weights=None, alpha=0.25, include_top=False)

shape = (1, 1, int(1024*0.25))
x = base_model(input)
x = layers.Reshape(shape, name='reshape_1')(x)
x = layers.Conv2D(1000, (1, 1),
                  padding='same',
                  name='conv_preds')(x)
x = layers.Activation('softmax', name='act_softmax')(x)
x = layers.Reshape((1000,), name='reshape_2')(x)
model = Model(input, x)

test_x, test_y = dataset.get_random_perm_of_test_set(n=64)
test_y = keras.utils.to_categorical(test_y, num_classes=1000)
if not os.path.exists(classifier_model_name):
    # model = MobileNet(weights='imagenet', include_top=True)
    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='categorical_crossentropy')
    for e in range(0, 100):
        for x_batch, y_batch, batch_index in dataset.generator(batchsize=64):
            loss = model.train_on_batch(x_batch, y_batch)
            evaluation = model.evaluate(test_x, test_y, batch_size=64, verbose=0)
            print('[Metrics] Training Classifier for Mode Estimation... E{}/100: loss = {}, test_loss = {}'.format(e, loss, evaluation), end='\r')
        print('[Metrics] Training Classifier for Mode Estimation... E{}/100: loss = {}'.format(e, loss))
    model.save(classifier_model_name)
else:
    model = MobileNet(weights=classifier_model_name, include_top=True)
'''
使用RNN进行图像的分类就是个many to one的问题. 对于mnist来说其图像的size是28*28，
如果将其看成28个step，每个step的size是28的话，刚好符合.
当我们得到最终的输出的时候将其做一次线性变换就可以加softmax来分类了，其实挺简单的.
'''

import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10


def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data(path='./data/mnist.npz')   # mnist.load_data()
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, n_step, n_input),
               unroll=True))

model.add(Dense(n_classes))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=training_iters,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])

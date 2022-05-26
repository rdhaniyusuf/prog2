from doctest import OutputChecker
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
def normalize_data(x,y):
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    y =  to_categorical(y)
    return x,y

def predict(model,x,y):
    (loss,accuracy) = model.evaluate(x,y, verbose=0)
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)
    y = np.argmax(y, axis=1)
    print(confusion_matrix(y, pred))
    print(accuracy*100)

def plot_bpn(history):
    figure, axis = plt.subplots(1)
    axis.plot(history.history['accuracy'])
    axis.plot(history.history['loss'])
    axis.set_title('Accuracy and Loss')
    axis.legend(['accuracy','loss'], loc='upper left')
    axis.set(xlabel = 'epoch',ylabel='accuracy')
    plt.show()
            
def compute(x,y, n_inputs, n_hidden,n_output,n_epochs, lr, path=None):
    x, y = normalize_data(x,y)
    model = Sequential()

    model.add(Dense(n_hidden,input_shape=(6,)))
    model.add(Activation("sigmoid"))
    model.add(Dense(n_output))
    model.add(Activation("sigmoid"))

    sgd = SGD(learning_rate=lr)
    model.compile(loss="MSE", optimizer=sgd,metrics=["accuracy"])
    history = model.fit(x,y, epochs=n_epochs, batch_size=3, verbose=0)
    if path is not None:
        model.save(path)
    # model.summary()
    print('Learning rate : ' + str(lr) + ', Hidden Layer : ' + str(n_hidden))
    predict(model, x,y)
    plot_bpn(history)

class ModelBackprop(object):
	def __init__(self, x=None, y=None, n_hidden=0, lr=0, path = None):
		self.lr = lr
		self.n_output = 7
		self.n_epoch = 1000
		self.n_hidden = n_hidden
		self.model = Sequential()
		self.sgd = SGD(learning_rate=lr)
		
		self.init()

		if x is not None and y is not None:
			x, y = normalize_data(x, y)
			self.compute(x, y)
		if path is not None:
			self.saved(path)

	def init(self):
		self.model.add(Dense(self.n_hidden, input_shape=(6,)))
		self.model.add(Activation("sigmoid"))
		self.model.add(Dense(self.n_output))
		self.model.add(Activation("sigmoid"))

	def compute(self, x, y):
		self.model.compile(loss="MSE", optimizer=self.sgd, metrics=['accuracy'])
		self.history = self.model.fit(x, y, epochs=self.n_epoch, batch_size=3, verbose=1)
	
	def predict(self,x):
		x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
		return self.model.predict(x)

	def saved(self, path):
		self.model.save(path)

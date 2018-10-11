from util import load_training_data, format_training_data
from models import cnn
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import set_random_seed
from keras import optimizers, losses

np.random.seed(0)
set_random_seed(0)

pairs, labels = load_training_data('./training_data')
train_x = format_training_data(pairs)
model = cnn((len(train_x[0]), len(train_x[0][0]), len(train_x[0][0][0])))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
print(history.history['acc'])
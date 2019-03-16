#855
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

df = pd.read_csv('clean_data_35_notencoded.csv')

Y=df.iloc[:, 6].values

X = pd.read_csv('encoded.csv')

Y.shape

X.drop(['Unnamed: 0'], inplace = True, axis = 1)


s = pd.Series(Y)

df2 = pd.concat([X, s], axis = 1)

df2 = shuffle(df2)

Y = df2[0]

df2.drop(0, axis = 1, inplace = True)
X = df2

Y = pd.get_dummies(Y)

print("Data Prepared...")

model = Sequential()
model.add(Dense(1000, input_shape = (315,)))
model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dense(250, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dense(35, activation='softmax'))

print("Compiling Model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

x_train = x_train.values
x_test = x_test.values

# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

print("Training...")
model.fit(x=x_train, y=y_train, epochs=200, validation_data=(x_test, y_test), shuffle=False)

print("Saving...")
fname = "model"
model_json = model.to_json()
with open(fname + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(fname + ".h5")
print("Model Saved.")

#855
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('clean_data_35_notencoded.csv')

Y=df.iloc[:, 6].values

X = pd.read_csv('encoded.csv')

Y = pd.get_dummies(Y)

Y.shape

X.drop(['Unnamed: 0'], inplace = True, axis = 1)


print("Data Prepared...")


model = Sequential()
model.add(Dense(1000, activation='relu',input_shape = (315, )))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(35, activation='softmax'))

print("Compiling Model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

x_train = x_train.values

print("Training...")
model.fit(x=x_train, y=y_train, epochs=1000, validation_data=(x_test, y_test))

print("Saving...")
fname = "model"
model_json = model.to_json()
with open(fname + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(fname + ".h5")
print("Model Saved.")

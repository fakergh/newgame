import pandas as pd

# Load the dataset from a CSV file

df = pd.read_csv('Boston.csv')

df
df = df.drop('Unnamed: 0', axis=1)

from sklearn.preprocessing import StandardScaler

X = df.drop('medv', axis=1)

y = df['medv']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Training set shape:', X_train.shape, y_train.shape)

print('Testing set shape:', X_test.shape, y_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(64, input_dim=13, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(1))

# Display the model summary

print(model.summary())

# Compile the model

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

from keras.callbacks import EarlyStopping

# Train the model

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# Plot the training and validation loss over epochs

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Training', 'Validation'])

plt.show()
# Evaluate the model on the testing set

loss, mae = model.evaluate(X_test, y_test)

# Print the mean absolute error

print('Mean Absolute Error:', mae)

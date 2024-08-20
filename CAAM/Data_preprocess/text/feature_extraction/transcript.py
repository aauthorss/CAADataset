from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout


def load_model():
    X = Input(shape=(400, 512,))
    X_gender = Input(shape=(1,))

    Y = LSTM(10, name='lstm_cell')(X)
    Y = Dropout(rate=0.2)(Y)
    Y = Concatenate(axis=-1)([Y, X_gender])
    Y = Dense(15, activation='relu')(Y)
    Y = Dropout(rate=0.2)(Y)
    Y = Dense(1, activation=None)(Y)

    model = Model(inputs=[X, X_gender], outputs=Y)
    return model


if __name__ == "__main__":
    m = load_model()

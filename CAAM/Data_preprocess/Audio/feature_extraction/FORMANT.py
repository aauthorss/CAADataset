from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Dropout, CuDNNGRU


def load_model():
    X = Input(shape=(20000, 5,))
    X_gender = Input(shape=(1,))

    Y = CuDNNGRU(10, name='lstm_cell')(X)
    Y = Dropout(rate=0.25)(Y)
    Y = Concatenate(axis=-1)([Y, X_gender])
    Y = Dense(6, activation='relu')(Y)
    Y = Dropout(rate=0.25)(Y)
    Y = Dense(1, activation=None)(Y)

    model = Model(inputs=[X, X_gender], outputs=Y)

    return model


if __name__ == "__main__":
    m = load_model()

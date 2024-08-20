from keras.models import Model
from keras.layers import Dense, Input, Dropout, GRU


def load_model():
    X = Input(shape=(10000, 205))

    Y = GRU(10, name='lstm_cell')(X)
    Y = Dropout(rate=0.3)(Y)

    Y = Dense(20, activation='relu')(Y)
    Y = Dropout(rate=0.3)(Y)

    Y = Dense(1, activation=None)(Y)

    model = Model(inputs=X, outputs=Y)
    return model

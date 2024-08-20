from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Dropout, Add, Lambda


def load_model():
    FORMANT = Input(shape=(10,))
    facial = Input(shape=(10,))
    transcript = Input(shape=(10,))
    X_gender = Input(shape=(1,))
    FORMANT_shortened = Dense(450, activation='relu')(FORMANT)

    facial_shortened = Dense(600, activation='relu')(facial)
    facial_shortened = Dense(450, activation='relu')(facial_shortened)

    transcript_elongated = Dense(315, activation='relu')(transcript)
    transcript_elongated = Dense(450, activation='relu')(transcript_elongated)

    B = Concatenate(axis=1)([FORMANT_shortened, facial_shortened, transcript_elongated])

    P = Dense(300, activation='tanh')(B)

    alpha = Dense(3, activation='softmax')(P)

    F = Lambda(lambda x: alpha[:, 0:1] * FORMANT_shortened + alpha[:, 1:2] * facial_shortened + alpha[:, 2:3] * transcript_elongated)(alpha)

    Y = Concatenate(axis=-1)([F, X_gender])

    Y = Dense(210, activation='relu')(Y)
    Y = Dropout(rate=0.5)(Y)

    Y = Dense(83, activation='relu')(Y)
    Y = Dropout(rate=0.3)(Y)
    Y = Dropout(rate=0.3)(Y)

    Y = Dense(1, activation=None)(Y)

    model = Model(inputs=[FORMANT, facial, X_gender, transcript], outputs=Y)
    return model


if __name__ == "__main__":
    m = load_model()

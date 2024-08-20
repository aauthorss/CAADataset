import time

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import keras
from keras.preprocessing import sequence
from sklearn import preprocessing, metrics
from Data_preprocess.Audio.feature_extraction.FORMANT import load_model as load_audio_feature_model
from Data_preprocess.text.feature_extraction.transcript import load_model as load_text_feature_model
from Data_preprocess.Video.feature_extraction.visual_features import load_model as load_video_feature_model
from multimodal.attention_fusion_network.load_model import load_model as load_main_model

config = {
    "encoder_path": "../../Data_preprocess/text/universal-sentence-encoder-large_3",
    "data_path": "../../Data/small_caa"
}


def preprocess(id_list: list):
    # TODO 注意内存占用
    embed = hub.Module(config["encoder_path"])

    audio_list = []
    text_list = []
    video_list = []

    start_time = time.time()
    for ID in id_list:
        # audio part
        audio_data = pd.read_csv(f'{config["data_path"]}/{ID}_P/{ID}_formant.csv', header=None).values
        a = np.arange(audio_data.shape[0])
        audio_data = preprocessing.scale(audio_data)
        audio_data = audio_data[a % 10 == 0]
        audio_list.append(sequence.pad_sequences(audio_data.T, maxlen=20000, dtype='float32', padding='pre').T.tolist())

        # video part
        video_data = pd.read_csv(f'{config["data_path"]}/{ID}_P/{ID}_feature3D.txt')
        video_data = video_data[(video_data['timestamp'] * 10) % 2 == 0][:]
        video_data = video_data[video_data['success'] == 1][:].values
        video_data = video_data[:, 4:]
        video_data = preprocessing.scale(video_data)
        video_list.append(sequence.pad_sequences(video_data.T, maxlen=10000, dtype='float32', padding='pre').T.tolist())

    print(f"audio and video preprocess finished, elapsed time: {time.time() - start_time:.2f}s")

    # text part
    start_time = time.time()
    with tf1.Session() as session:
        with tf.device('/cpu:0'):
            session.run([tf1.global_variables_initializer(), tf1.tables_initializer()])
            for ID in id_list:
                transcript = pd.read_csv(f'{config["data_path"]}/{ID}_P/{ID}_Transcript.csv')
                transcript = transcript[transcript['speaker'] == 'Teacher'].values
                transcript = transcript[:, 3]
                x = session.run(embed(transcript))
                tmp_text_data = sequence.pad_sequences(x.T, maxlen=400, dtype='float32', padding='pre').T.tolist()
                text_list.append(tmp_text_data)

    print(f"text preprocess finished, elapsed time: {time.time() - start_time:.2f}s")

    audio_list = np.array(audio_list)
    text_list = np.array(text_list)
    video_list = np.array(video_list)

    # gender part
    result_csv = pd.read_csv(f'{config["data_path"]}/test.csv')
    result_csv.set_index('Lesson_ID', inplace=True)
    labels = result_csv['Atmosphere-Score'][id_list].values.tolist()
    genders = result_csv['Gender'].values

    # audio feature extract
    start_time = time.time()
    FORMANT_features_model = load_audio_feature_model()
    FORMANT_features_extractor = keras.models.Model(inputs=FORMANT_features_model.inputs,
                                                    outputs=FORMANT_features_model.layers[1].output)
    encoded_audio = FORMANT_features_extractor.predict([audio_list, genders], batch_size=audio_list.shape[0])
    keras.backend.clear_session()
    print(f"audio feature extract finished, elapsed time: {time.time() - start_time:.2f}s")

    # text feature extract
    start_time = time.time()
    model = load_text_feature_model()
    new_model = keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
    encoded_text = new_model.predict([text_list, genders])
    keras.backend.clear_session()
    print(f"text feature extract finished, elapsed time: {time.time() - start_time:.2f}s")

    # video feature extract
    start_time = time.time()
    visual_features_model = load_video_feature_model()
    visual_features_extractor = keras.models.Model(inputs=visual_features_model.inputs,
                                                   outputs=visual_features_model.layers[1].output)
    encoded_video = visual_features_extractor.predict(video_list, batch_size=video_list.shape[0])
    keras.backend.clear_session()
    print(f"text feature extract finished, elapsed time: {time.time() - start_time:.2f}s")

    return encoded_audio, encoded_text, encoded_video, genders, labels


if __name__ == "__main__":
    start_time = time.time()
    sample_num = 1
    test_id_list = [1000 + i for i in range(1, sample_num + 1)]
    test_FORMANT, test_transcript, test_facial, test_X_gender, test_Y = preprocess(test_id_list)
    print(f"preprocess finished, elapsed time: {time.time() - start_time:.2f}s")

    start_time = time.time()
    model = load_main_model()
    model.load_weights('optimal_weights.h5')
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    test_Y_hat = model.predict([test_FORMANT, test_facial, test_X_gender, test_transcript])
    print(f"Predict finished, elapsed time: {time.time() - start_time:.2f}s")

    test_Y = np.array(test_Y)
    test_Y_hat = test_Y_hat.reshape((test_Y.shape[0],))
    print(test_Y)
    print([int(i) for i in test_Y_hat])

RMSE = np.sqrt(metrics.mean_squared_error(test_Y, test_Y_hat))
MAE = metrics.mean_absolute_error(test_Y, test_Y_hat)
EVS = metrics.explained_variance_score(test_Y, test_Y_hat)

print('RMSE :', RMSE)
print('MAE :', MAE)
print('EVS :', EVS)

from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense
from keras.layers.core import Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import config as cfg
import numpy as np

def build_encode_decode_model():
    model = Sequential()
    return model


def build_model(notes):
    out_space = len(set(notes))

    model = Sequential()
    model.add(LSTM(
        128,
        return_sequences=True,
        input_shape=(cfg.sequence_length, out_space)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(out_space))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy')

    try:
        model.load_weights(cfg.weights)
    finally:
        return model


def _build_encoder_structure(notes):
    encoder_in = Input(shape=(None, len(notes)))
    encoder_lstm1 = LSTM(cfg.n_neurons, return_sequences=True, name='enLSTM1')

    encoder_out = encoder_lstm1(encoder_in)
    encoder_out = Dropout(0.2)(encoder_out)
    encoder_lstm2 = LSTM(cfg.n_neurons, return_state=True, name='enLSTM2')

    encoder_out, state_h, state_c = encoder_lstm2(encoder_out)
    encoder_states = [state_h, state_c]

    return encoder_in, encoder_states


def build_training_model(note_set):
    # note_set = set(notes)
    encoder_in, encoder_states = _build_encoder_structure(note_set)

    decoder_in = Input(shape=(None, len(note_set)))
    decoder_lstm1 = LSTM(cfg.n_neurons, name='deLSTM1',
                         return_sequences=True)
    decoder_out = decoder_lstm1(decoder_in, initial_state=encoder_states)

    decoder_lstm2 = LSTM(cfg.n_neurons, name='deLSTM2',
                         return_sequences=True, return_state=True)

    decoder_out, _, _ = decoder_lstm2(decoder_out)
    decoder_dense = Dense(len(note_set), activation='softmax')
    decoder_out = decoder_dense(decoder_out)

    model = Model(inputs=[encoder_in, decoder_in], outputs=[decoder_out])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    try:
        model.load_weights(cfg.weights, by_name=True)
    except Exception as exception:
        print('Error: previous weights didn\'t fit correctly')
        print(exception)
    finally:
        return model


def build_generative_model(notes):
    encoder_in, encoder_states = _build_encoder_structure(notes)
    encoder_model = Model(encoder_in, encoder_states)

    decoder_in = Input(shape=(None, len(notes)))
    decoder_state_in_h = Input(shape=(cfg.n_neurons,))
    decoder_state_in_c = Input(shape=(cfg.n_neurons,))
    decoder_state_ins = [decoder_state_in_h, decoder_state_in_c]

    decoder_lstm1 = LSTM(cfg.n_neurons, name="deLSTM1", return_sequences=True)
    decoder_lstm2 = LSTM(cfg.n_neurons, name="deLSTM2",
                         return_sequences=True, return_state=True)
    # try:
    #     weights = np.load(cfg.weights_dir + 'deLSTM1' + cfg.weights_ext,
    #                       allow_pickle=True)
    #     # biases = np.load(cfg.weights_dir + 'deLSTM1' + cfg.biases_ext)
    #     decoder_lstm.set_weights(weights)
    # except Exception as execption:
    #     print('probably couldn\'t find the file')
    #     print(execption)

    decoder_out = decoder_lstm1(decoder_in, initial_state=decoder_state_ins)
    decoder_out, state_h, state_c = decoder_lstm2(decoder_out)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(len(notes), activation='softmax')
    decoder_out = decoder_dense(decoder_out)

    decoder_model = Model([decoder_in] + decoder_state_ins,
                          [decoder_out] + decoder_states)

    try:
        encoder_model.load_weights(cfg.weights, by_name=True)
        decoder_model.load_weights(cfg.weights, by_name=True)
    finally:
        return encoder_model, decoder_model


def train_model(model, epochs, model_in, model_out):
    # Create call back for checking how well the model is training
    mc = ModelCheckpoint(cfg.weights, save_weights_only=True, period=16)
    model.fit([model_in, model_in], model_out, epochs=epochs, callbacks=[mc])

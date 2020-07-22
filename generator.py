import csv, glob, sys, getopt
import config as cfg
from lstm import *
from midi_io import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# Things to do
# DONE Read in a number of training midi's
# DONE Standardize each note type.
# DONE Write Code for command line
# DONE Update config to read/write to external csv/txt

# Trains model
def _old_prep():
    abstract_notes = ''
    for file in glob.glob(cfg.training_midi + '*.mid'):
        abstract_notes += get_abstract_notes_from_midi(file) + ' '
    # abstract_notes = get_abstract_notes_from_midi(cfg.user_file)
    abstract_notes = abstract_notes.rstrip()

    notes, note_set, index_note_lookup, note_index_lookup \
        = get_note_data_from_abstract_notes(abstract_notes)
    model_in, model_out = create_sequences_from_note_data(notes, note_set, note_index_lookup)

    # Model creation and training
    # model = build_model(notes)
    model = build_training_model(note_set)

    return note_set, model_in, model_out, model

def _prep_model_and_data():
    abstract_notes_by_file = []
    for file in glob.glob(cfg.training_midi + '*.mid'):
        abstract_notes_by_file.append(get_abstract_notes_from_midi(file))

    notes, note_set, index_note_lookup, note_index_lookup\
        = get_note_data_from_abstract_notes_new(abstract_notes_by_file)

    model_in, model_out = create_sequences_from_note_data_new(notes, note_set, note_index_lookup)
    model = build_training_model(note_set)

    return note_set, model_in, model_out, model


def train():
    # note_set, model_in, model_out, model = _old_prep()
    note_set, model_in, model_out, model = _prep_model_and_data()
    train_model(model=model, epochs=cfg.n_epochs, model_in=model_in, model_out=model_out)

    # Saving model weights and note set
    model.save_weights(cfg.weights)
    with open(cfg.note_set_file, 'w', newline='') as noteSetFile:
        wr = csv.writer(noteSetFile, quoting=csv.QUOTE_ALL)
        wr.writerow(note_set)


# Helper function for generate. Gets next predicted note
def _predict_next(model, model_in, index_note_lookup):
    raw_out = model.predict(model_in, verbose=0)[0]
    binary_out = np.zeros((len(model_in[0][0])), dtype=bool)
    binary_out[np.argmax(raw_out)] = True
    return [binary_out, index_note_lookup[np.argmax(raw_out)]]


def _update_pitch(curr_pitch, abs_note):
    parts = abs_note.split(',')
    if parts[0] == 'R':
        return curr_pitch
    elif (curr_pitch + int(parts[0])) <= 32:
        curr_pitch += int(parts[0]) + 13
        return curr_pitch, str(int(parts[0]) + 13) + ',' + parts[1]
    else:
        return [curr_pitch + int(parts[0]), abs_note]


class GeneratorModel:
    _instance = None


    def __init__(self):
        with open(cfg.note_set_file, newline='') as noteSetFile:
            reader = csv.reader(noteSetFile)
            note_set = set(list(reader)[0])

        index_note_lookup, note_index_lookup = get_lookup_data_from_note_set(note_set)

        encode_model, decode_model = build_generative_model(note_set)


def _generate_abstract_notes():
    # Build model from training
    with open(cfg.note_set_file, newline='') as noteSetFile:
        reader = csv.reader(noteSetFile)
        note_set = set(list(reader)[0])

    index_note_lookup, note_index_lookup = get_lookup_data_from_note_set(note_set)

    encode_model, decode_model = build_generative_model(note_set)

    # Load starting point from song
    abstract_notes_start = get_abstract_notes_from_midi(cfg.user_file)
    current_sequence = get_generated_start_sequence(abstract_notes_start,
                                                    note_set, note_index_lookup)
    current_states = encode_model.predict(current_sequence.reshape(1, cfg.sequence_length, len(note_set)))

    current_sequence = np.zeros((1, 1, len(note_set)), dtype=bool)
    current_sequence[0, 0, randint(0, len(note_set))] = True

    generated_abstract_notes = ''

    # Generate song in abstract notes
    for i in range(cfg.n_generated_notes):
        # predict next sequence, and get new set of states
        out, hidden_state, cell_state = decode_model.predict([current_sequence] + current_states)

        # get new note as an indexed value and as a one_hot_encoded array
        new_note_index = np.argmax(out[0, -1, :])
        generated_abstract_notes += index_note_lookup[new_note_index] + ' '
        current_sequence = np.zeros((1, 1, len(note_set)), dtype=bool)
        current_sequence[0, 0, new_note_index] = True
        # current_sequence = current_sequence[0][0:][:]
        # current_sequence = np.vstack((current_sequence, new_note_hot_encoding))
        # current_sequence = current_sequence.reshape((tuple([1]) + current_sequence.shape))

        # set new cell states
        current_states = [hidden_state, cell_state]

    return generated_abstract_notes.rstrip()


# Generates a new melody
def generate():
    generated_abstract_notes = _generate_abstract_notes()

    # Build melody stream from abstract notes
    melody_stream = build_stream_from_abstract_notes(generated_abstract_notes)

    # Write the newly generated file
    outMidi = m21.midi.translate.streamToMidiFile(melody_stream)
    outMidi.open(cfg.generated_midi + 'generated_' + cfg.generated_file_base, 'wb')
    outMidi.write()
    outMidi.close()


def evaluate():
    corpus = []
    for file in glob.glob(cfg.training_midi + '*.mid'):
        abstract_notes = ''
        abstract_notes += get_abstract_notes_from_midi(file) + ' '
        corpus.append(abstract_notes.split(' '))

    generated_notes = _generate_abstract_notes().split(' ')
    score = sentence_bleu(corpus, generated_notes, weights=(0, 1, 0, 0))

    print('BLEU score of generated song: ', score)


"""Run the model """
if __name__ == '__main__':
    optlist, args = getopt.getopt(sys.argv[1:], 'egt:f', ['n-epochs'])
    doTrain = False
    doGenerate = False
    doEvaluation = False

    for o, a in optlist:
        if o == '-f':
            print(str(a))
            cfg.user_file = a
        if o == '-t':
            doTrain = True
        if o == '-g':
            doGenerate = True
        if o == '--n-epochs':
            cfg.n_epochs = a
        if o == '-e':
            doEvaluation = True

    print(cfg.user_file)

    if doTrain == doGenerate:
        print('Doing full training and generation')
        train()
        generate()
    elif doTrain:
        print('Only training')
        train()
    else:
        print('Only generating')
        generate()

    if doEvaluation:
        print('Evaluating current model')
        evaluate()
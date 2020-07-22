from fractions import Fraction
import music21 as m21
import config as cfg
import numpy as np
from random import randint


def get_abstract_notes_from_midi(file):
    # Get note/rest data from the midi
    midi_data = m21.converter.parse(file)
    melody_part = midi_data[0]

    prev_pitch = 60  # going to make this representation start at middle C

    abstractNotes = ''
    if len(melody_part.sorted.notes) == 0:
        return abstractNotes
    elif isinstance(melody_part.sorted.notes[0], m21.note.Rest):
        elements = melody_part.sorted.notes[1:]
    else:
        elements = melody_part.sorted.notes

    for note in elements:
        abs_note = ''
        if isinstance(note, m21.note.Note):
            abs_note += str(note.pitch.midi) \
                        + ',' + str(Fraction(note.duration.quarterLength))
            prev_pitch = note.pitch.midi
            abstractNotes += abs_note + ' '
        elif isinstance(note, m21.note.Rest):
            abs_note += 'R' + ',' + str(Fraction(note.duration.quarterLength))
            abstractNotes += abs_note + ' '

    return abstractNotes.rstrip()


def build_stream_from_abstract_notes(abstract_notes):
    melody = m21.stream.Voice()
    melody.timeSignature = m21.meter.TimeSignature(cfg.time_signature)
    melody.insert(0.0, m21.tempo.MetronomeMark(number=cfg.tempo))

    prevPitch = 60  # Replace with config file

    for element in abstract_notes.split(' '):
        parts = element.split(',')
        if parts[0] == 'R':
            elementToInsert = m21.note.Rest(quarterLength=Fraction(parts[1]))
        else:
            elementToInsert = m21.note.Note(quarterLength=Fraction(parts[1]),
                                            midi=int(parts[0]))
            prevPitch += int(parts[0])
        melody.append(elementToInsert)

    melody.append(m21.bar.Barline(type='final'))

    return melody


def get_note_data_from_abstract_notes(abstract_notes):
    # Loads in individual abstract notes as an object, and creates
    #   a number of helpful indices for quick look up
    notes = [x for x in abstract_notes.split(' ')]
    note_set = set(notes)
    index_note_lookup = dict((i, n) for i, n in enumerate(note_set))
    note_index_lookup = dict((n, i) for i, n in enumerate(note_set))

    return notes, note_set, index_note_lookup, note_index_lookup


def get_note_data_from_abstract_notes_new(abstract_notes_by_file):
    notes = []
    note_set = set()
    for abstract_file_notes in abstract_notes_by_file:
        file_notes = abstract_file_notes.split(' ')
        if len(file_notes) < 1:
            continue
        notes.append(file_notes)
        note_set.update(file_notes)
    index_note_lookup = dict((i, n) for i, n in enumerate(note_set))
    note_index_lookup = dict((n, i) for i, n in enumerate(note_set))

    return notes, note_set, index_note_lookup, note_index_lookup

def get_lookup_data_from_note_set(note_set):
    index_note_lookup = dict((i, n) for i, n in enumerate(note_set))
    note_index_lookup = dict((n, i) for i, n in enumerate(note_set))

    return index_note_lookup, note_index_lookup


def create_sequences_from_note_data(notes, note_set, note_index_lookup):
    # Read in sequences as abstract notes
    seq_len = cfg.sequence_length
    in_sequences = []
    # out_sequences = []

    num_sequences = len(notes) - seq_len

    for i in range(0, num_sequences, 1):
        sequence_in = notes[i:i + seq_len]
        # next_note = notes[i+1:i+1 + seq_len]
        in_sequences.append(sequence_in)
        # out_sequences.append(next_note)

    # Now we convert those sequences into binary matricies
    model_in  = np.zeros((num_sequences, seq_len, len(note_set)), dtype=np.bool)
    # model_out = np.zeros((num_sequences, seq_len, len(note_set)), dtype=np.bool)

    for i, sequence in enumerate(in_sequences):
        for j, note in enumerate(sequence):
            model_in[i, j, note_index_lookup[note]] = 1
    #     model_out[i, note_index_lookup[next_notes[i]]] = 1


    model_out = model_in[1:]
    model_in = model_in[:-1]

    return model_in, model_out


def create_sequences_from_note_data_new(notes, note_set, note_index_lookup):
    seq_len = cfg.sequence_length
    num_sequences = 0
    sequences_by_file = []

    for file_notes in notes:
        temp_num_seq = (len(file_notes) - seq_len)
        in_sequences = []
        for i in range(0, temp_num_seq, 1):
            in_sequence = file_notes[i:i + seq_len]
            in_sequences.append(in_sequence)
        sequences_by_file.append(in_sequences)
        num_sequences += (temp_num_seq - 1)

    # I have no idea why I need this
    num_sequences += seq_len

    model_in = np.empty((num_sequences, seq_len, len(note_set)), dtype=bool)
    model_out = np.empty((num_sequences, seq_len, len(note_set)), dtype=bool)
    temp_step = 0

    for in_sequences in sequences_by_file:
        file_model_in = np.zeros((len(in_sequences), seq_len, len(note_set)), dtype=np.bool)
        for i, sequence in enumerate(in_sequences):
            for j, note in enumerate(sequence):
                file_model_in[i, j, note_index_lookup[note]] = 1

        file_model_out = file_model_in[1:]
        file_model_in = file_model_in[:-1]

        model_in[temp_step:file_model_in.shape[0]+temp_step, :, :] = file_model_in
        model_out[temp_step:file_model_out.shape[0]+temp_step, :, :] = file_model_out

        temp_step += len(in_sequences) - 1

    return model_in, model_out

def get_generated_start_sequence(abstract_notes, note_set, note_index_lookup):
    seq_len = cfg.sequence_length
    notes = [x for x in abstract_notes.split(' ')]
    salt = randint(0, 10) * -1
    abstract_start = notes[-seq_len + salt: salt]

    binary_start = np.zeros((1, seq_len, len(note_set)), dtype=np.bool)
    for i, note in enumerate(abstract_start):
        binary_start[0, i, note_index_lookup[note]] = 1

    return binary_start


def test_to_and_from_midi():
    test_abstract = get_abstract_notes_from_midi('midi/dd.mid')
    print('Original  Abstract file: ' + test_abstract)
    test_melody = build_stream_from_abstract_notes(test_abstract)

    outMidi = m21.midi.translate.streamToMidiFile(test_melody)
    outMidi.open(cfg.generated_midi + 'test.mid', 'wb')
    outMidi.write()
    outMidi.close()

    test_generated_abstract = get_abstract_notes_from_midi(cfg.generated_midi + 'test.mid')
    print('Generated Abstract file: ' + test_generated_abstract)

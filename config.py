# model configurations
n_epochs = 256
sequence_length = 32
n_generated_notes = 64
n_neurons = 256

# Generated Music
time_signature = '4/4'
tempo = 160

# Paths
training_midi = 'midi/clifford_brown/'
generated_midi = 'midi/generated/'
weights = 'weights.h5'
note_set_file = 'training_note_set.csv'

generated_file_base = 'CliffordBrown_Stompin\'AtTheSavoy_FINAL.mid'
user_file = training_midi + generated_file_base

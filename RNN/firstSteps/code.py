import numpy as np 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

START_CHAR = '\b' 
END_CHAR = '\t' 
PADDING_CHAR = '\a'

chars = set( [START_CHAR, '\n', END_CHAR, PADDING_CHAR] )

input_frame = "APEO.txt"
output_fname = "out.txt"
model_fname = "model"

with open(input_frame) as f :
	for line in f :
		chars.update( list( line.strip().lower() ) )

indices_to_chars = { c : i for i,c in enumerate(sorted(list(chars))) }
num_chars = len(chars)

def get_one(i, sz):
	result = np.zeros(sz)
	result[i] = 1
	return result

char_vectors = { 
	c : (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars ) ) 
	for c, v in indices_to_chars.items()
}

senrence_end_marker = set('.!?')

sentences = []
current_sentence = ''
with open( input_frame, 'r' ) as f :
	for line in f :
		s = line.strip().lower()
		if len(s) > 0 :
			current_sentence += s + "\n"
		if( len(s) == 0 or s[-1] in senrence_end_marker ) :
			current_sentence = current_sentence.strip()
			if len(current_sentence) > 10 :
				sentences.append(current_sentence)
			current_sentence = ''


def get_matrices(sentences) :
	max_sentences_lenght = np.max([len(x) for x in sentences ])
	X = np.zeros((len(sentences), max_sentences_lenght, len(chars),), dtype=np.bool)
	Y = np.zeros((len(sentences), max_sentences_lenght, len(chars),), dtype=np.bool)
	for i, sentence in enumerate(sentences) :
		char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentences_lenght+1, PADDING_CHAR)
		for t in range(max_sentences_lenght) :
			X[i, t, :] = char_vectors[char_seq[t]]
			Y[i, t, :] = char_vectors[char_seq[t+1]]
	return X,Y


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation

model = Sequential()
model.add( LSTM( output_dim=128, activation='tanh', return_sequences=True, input_dim=num_chars) )
model.add( Dropout(0.2) )
model.add( TimeDistributed(Dense(units=num_chars) ) )
model.add( Activation('softmax') )

from keras.optimizers import Adam

model.compile( loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'] )

test_indices    = np.random.choice( range(len(sentences) ), int(len(sentences) * 0.05) )
sentences_train = [ sentences[x]
	for x in set( range( len( sentences ) ) ) - set(test_indices) ]
sentences_test  = [ sentences[x] for x in test_indices ]
sentences_train = sorted( sentences_train, key=lambda x : len(x))
X_test, Y_test  = get_matrices(sentences_test)
batch_size = 16


def generate_batch():
	while True :
		for i in range( int( len(sentences_train) / batch_size) ) :
			sentences_batch = sentences_train[ i * batch_size : (i+1) * batch_size]
			yield get_matrices(sentences_batch)


from keras.callbacks import Callback 


class CharSampler(Callback) :
	def __init__(self, char_vectors, model) :
		self.char_vectors = char_vectors
		self.model = model

	def on_train_begin(self, logs={}) :
		self.epoch = 0
		if os.path.isfile(output_fname) :
			os.remove(output_fname)

	def sample( self, preds, temperature=1.0) :
		preds  = np.asarray(preds).astype('float64')
		preds  = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds  = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)

	def sample_one(self, T) :
		result = START_CHAR
		while len(result) < 500 :
			Xsampled = np.zeros( ( 1, len(result), num_chars) )
			for t, c in enumerate( list( result) ) :
				Xsampled[0,t,:] = self.char_vectors[ c ]
			Ysampled = self.model.predict( Xsampled, batch_size=1)[0,:]
			Yv = Ysampled[len(result)-1, :]
			selected_char = indices_to_chars[ self.sample( Yv, T ) ]
			if selected_char == END_CHAR :
				break
			result = result + selected_char
		return result

	def on_epoch_end(self, batch, logs={}) :
		self.epoch = self.epoch + 1
		if self.epoch % 1 == 0 :
			print("\nEpoch %d text sampling:" % self.epoch)
			with open( output_fname, 'a' ) as outf :
				outf.write( '\n===== Epoch %d =====\n' % self.epoch )
				for T in [0.3, 0.5, 0.7, 0.9, 1.1] :
					print('\tsampling, T = %.1f...' % T)
					for _ in range(5) :
						self.model.reset_states()
						res = self.sample_one(T)
						outf.write( '\nT = %.1f\n$s\n' % (T, res[1:]) )


from keras.callbacks import ModelCheckpoint, CSVLogger

cb_sampler = CharSampler(char_vectors, model)
cb_logger  = CSVLogger(model_fname + '.log' )


model.fit_generator( generate_batch(), int(len( sentences_train ) / batch_size ) * batch_size, 
	epochs=10, verbose=True, validation_data=(X_test, Y_test ), callbacks=[cb_logger,cb_sampler] )
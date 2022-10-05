import numpy as np
import re
import os
import unicodedata
from collections import Counter

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_translation_pairs_from_file(filename, max_length=10):
	sentence_pairs = []
	with open(filename, "r", encoding='utf-8') as fin:
		for line in fin:
			sentence_pair = [[w for w in normalize_string(s).split()] for s in line.strip().split('\t')]
			if len(sentence_pair[0]) < max_length and len(sentence_pair[1]) < max_length:
				sentence_pairs.append(sentence_pair[::-1])
	return sentence_pairs

def read_cornell_pairs_from_file(filename, max_length=10):
	sentence_pairs = []
	sentences = []
	with open(filename, "r", encoding='utf-8') as fin:
		for line in fin:
			sentence_pair = [[w for w in normalize_string(s).split()] for s in line.rstrip().split('\t')]
			if len(sentence_pair[0]) > 0 and len(sentence_pair[0]) < max_length and \
				len(sentence_pair[1]) > 0 and len(sentence_pair[1]) < max_length:
				sentence_pairs.append(sentence_pair)
	return sentence_pairs

def build_vocab_from_sentence_pairs(sentence_pairs, min_frequency=3):
	"""
	Builds a vocabulary mapping from word to index based on the sentences.
	Remove low frequency words.
	Returns vocabulary.
	"""
	vocabulary = dict()
	vocabulary['<pad>'] = 0
	vocabulary['<sos>'] = 1
	vocabulary['<eos>'] = 2
	vocabulary['<unk>'] = 3

	word_counter = Counter()
	
	for sentence_pair in sentence_pairs:
		for sentence in sentence_pair:
			for word in sentence:
				word_counter[word] += 1

	for k, v in word_counter.items():
		if v >= min_frequency:
			vocabulary[k] = len(vocabulary)
	return vocabulary

def build_input_data(sentence_pairs, vocabulary, max_length=20):
	"""
	Maps sentences and labels to vectors based on a vocabulary.
	"""
	pad_idx = vocabulary['<pad>']
	sos_idx = vocabulary['<sos>']
	eos_idx = vocabulary['<eos>']
	unk_idx = vocabulary['<unk>']

	encoder_input = []
	decoder_input = []
	decoder_target = []
	for sentence_pair in sentence_pairs:
		s1 = [vocabulary.get(w, unk_idx) for w in sentence_pair[0][:max_length-1]]
		# encoder_input.append([sos_idx] + s1 + [pad_idx] * (max_length-len(s1)-1))
		encoder_input.append(s1 + [pad_idx] * (max_length-len(s1)))
		s2 = [vocabulary.get(w, unk_idx) for w in sentence_pair[1][:max_length-1]]
		decoder_input.append([sos_idx] + s2 + [pad_idx] * (max_length-len(s2)-1))
		decoder_target.append(s2 + [eos_idx] + [pad_idx] * (max_length-len(s2)-1))
	encoder_input = np.array(encoder_input)
	decoder_input = np.array(decoder_input)
	decoder_target = np.array(decoder_target)
	
	return encoder_input, decoder_input, decoder_target

def load_data(data_type):
	# determine the data_path and read data from files
	if data_type == 'translation':
		train_path="data/fra_cleaned.txt"
		valid_path="data/fra_cleaned.txt"
		max_length = 10
		train_data = read_translation_pairs_from_file(train_path, max_length=max_length)
		valid_data = read_translation_pairs_from_file(valid_path, max_length=max_length)
		valid_data = valid_data[::10]
	elif data_type == 'dialogue':
		train_path="data/cornell_formatted_movie_lines.txt"
		valid_path="data/cornell_formatted_movie_lines.txt"
		max_length = 10
		train_data = read_cornell_pairs_from_file(train_path, max_length=max_length)
		valid_data = read_cornell_pairs_from_file(valid_path, max_length=max_length)
		valid_data = valid_data[::10]
	else:
		raise NotImplementedError

	# build vocabulary from training data
	vocabulary = build_vocab_from_sentence_pairs(train_data, min_frequency=3)

	# get input data
	encoder_input_train, decoder_input_train, decoder_target_train = build_input_data(train_data, vocabulary, max_length=max_length)
	encoder_input_valid, decoder_input_valid, decoder_target_valid = build_input_data(valid_data, vocabulary, max_length=max_length)

	return encoder_input_train, decoder_input_train, decoder_target_train, \
		encoder_input_valid, decoder_input_valid, decoder_target_valid, vocabulary

if __name__ == "__main__":
	encoder_input_train, decoder_input_train, decoder_target_train, \
		encoder_input_valid, decoder_input_valid, decoder_target_valid, vocabulary = load_data('translation')
	print(len(vocabulary))
	print(encoder_input_train.shape)
	print(encoder_input_valid.shape)
	print(encoder_input_train[0])
	print(decoder_input_train[0])
	print(encoder_input_valid[0])
	print(decoder_input_valid[0])

import csv
import itertools
import nltk
import numpy as np
import os
import sys

root_path = os.path.dirname(os.path.realpath(__file__))

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read Data
print("Reading CSV file...")
with open(os.path.join(root_path, "data/reddit-comments-2015-08.csv"), "r") as fp:
    reader = csv.reader(fp, skipinitialspace=True)
    next(reader)
    # split full comments into sentences
    sentences = itertools.chain(
        *[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["{} {} {}".format(
        sentence_start_token, sentence, sentence_end_token) for sentence in sentences]
print("Parsed sentences: {}".format(len(sentences)))

# Tokensize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found {} unique words tokens.".format(len(word_freq.keys())))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size {}".format(vocabulary_size))
print("The least frequency word in our vocabulary is {} and appeared {} times".format(
    vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [
        w if w in word_to_index else unknown_token for w in sent]

print("Example sentence: {}".format(sentences[0]))
print(
    "Example sentence after Pre-processing: {}".format(tokenized_sentences[0]))

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]]
                      for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]]
                      for sent in tokenized_sentences])
np.save(os.path.join(root_path, "data/X_train"), X_train)
np.save(os.path.join(root_path, "data/y_train"), y_train)

import sys
import os
import re
import random
from collections import Counter
import cPickle as pickle
import numpy as np

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
RANDOM_SEED = 1234

class TextReader(object):

    def __init__(self, data_dir, num_classes=2, suffix_list=None):
        self.data_dir = data_dir
        self.num_classes = num_classes
        if suffix_list:
            self.suffix_list = suffix_list
        else:
            self.suffix_list = [str(x) for x in range(num_classes)]
        self.data_files = None

    def get_filenames(self):
        if not os.path.exists(self.data_dir):
            sys.exit('Data directory does not exist.')
        data_files = []
        for f in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, f)
            if os.path.isfile(f):
                chunks = f.split('.')
                if chunks[-1] in self.suffix_list:
                    data_files.append(f)
        assert data_files
        self.data_files = data_files
        return data_files

    def clean_str(self, string):
        """
        Tokenization/string cleaning.
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()

    def prepare_dict(self, vocab_size=10000):
        max_sent_len = 0
        c = Counter()
        data_files = self.get_filenames()
        # store the preprocessed raw text to avoid cleaning it again
        self.raw_text = []
        for f in data_files:
            strings = []
            with open(f, 'r') as infile:
                for line in infile:
                    clean_string = self.clean_str(line)
                    strings.append(clean_string)
                    toks = clean_string.split()
                    if len(toks) > max_sent_len:
                        max_sent_len = len(toks)
                    for t in toks:
                        c[t] += 1
            self.raw_text.append(strings)
        total_words = len(c)
        assert total_words >= vocab_size
        word_list = [p[0] for p in c.most_common(vocab_size - 2)]
        word_list.insert(0, PAD_TOKEN)
        word_list.insert(0, UNK_TOKEN)
        self.word2idx = dict()
        vocab_file = os.path.join(self.data_dir, 'vocab')
        with open(vocab_file, 'w') as outfile:
            for idx, w in enumerate(word_list):
                self.word2idx[w] = idx
                outfile.write(w + '\t' + str(idx) + '\n')
        print '%d words found in training set. Truncate to vocabulary size %d.' % (total_words, vocab_size)
        print 'Dictionary saved to file %s. Max sentence length in data is %d.' % (vocab_file, max_sent_len)
        return max_sent_len

    def generate_index_data(self, max_sent_len=100):
        self.max_sent_len = max_sent_len
        sentence_and_label_pairs = []
        for label, strings in enumerate(self.raw_text):
            for s in strings:
                toks = s.split()
                toks_len = len(toks)
                if toks_len <= max_sent_len:
                    pad_left = (max_sent_len - toks_len) / 2
                    pad_right = int(np.ceil((max_sent_len - toks_len) / 2.0))
                else:
                    continue
                toks_idx = [1 for i in range(pad_left)] + [self.word2idx[t] if t in self.word2idx else 0 for t in toks] + \
                    [1 for i in range(pad_right)]
                sentence_and_label_pairs.append((toks_idx, label))
        return sentence_and_label_pairs

    def shuffle_and_split(self, sentence_and_label_pairs, test_fraction=0.2):
        random.seed(RANDOM_SEED)
        random.shuffle(sentence_and_label_pairs)
        self.num_examples = len(sentence_and_label_pairs)
        sentences, labels = zip(*sentence_and_label_pairs)
        test_num = int(self.num_examples * test_fraction)
        self.test_data = (sentences[:test_num], labels[:test_num])
        self.train_data = (sentences[test_num:], labels[test_num:])
        dump_to_file(os.path.join(self.data_dir, 'train.cPickle'), self.train_data)
        dump_to_file(os.path.join(self.data_dir, 'test.cPickle'), self.test_data)
        print 'Split dataset into training and test set: %d for training, %d for testing.' % \
            (self.num_examples - test_num, test_num)
        return

    def prepare_data(self, vocab_size=10000, test_fraction=0.2):
        max_sent_lent = self.prepare_dict(vocab_size)
        sentence_and_label_pairs = self.generate_index_data(max_sent_lent)
        self.shuffle_and_split(sentence_and_label_pairs, test_fraction)
        return

class DataLoader(object):

    def __init__(self, filename, batch_size=50):
        self._x, self._y = load_from_dump(filename)
        assert len(self._x) == len(self._y)
        self._pointer = 0
        self._num_examples = len(self._x)

        self.batch_size = batch_size
        self.num_batch = int(np.ceil(self._num_examples / self.batch_size))
        print 'Loaded data with %d examples. %d examples per batch will be used.' % (self._num_examples, self.batch_size)

    def next_batch(self):
        # reset pointer
        if self.batch_size + self._pointer >= self._num_examples:
            batch_x, batch_y = self._x[self._pointer:], self._y[self._pointer:]
            self._pointer = (self._pointer + self.batch_size) % self._num_examples
            return (batch_x + self._x[:self._pointer], batch_y + self._y[:self._pointer])
        self._pointer += self.batch_size
        return (self._x[self._pointer-self.batch_size:self._pointer], 
            self._y[self._pointer-self.batch_size:self._pointer])


def dump_to_file(filename, obj):
    with open(filename, 'w') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'r') as infile:
        obj = pickle.load(infile)
    return obj


def test():
    reader = TextReader('./data/mr/', suffix_list=['neg', 'pos'])
    reader.prepare_data()
    print len(reader.train_data)
    print reader.train_data[0]
    print reader.max_sent_len

if __name__ == '__main__':
    test()
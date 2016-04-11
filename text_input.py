import sys
import os
import random
from collections import Counter
import cPickle as pickle
import numpy as np

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

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

    def prepare_dict(self, vocab_size=10000):
        max_sent_len = 0
        c = Counter()
        data_files = self.get_filenames()
        for f in data_files:
            with open(f, 'r') as infile:
                for line in infile:
                    toks = line.strip().split()
                    if len(toks) > max_sent_len:
                        max_sent_len = len(toks)
                    for t in toks:
                        c[t] += 1
        assert len(c) >= vocab_size
        word_list = [p[0] for p in c.most_common(vocab_size - 2)]
        word_list.insert(0, PAD_TOKEN)
        word_list.insert(0, UNK_TOKEN)
        self.word2idx = dict()
        with open(os.path.join(self.data_dir, 'vocab'), 'w') as outfile:
            for idx, w in enumerate(word_list):
                self.word2idx[w] = idx
                outfile.write(w + '\t' + str(idx) + '\n')
        return max_sent_len

    def generate_index_data(self, max_sent_len=100):
        self.max_sent_len = max_sent_len
        sentences_and_labels = []
        for label, f in enumerate(self.data_files):
            with open(f, 'r') as infile:
                for line in infile:
                    toks = line.strip().split()
                    toks_len = len(toks)
                    if toks_len <= max_sent_len:
                        pad_left = (max_sent_len - toks_len) / 2
                        pad_right = int(np.ceil((max_sent_len - toks_len) / 2.0))
                    else:
                        continue
                    toks_idx = [1 for i in range(pad_left)] + [self.word2idx[t] if t in self.word2idx else 0 for t in toks] + \
                        [1 for i in range(pad_right)]
                    sentences_and_labels.append((toks_idx, label))
        return sentences_and_labels

    def shuffle_and_split(self, sentences_and_labels, test_fraction=0.2):
        random.shuffle(sentences_and_labels)
        self.num_examples = len(sentences_and_labels)
        test_num = int(self.num_examples * test_fraction)
        self.test_data = sentences_and_labels[:test_num]
        self.train_data = sentences_and_labels[test_num:]
        dump_data(self.data_dir, 'train.cPickle', self.train_data)
        dump_data(self.data_dir, 'test.cPickle', self.test_data)
        print 'Split dataset into training and test set: %d for training, %d for test.' % \
            (self.num_examples - test_num, test_num)
        return

    def prepare_data(self, vocab_size=10000, test_fraction=0.2):
        max_sent_lent = self.prepare_dict(vocab_size)
        sentences_and_labels = self.generate_index_data(max_sent_lent)
        self.shuffle_and_split(sentences_and_labels, test_fraction)
        return

    def get_data_and_labels(self, test=False):
        if test:
            return zip(*self.test_data)
        return zip(*self.train_data)


class DataLoader(object):

    def __init__(self, x, y, batch_size=50):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.cur_idx = 0
        self.num_examples = len(x)

    def next_batch(self):
        if self.batch_size + self.cur_idx >= self.num_examples:
            batch_x, batch_y = self.x[self.cur_idx:], self.y[self.cur_idx:]
            self.cur_idx = (self.cur_idx + self.batch_size) % self.num_examples
            return (batch_x + self.x[:self.cur_idx], batch_y + self.y[:self.cur_idx])
        self.cur_idx += self.batch_size
        return (self.x[self.cur_idx-self.batch_size:self.cur_idx], 
            self.y[self.cur_idx-self.batch_size:self.cur_idx])

    def batches_per_epoch(self):
        return int(np.ceil(self.num_examples / self.batch_size))


def dump_data(data_dir, filename, data):
    dump_file = os.path.join(data_dir, filename)
    with open(dump_file, 'w') as outfile:
        pickle.dump(data, file=outfile)
    return

def load_data_from_dump(data_dir, filename):
    dump_file = os.path.join(data_dir, filename)
    with open(dump_file, 'r') as infile:
        data = pickle.load(infile)
    return zip(*data)

def main():
    reader = TextReader('./data/mr/', suffix_list=['neg', 'pos'])
    reader.prepare_data()
    print len(reader.train_data)
    print reader.train_data[0]
    print reader.max_sent_len
    x, y = reader.get_data_and_labels()
    loader = DataLoader(x, y, batch_size=1000)
    print 'Num examples: %d' % loader.num_examples
    for i in range(10):
        x,y = loader.next_batch()
        print 'Loader generates %d examples, current index at %d' % (len(x), loader.cur_idx)

if __name__ == '__main__':
    main()
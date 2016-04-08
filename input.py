import sys
import os
from collections import Counter
import cPickle as pickle
import numpy as np

UNK_TOKEN = '<unk>'

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
        c = Counter()
        data_files = self.get_filenames()
        for f in data_files:
            with open(f, 'r') as infile:
                for line in infile:
                    toks = line.strip().split()
                    for t in toks:
                        c[t] += 1
        assert len(c) >= vocab_size
        word_list = [p[0] for p in c.most_common(vocab_size - 1)]
        word_list.insert(0, UNK_TOKEN)
        self.word2idx = dict()
        with open(os.path.join(self.data_dir, 'vocab'), 'w') as outfile:
            for idx, w in enumerate(word_list):
                self.word2idx[w] = idx
                outfile.write(w + '\t' + str(idx) + '\n')
        return

    def generate_index_file(self):
        self.text_idx_data = []
        for f in self.data_files:
            file_idx_data = []
            with open(f, 'r') as infile:
                for line in infile:
                    toks = line.strip().split()
                    tok_idx = [self.word2idx[t] if t in self.word2idx else 0 for t in toks]
                    file_idx_data.append(tok_idx)
            self.text_idx_data.append(file_idx_data)
        dump_file = os.path.join(self.data_dir, 'idx.cPickle')
        with open(dump_file, 'w') as outfile:
            pickle.dump(self.text_idx_data, file=outfile)
        return

    def prepare_data(self, vocab_size=10000):
        self.prepare_dict(vocab_size)
        self.generate_index_file()
        return self.text_idx_data

def main():
    reader = TextReader('./data/mr/', suffix_list=['neg', 'pos'])
    text_idx_data = reader.prepare_data()
    print len(text_idx_data)
    print len(text_idx_data[0])
    print text_idx_data[0][0]

if __name__ == '__main__':
    main()
import sys
import os
from collections import Counter
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
            print 'Data directory does not exist.'
            sys.exit(-1)
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

def main():
    reader = TextReader('./data/mr/', suffix_list=['neg', 'pos'])
    reader.prepare_dict(vocab_size=1000)

if __name__ == '__main__':
    main()
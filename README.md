A Tensorflow implementation of convolutional neural network to classify sentences
=========
This implementation uses Tensorflow's `tf.conv2d` to perform 1D convolution on word sequences. It also supports using Google News word2vec pre-trained vectors to initialize word embeddings, which boosts the performance on movie review dataset from ~76% to ~81%.

The original theano implementation of this model by the author is [here](https://github.com/yoonkim/CNN_sentence). Another tensorflow implementation that does not support loading pretrained vectors is [here](https://github.com/dennybritz/cnn-text-classification-tf).

## Dependency

- python2.7+
- numpy
- tensorflow 1.0+

## Data

The data in `data/mr/` are movie review polarity data provided [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/). The current `data/word2vec` directory is empty. To use the pretrained word2vec embeddings, download the Google News pretrained vector data from [this Google Drive link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), and unzip it to the directory. It will be a `.bin` file.

## Usage

#### Preprocess the data

    python text_input.py

#### Train

    python train.py

By default the pretrained vectors will be loaded and used to initialize the embeddings. To suppress this, use

    python train.py --use_pretrain False

#### Evaluate

    python eval.py

By default evaluation is run over test set. To evaluate over training set, run

    python eval.py --train_data

## References

1. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014). [link](http://arxiv.org/abs/1408.5882)

## License

MIT

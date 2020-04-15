mkdir -p models/word_embeddings
cd models/word_embeddings
# wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip
# rm glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
rm wiki-news-300d-1M-subword.vec.zip

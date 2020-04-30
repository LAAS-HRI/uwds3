mkdir -p models/word_embeddings
cd models/word_embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip
unzip wiki-news-300d-1M-subword.bin.zip
rm wiki-news-300d-1M-subword.bin.zip

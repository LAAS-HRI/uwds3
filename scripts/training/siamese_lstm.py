from word_embeddings import create_embedding_layer
from siamese import siamese_network
from keras.models import Model
from keras.layers import Input, LSTM, Dropout


def base_model(pre_trained_embedding_file, max_length, lstm_dim):
    """Create base model of the siamese network
    """
    input = Input([max_length])
    embedding = create_embedding_layer(pre_trained_embedding_file)(input)
    dropout = Dropout(0.5)(embedding)
    preds = LSTM(lstm_dim)(dropout)
    model = Model(input, preds)
    print "Sentence features extractor:"
    model.summary()
    return model


def siamese_lstm(pre_trained_embedding_file, max_length, lstm_dim):
    """Create siamese network
    """
    base = base_model(pre_trained_embedding_file, max_length, lstm_dim)
    return siamese_network([max_length], base, metric="cosine")


if __name__ == '__main__':
    glove_embedding_file = "../../models/features/glove/glove.6B.300d.txt"
    model = siamese_lstm(glove_embedding_file, 20, 300)

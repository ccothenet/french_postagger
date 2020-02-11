from keras.layers import Input, Dense, Bidirectional, GRU, LSTM, Flatten, TimeDistributed, Embedding
from keras.models import Sequential
from keras.models import load_model
from numpy.random import seed
from tensorflow import set_random_seed
from keras import optimizers
import os

from utils import GetEmbedding


def build_or_load_model(options, vocab_train):
    # Build model or save model. Option --reset_model builds a new model
    if options.reset_model or not os.path.isfile(
            options.model_dir + "model.pt"):
        model = build_model(embedding_path=options.embedding_path,
                           nb_classes=18, maxlen=options.maxlen,
                           vocab_size=len(vocab_train) + 1)
    else:
        model = load_model(options.model_dir + "model.pt")
    return model


def build_model(embedding_path="None", nb_classes=18, maxlen=100, vocab_size=100000):
    """
    Model with only a dense layer to evaluate the different embeddings
    :param embedding_path: string
    :param nb_classes: int - Number of possible tags for pos-tagging
    :param maxlen: int - maximum token number for each sentence
    :param vocab_size: int - if keras embedding : size of vocabulary
    :return: model - model for evaluation, called in main
    """
    seed(0)
    set_random_seed(0)
    if embedding_path != "None":
        embedding = GetEmbedding(embedding_path=embedding_path)

    model = Sequential()
    if embedding_path == "None":
        model.add(Embedding(vocab_size, 500, input_length=maxlen, trainable=True, mask_zero=True))
    else:
        model.add(embedding.keras_embeddings(train_embeddings=False))
    # model.add(Bidirectional(GRU(128, return_sequences=True)))
    # model.add(Bidirectional(GRU(128, return_sequences=True)))
    # model.add(Flatten(input_shape=[8, 20, 128]))
    model.add(Dense(nb_classes, activation="softmax"))
    model.compile(optimizer=optimizers.Adam(lr=0.1), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
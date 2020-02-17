from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing import sequence
import numpy as np
from tqdm import tqdm
import re

from get_statistics import get_statistics
from utils import GetEmbedding


def load_conllu_file(conllu_path):
    # sent_words : list of words in a sentence
    # sent_tags : list of tags in a sentence
    # words : list of sent_words
    # tags : list of sent_tags
    with open(conllu_path, "r", encoding="utf-8") as rfile:
        lines = rfile.readlines()
    words, tags = [], []
    sent_words, sent_tags = [], []
    unique_words = []
    for line in tqdm(lines):
        if not line.startswith("#") and len(line.strip()) > 0:
            splitted_line = line.split("\t")
            word = splitted_line[1]
            tag = splitted_line[3]
            sent_words.append(word)
            sent_tags.append(tag)
            if word not in unique_words:
                unique_words.append(word)
        if len(line.strip()) == 0:
            assert len(sent_words) == len(sent_tags), "Number of tags in the sentence differs from the number of tokens"
            words.append(np.array(sent_words))
            tags.append(np.array(sent_tags))
            sent_words = []
            sent_tags = []
    assert len(words) == len(tags), "Number of sent_tags differs from the number of sent_words"
    return np.array(words), np.array(tags), unique_words


def get_indices(embedding, X_set):
    """
    Get embedding_index for each word in the sentence
    :param embedding: Instance of the GetEmbedding class
    :param X_set: X_train, X_dev or X_test
    :return: list of list with indices of words in embedding rather than tokens
    """
    return [[embedding.get_index(o) for o in sub] for sub in X_set]


def prepare_train(options):
    """
    1 - Load CONLL-U files and vocabulary
    2 - Get indices of words in embeddings (pre-trained or not)
    3 - Pad Tokens to options.maxlen
    4 - One-Hot encode labels
    5 - Pad labels to options.maxlen
    :return: datasets for train, dev and test, and train-set vocabulary
    """
    # 1 - Load CONLLU files (use get_data.sh to use them)
    X_train, y_train, vocab_train = load_conllu_file(options.train_set)
    X_dev, y_dev, vocab_dev = load_conllu_file(options.dev_set)
    X_test, y_test, vocab_test = load_conllu_file(options.test_set)

    get_statistics(vocab_train, vocab_dev, vocab_test)

    # 2 - Get indices of words in embeddings
    if options.embedding_path != "None":
        embedding = GetEmbedding(options.embedding_path)
        X_train = get_indices(embedding, X_train)
        X_dev = get_indices(embedding, X_dev)
        X_test = get_indices(embedding, X_test)
    else:
        word_encoder = LabelEncoder()
        word_encoder.fit(["<unk>"] + vocab_train)
        # Words not in train are encoded as "<unk>" as they would be in prod.
        X_dev = [[o if o in word_encoder.classes_ else "<unk>" for o in sub] for
                 sub in X_dev]
        X_test = [[o if o in word_encoder.classes_ else "<unk>" for o in sub] for
                  sub in X_test]
        X_train = np.array([word_encoder.transform(o) for o in X_train])
        X_dev = np.array([word_encoder.transform(o) for o in X_dev])
        X_test = np.array([word_encoder.transform(o) for o in X_test])

        np.save(options.output_dir + "word_encoder.npy", word_encoder.classes_)

    # 3 - Pad sequences so that they are of the same length (length of options.maxlen).
    X_train = sequence.pad_sequences(X_train, maxlen=options.maxlen, padding="post")
    X_dev = sequence.pad_sequences(X_dev, maxlen=options.maxlen, padding="post")
    X_test = sequence.pad_sequences(X_test, maxlen=options.maxlen, padding="post")

    print("Nb sentences in train : {}".format(X_train.shape[0]))
    print("Nb sentences in dev : {}".format(X_dev.shape[0]))
    print("Nb sentences in test : {}".format(X_test.shape[0]))

    # 4 - Encodes output so that they are one-hot encoded.
    lb_encoder = LabelEncoder()
    lb_encoder.fit([o for sub in y_train for o in sub])

    y_train = np.array([lb_encoder.transform(sub) for sub in y_train])
    y_dev = np.array([lb_encoder.transform(sub) for sub in y_dev])
    y_test = np.array([lb_encoder.transform(sub) for sub in y_test])

    # 5 - Pad labels so that they are of the same length
    y_train = sequence.pad_sequences(y_train, maxlen=options.maxlen, padding="post")
    y_dev = sequence.pad_sequences(y_dev, maxlen=options.maxlen, padding="post")
    y_test = sequence.pad_sequences(y_test, maxlen=options.maxlen, padding="post")

    y_train = np_utils.to_categorical(y_train)
    y_dev = np_utils.to_categorical(y_dev)
    y_test = np_utils.to_categorical(y_test)

    # Save target labels. .npy to load it on evaluation and .txt to load it for
    # predict
    np.save(options.output_dir + "lb_encoder.npy", lb_encoder.classes_)

    with open(options.output_dir + "target_labels.txt", "w+") as wfile:
        for tag in lb_encoder.classes_:
            wfile.write("{}\n".format(tag))

    return X_train, X_dev, X_test, y_train, y_dev, y_test, vocab_train


def prepare_evaluation(options, options_train):
    """
    1 - Load CONLL-U files and vocabulary
    2 - Get indices of words in embeddings (pre-trained or not)
    3 - Pad Tokens to options.maxlen
    4 - One-Hot encode labels
    5 - Pad labels to options.maxlen
    :param options_train: options given during the training of the model
    :return: X_test, y_test
    """

    X_test, y_test, vocab_test = load_conllu_file(options.test_set)

    if options_train["embedding_path"] == "None":
        word_encoder = LabelEncoder()
        word_encoder.classes_ = np.load(
            options_train["output_dir"] + "word_encoder.npy")
        X_test = [[o if o in word_encoder.classes_ else "<unk>" for o in sub] for
                  sub in X_test]
        X_test = np.array([word_encoder.transform(o) for o in X_test])
    else:
        embedding = GetEmbedding(options_train["embedding_path"])
        X_test = get_indices(embedding, X_test)

    X_test = sequence.pad_sequences(X_test, maxlen=options_train["maxlen"], padding="post")

    lb_encoder = LabelEncoder()
    lb_encoder.classes_ = np.load(options_train["output_dir"] + "lb_encoder.npy")
    y_test = np.array([lb_encoder.transform(sub) for sub in y_test])
    y_test = sequence.pad_sequences(y_test, maxlen=options_train["maxlen"], padding="post")
    y_test = np_utils.to_categorical(y_test)

    return X_test, y_test


def prepare_prediction(options, options_train):
    string = options.str
    string = re.sub(r"(\W+)|(\W+)", r" \1 ",
                    string.lower())  # https://regex101.com/r/CdWuiL/1
    string = re.sub(r"\s+", " ", string)
    string = string.split()

    if options_train["embedding_path"] == "None":
        word_encoder = LabelEncoder()
        word_encoder.classes_ = np.load(
            options.output_dir + "word_encoder.npy")
        X_pred = [o if o in word_encoder.classes_ else "<unk>" for o in
                  string]
        X_pred = word_encoder.transform(X_pred)
    else:
        embedding = GetEmbedding(
            embedding_path=options_train["embedding_path"])
        X_pred = [embedding.get_index(o) for o in string]

    X_pred = sequence.pad_sequences([X_pred], maxlen=options_train["maxlen"], padding="post")

    return string, X_pred
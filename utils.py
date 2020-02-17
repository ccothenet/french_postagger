from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from optparse import OptionParser
from keras import backend as K


def load_embedding(embedding_path):
    if embedding_path.endswith(".bin"):  # Fauconnier's models, binary file
        embedding = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    elif embedding_path.endswith(".model"):  # Lettria's models
        embedding = KeyedVectors.load(embedding_path).wv
    elif embedding_path.endswith(".txt"):  # GloVe's models
        glove2word2vec(embedding_path, "embeddings/glove_w2v.txt")
        embedding = KeyedVectors.load_word2vec_format("embeddings/glove_w2v.txt")
    elif embedding_path == "None":
        embedding = None
    else:
        print(embedding_path)
    return embedding


class GetEmbedding(object):
    def __init__(
            self,
            embedding_path
    ):
        self.use_gpu = False
        self.embedding_path = embedding_path
        self.key_vect = load_embedding(embedding_path)
        self.word2index = {
            token: token_index for token_index, token in enumerate(
                self.key_vect.index2word
            )
        }

    def get_index(self, token):
        try:
            return self.word2index[token]
        except KeyError:
            return 0

    def vocab_dim(self):
        return self.key_vect.vectors.shape[0]

    def vect_dim(self):
        return self.key_vect.vectors.shape[1]

    def keras_embeddings(self, train_embeddings=False):
        return self.key_vect.get_keras_embedding(
            train_embeddings=train_embeddings)


def option_parser():
    parser = OptionParser()

    parser.add_option("--reset_model", action="store_true", help="Whether or not to reset model. If no model is found, a new one is created in all case.")
    parser.add_option("--output_dir", help="Folder to save output results", default="output/")
    parser.add_option("--model_dir", help="Folder to save model file", default="model/")
    parser.add_option("--log_dir", help="Folder to save logs of training", default="evaluations/")
    parser.add_option("--embedding_path", help="File of embedding", default="None")
    parser.add_option("--maxlen",
                      help="Max len (list of tokens) of the strings", default=100)
    parser.add_option("--evaluate", action="store_true")
    parser.add_option("--predict", action="store_true")
    parser.add_option("--str", help="string to predict. Only if predict is parsed", default="Je mange une pomme.")
    parser.add_option("--train_set", default="data/fr_partut-ud-train.conllu")
    parser.add_option("--dev_set", default="data/fr_partut-ud-dev.conllu")
    parser.add_option("--test_set", default="data/fr_partut-ud-test.conllu")
    parser.add_option("--eval_model", default="None")
    parser.add_option("--model_type", default="direct", help="'direct' model or 'bigru' or 'bilstm'")
    (options, args) = parser.parse_args()

    return options
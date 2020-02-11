

def get_statistics(vocab_train, vocab_dev, vocab_test):
    print("Size of Vocab Train : {}".format(len(vocab_train)))
    print("Size of Vocab Dev : {}".format(len(vocab_dev)))
    print("Size of Vocab Test : {}".format(len(vocab_test)))

    # Vocabulary differences
    unique_words = []
    for w in vocab_train + vocab_dev + vocab_test:
        if w not in unique_words:
            unique_words.append(w)
    print("Total size of vocabulary : {}".format(
        len(unique_words)))

    words_not_train = []
    for w in vocab_test + vocab_dev:
        if w not in vocab_train:
            words_not_train.append(w)
    print(
        "Total size of vocabulary in dev or test but not in train : {} ({} % \
        of dev and test set.)".format(
            len(words_not_train),
            len(words_not_train) * 100 / (len(vocab_dev) + len(vocab_test))))


import numpy as np
import json
import os

from keras.models import load_model
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from utils import option_parser
from build_model import build_or_load_model
from prepare_datasets import prepare_train, prepare_evaluation, prepare_prediction

if __name__ == "__main__":

    options = option_parser()

    if not options.predict and not options.evaluate:

        X_train, X_dev, X_test, y_train, y_dev, y_test, vocab_train = prepare_train(options)

        model = build_or_load_model(options, vocab_train)

        path_evaluation = options.log_dir
        if not os.path.isdir(path_evaluation):
            os.mkdir(path_evaluation)

        ModelCheckpoint = ModelCheckpoint(path_evaluation + "model.pt",
                                                  monitor='val_loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  mode='auto', period=1)

        ReduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=0.001, verbose=1)
        # Train the model
        history = model.fit(X_train,
                            y_train,
                            batch_size=8,
                            epochs=30,
                            validation_data=[X_dev, y_dev],
                            verbose=1,
                            callbacks=[ReduceLROnPlateau, ModelCheckpoint])

        print(history.history)
        print(type(history.history))
        # Evaluate model
        score = model.evaluate(X_test, y_test, batch_size=8)
        print("Evaluation score : {}".format(score))


        # Save model options
        with open(options.output_dir + "options_dict.json",
                  "w+") as wfile:
            json.dump(options.__dict__, wfile)


        with open(path_evaluation + "history.json", "w+") as wfile:
            json.dump(str(history.history), wfile)



        # Save model
        model.save(options.model_dir + "model.pt")
        model.save(path_evaluation + "model.pt")
        print("Model saved in file : {} and {}".format(options.model_dir + "model.pt", path_evaluation + "model.pt"))




    # 'evaluate' or 'predict'
    else:
        with open(options.output_dir + "options_dict.json", "r") as rfile:
            options_train = json.load(rfile)

        model = load_model(options_train["model_dir"] + "model.pt")

        # ############################################# #
        # ------------------EVALUATION----------------- #
        # ############################################# #
        if options.evaluate:
            X_test, y_test = prepare_evaluation(options_train)
            score = model.evaluate(X_test, y_test, batch_size=8)
            print("Evaluation score : {}".format(score))



    # ############################################# #
    # ------------------PREDICTION----------------- #
    # ############################################# #
        elif options.predict:
            string, X_pred = prepare_prediction(options, options_train)

            model = load_model(options_train["model_dir"] + "model.pt")

            with open(options.output_dir + "target_labels.txt", "r") as rfile:
                lines = rfile.readlines()
            idx2tag = {enum: tag.strip() for enum, tag in enumerate(lines)}

            result = model.predict(X_pred)

            indices_argmax = np.argmax(result[0], axis=1)
            tags = [idx2tag[o] for o in indices_argmax]
            for word, tag in zip(string, tags):
                print(word, tag, end="\t")

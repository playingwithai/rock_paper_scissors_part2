import logging
import os
import random
from enum import Enum

import click
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils

logging.getLogger("tensorflow").setLevel(logging.ERROR)


class MovesEnum(int, Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class NextMovePredictor:
    INPUT_SHAPE = (1, -1, 1)
    OUTPUT_SHAPE = (1, -1, 3)

    def __init__(self, dataset_name="rock_paper_scissors", model_name="nmp_model.h5"):
        base_path = os.getcwd()
        self.dataset_path = os.path.join(base_path, dataset_name)
        self.model_path = os.path.join(base_path, dataset_name, model_name)
        self.played_moves = []
        self.model = self._create_model()
        self.load_model()

    def _create_model(self):
        model = Sequential()
        model.add(
            LSTM(
                units=64,
                input_shape=(None, 1),
                return_sequences=True,
                activation="sigmoid",
            )
        )
        model.add(LSTM(units=64, return_sequences=True, activation="sigmoid"))
        model.add(LSTM(units=64, return_sequences=True, activation="sigmoid"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy", "categorical_crossentropy"],
        )
        return model

    def _get_input_data(self, moves):
        return np.array(moves).reshape(self.INPUT_SHAPE)

    def _get_output_data(self):
        return np_utils.to_categorical(
            np.array(self.played_moves[1:]), num_classes=3
        ).reshape(self.OUTPUT_SHAPE)

    def train(self, user_move, verbose=0):
        self.played_moves.append(user_move)
        if len(self.played_moves) <= 1:
            return
        input_data = self._get_input_data(self.played_moves[:-1])
        output_data = self._get_output_data()
        self.model.fit(input_data, output_data, epochs=1, verbose=verbose)

    def load_model(self):
        if os.path.exists(self.model_path):
            click.echo("Model loaded")
            self.model.load_weights(self.model_path)

    def save_model(self):
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        self.model.save(self.model_path)
        click.echo("Model saved")

    def predict_next_move(self):
        if not self.played_moves:
            return random.choice(list(map(int, MovesEnum.__iter__())))
        predictions = self.model.predict(self._get_input_data(self.played_moves))
        return np.argmax(predictions[0], axis=1)[0]

import click

from .next_move_prediction import NextMovePredictor, MovesEnum


class Game:

    """
    Point table:

             | Rock | Paper | Scissors
    ---------|------|-------|---------
    Rock     |   0  |  -1   |   1
    ---------|------|-------|---------
    Paper    |   1  |   0   |   -1
    ---------|------|-------|---------
    Scissors |   -1 |   1   |   0
    -----------------------------------
    """

    point_table = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ]

    def __init__(self):
        self.next_move_predictor = NextMovePredictor()
        self.user_score = 0
        self.bot_score = 0

    def get_user_point(self, user_move, bot_move):
        return self.point_table[user_move][bot_move]

    def get_bot_move(self):
        return self.point_table[self.next_move_predictor.predict_next_move()].index(-1)

    def update_bot(self, user_move):
        self.next_move_predictor.train(user_move)

    def _update_scores(self, user_point):
        if user_point > 0:
            self.user_score += 1
        elif user_point < 0:
            self.bot_score += 1

    def play_round(self, user_move):
        user_point = self.get_user_point(user_move, self.get_bot_move())
        self._update_scores(user_point)
        self.update_bot(user_point)
        return user_point

    def save_bot_training(self):
        self.next_move_predictor.save_model()

    def print_score(self):
        click.echo("Scores:")
        click.echo(f"You: {self.user_score}")
        click.echo(f"Bot:{self.bot_score}")

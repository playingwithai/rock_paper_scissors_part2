import click

from helpers.game import Game
from helpers.next_move_prediction import MovesEnum

MOVES = {
    "r": MovesEnum.ROCK,
    "p": MovesEnum.PAPER,
    "s": MovesEnum.SCISSORS,
}


if __name__ == "__main__":

    game = Game()
    click.clear()
    click.echo("Welcome to Rock Paper Scissors!")
    while True:
        user_move = click.prompt(
            "\nWhich move do you want to play? [R]ock, [P]aper, [S]cissors or [Q]uit",
            type=str,
        )
        user_move = user_move.lower()
        if user_move == "q":
            break
        user_move = MOVES.get(user_move)
        if user_move is None:
            click.echo("Insert a valid move!")
            continue
        point = game.play_round(user_move)
        click.clear()
        game.print_score()
        click.echo("\nLast result: ", nl=False)
        if point == 1:
            click.echo("You win :)")
        elif point == 0:
            click.echo("Draw :|")
        elif point == -1:
            click.echo("You lost :(")

    game.save_bot_training()
    click.echo("Bye")

from .algo import Algo, Piece

import random

class RandomAI(Algo):
    """
    Random AI that plays a random move.
    """

    def play(self, pieces: list[Piece]) -> dict:
        """
        Return the move played by the AI.
        
        :param pieces: list of pieces of the AI
        :type pieces: list[Piece]
        :return: {"from": (int, int), "to": (int, int), ["promotion": str]}
        :rtype: dict
        """

        random.shuffle(pieces)
        for piece in pieces:
            actions = piece.get_possible_actions(self.game.board)
            if actions:
                return random.choice(actions)
        
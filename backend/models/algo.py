from src.chess.game import Game
from src.chess.piece import Piece
from src.chess.player import Player

class Algo(Player):

    def __init__(self):
        self.color: str = 'undefined'
        self.game: Game = None

    def play(self, pieces: list[Piece]) -> dict:
        """
        Return the move played by the AI.
        
        :param pieces: list of pieces of the AI
        :type pieces: list
        :return: {"from": (int, int), "to": (int, int)}
        :rtype: dict
        """
        raise NotImplementedError
    
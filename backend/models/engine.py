from src.chess.game import Game
from src.chess.player import Player

class Engine(Player):

    def __init__(self):
        super().__init__(self.__class__.__name__, True)
        self.color: str = 'undefined'
        self.game: Game = None

    def play(self) -> dict:
        """
        Return the move played by the AI.
        
        :return: {"from": (int, int), "to": (int, int)}
        :rtype: dict
        """
        raise NotImplementedError
    
from src.chess.game import Game
from src.chess.player import Player

from src.utils.console import Style

class Engine(Player):

    class UndefinedAuthorError(Exception):
        def __init__(self, engine: type, *args):
            super().__init__(*args)
            self.message = "[Engine] Author not defined \n|__ You must define the __author__ of the engine: <" + engine.__name__ + ">"

        def __str__(self):
            return self.__repr__() + "\n" + Style("ERROR", self.message).__str__()
        

    class UndefinedDescriptionError(Exception):
        def __init__(self, engine: type, *args):
            super().__init__(*args)
            self.message = "[Engine] Description not defined \n|__ You must define the __description__ of the engine: <" + engine.__name__ + ">"
        def __str__(self):
            return self.__repr__() + "\n" + Style("ERROR", self.message).__str__()
        
    class UndefinedPlayMethodError(Exception):
        def __init__(self, engine: type, *args):
            super().__init__(*args)
            self.message = "[Engine] Play method not defined \n|__ You must define the play method of the engine: <" + engine.__name__ + ">"
        def __str__(self):
            return self.__repr__() + "\n" + Style("ERROR", self.message).__str__()
        
        
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
    
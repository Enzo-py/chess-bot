
class Player:
    WHITE = 'w'
    BLACK = 'b' # HARD-CODED in piece.py and server.py

    def __init__(self, color):
        self.color = color

    def play(self, pieces):
        """
        Return the move played by the AI.
        
        :param pieces: list of pieces of the AI
        :type pieces: list
        :return: {"from": (int, int), "to": (int, int)}
        :rtype: dict
        """
        raise NotImplementedError

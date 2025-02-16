try: from .player import Player
except ImportError: from player import Player

import chess

class Game:
    BB_EDGE = chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8 | chess.BB_FILE_A | chess.BB_FILE_H)

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 10 # not really important
    }

    def __init__(self):

        self.board = None

        # Players (can be AI or human)
        self.white = None
        self.black = None

        self.ia_move_handler = None
        self.draw = None
        self.checkmate = None
        self.king_in_check = {chess.WHITE: False, chess.BLACK: False}
        self.last_player = chess.BLACK

    def __str__(self):
        return self.board.__str__()
    
    def __repr__(self):
        return self.board.__repr__()

    def fen(self):
        return self.board.fen()
       
    def load_fen(self, fen):
        self.board = chess.Board(fen)
        return self
    
    def move(self, move: chess.Move):
        # check if the move is legal
        if move not in self.board.legal_moves:
            raise Exception("Illegal move: " + str(move))
        
        self.board.push(move)

        # check for draw
        if self.board.is_stalemate():
            self.draw = "stalemate"
        elif self.board.is_insufficient_material():
            self.draw = "insufficient material"
        elif self.board.is_seventyfive_moves():
            self.draw = "seventyfive moves"
        elif self.board.is_fivefold_repetition():
            self.draw = "fivefold repetition"
        elif self.board.is_checkmate():
            self.checkmate = self.last_player
        
        if self.board.is_check():
            self.king_in_check[self.last_player] = True
        else:
            self.king_in_check[self.last_player] = False
        self.king_in_check[not self.last_player] = False

        self.last_player = chess.WHITE if self.last_player == chess.BLACK else chess.BLACK
    
    def play_engine_move(self):
        """
        Get the move from the AI.
        """
        if self.checkmate is not None or self.draw is not None: return

        if self.board.turn == chess.WHITE and self.white.is_engine:
            move = self.white.play()
        elif self.board.turn == chess.BLACK and self.black.is_engine:
            move = self.black.play()
        else: return
        
        if move is None: # surrender
            raise Exception("No legal moves")
        
        self.move(move)

        if self.ia_move_handler is not None: self.ia_move_handler(move)
    
    def get_score(self, color):
        """
            Return a score, positive if <color> is winning, negative otherwise.
            The score depends of the material balance.
        """
        score = sum(
            self.PIECE_VALUES[piece] * (
                len(self.board.pieces(piece, color)) - len(self.board.pieces(piece, not color))
            )
            for piece in self.PIECE_VALUES
        )

        return score
    
    def get_box_idx(self, *args) -> chess.Piece:
        """
        Get a piece from the board.
        """

        # we want to transform args into a string like "a1"

        if len(args) == 1:
            coords = args[0]
            if type(coords) is int: # already a box index
                return coords
        else:
            coords = (args[0], args[1])

        if type(coords) == tuple:
            x, y = coords
            if type(x) == int:
                x = chr(x + 65)
                y += 1
            coords = x + str(y)

        box_idx = chess.parse_square(coords.lower())
        return box_idx
    
    def get_coords(self, *args) -> tuple:
        """
        Get the coordinates of a box.
        
        :param args: info of the box
        :type args: tuple or str
        :return: coordinates
        :rtype: tuple (int, int)
        """

        if len(args) == 1:
            coords = args[0]
            if type(coords) is tuple: return coords
            if type(coords) is int:
                x = coords % 8
                y = coords // 8
                return (x, y)
        else:
            coords = (args[0], args[1])

        if type(coords) == str:
            x, y = coords[0], int(coords[1])
            x = ord(x) - 65
            y -= 1
            coords = (x, y)
        return coords

    def get_piece(self, *args) -> chess.Piece:
        """
        Get a piece from the board using coordinates.

        :param args: coordinates of the piece
        :type args: tuple or str
        :return: the piece
        :rtype: chess.Piece
        """
        if self.board is None: raise Exception("No board loaded")

        coords = self.get_box_idx(*args)
        return self.board.piece_at(coords)
    
    def find_piece_box(self, piece_to_find: chess.Piece, color, _exception=True) -> int:
        """
        Find the box of a piece.
        """
        for square, piece in self.board.piece_map().items():
            if piece.piece_type == piece_to_find and piece.color == color:
                return square
        if _exception: raise Exception("Piece not found")
        return None
    
    def get_possible_moves(self, *args) -> list:
        """
        Get the possible moves of a piece knowing its coordinates.

        :param args: coordinates of the piece
        :type args: tuple or str
        :return: list of possible moves
        :rtype list[chess.Move]
        """
        if self.board is None: raise Exception("No board loaded")

        coords = self.get_box_idx(*args)
        return [move for move in self.board.legal_moves if move.from_square == coords]

    def play(self, white, black, fen=None):
        """
        Play a game between two players.
        """

        self.white = white
        self.black = black

        if issubclass(type(white), Player):
            white.color = chess.WHITE
            white.game = self
        if issubclass(type(black), Player):
            black.color = chess.BLACK
            black.game = self

        if fen is not None:
            self.load_fen(fen)
        else:
            self.board = chess.Board()

        return self

    def is_game_over(self):
        return self.checkmate is not None or self.draw is not None


if __name__ == "__main__":
    game = Game()
    game.play(white=Player(False), black=Player(False))
    print(game.get_piece('A', 1), "==", game.get_piece('A1'))
    print(game)

    moves = game.get_possible_moves('a2')
    print("all moves of A2:", moves)
    print("captures:", [game.board.san(move) for move in moves if game.board.is_capture(move)])
    print("checks:", [game.board.san(move) for move in moves if game.board.gives_check(move)])

    game.move(chess.Move.from_uci('d2d4'))
    game.move(chess.Move.from_uci('d7d6'))
    game.move(chess.Move.from_uci('d4d5'))
    game.move(chess.Move.from_uci('c7c5'))

    moves = game.get_possible_moves("D5")
    print(game)

    print("all moves of D5:", moves)
    print("captures:", [game.board.san(move) for move in moves if game.board.is_capture(move)])


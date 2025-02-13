import piece as p
from player import Player


class Board:
    """
        Board class represents the chess board.
        Size of the board is 8x8.

          a|b|c|d|e|f|g|h
        8|B| | |W|B| | |W|8  <-- White side: direction = 1
        7| | | | | | | | |7
        6| | | | | | | | |6
        5| | | | | | | | |5
        4| | | | | | | | |4
        3| | | | | | | | |3
        2| | | | | | | | |2
        1|W| | |B|W| | |B|1  <-- Black side: direction = -1
          a|b|c|d|e|f|g|h
    """
    def __init__(self):
        self.board: dict[str, list] = {
            'a': [None] * 8,
            'b': [None] * 8,
            'c': [None] * 8,
            'd': [None] * 8,
            'e': [None] * 8,
            'f': [None] * 8,
            'g': [None] * 8,
            'h': [None] * 8
        }

        self.white_pieces = set()
        self.black_pieces = set()

        self.setup()

    def is_valid_position(self, pos):
        """
        Check if the given position is valid.
        """
        x, y = pos
        if type(x) is not int:
            if x not in self.board: return False
        elif x < 0 or x > 7: return False

        return y >= 0 and y < 8

    def get(self, letter, number=None, _exception=True) -> p.Piece:
        """
        Get the piece at the given position.

        letter: str 
            from 'A' to 'H' OR box name like 'A1'

        number: int 
            from 1 to 8 OR None if letter is box name
        """

        if number is None:
            letter, number = letter[0], int(letter[1])

        if type(letter) is int:
            letter = chr(letter + 97)
            number += 1

        letter = letter.lower()
        number -= 1

        if letter not in self.board:
            if _exception:
                raise ValueError(f"Invalid letter: {letter.upper()}")
            return None
        if number < 0 or number > 7:
            if _exception:
                raise ValueError(f"Invalid number: {number + 1}")
            return None
        
        return self.board[letter][number]
    
    def get_coords(self, pos):
        """
        Get the coordinates of the given position.
        """
        letter, number = pos
        return ord(letter) - 97, int(number) - 1
    
    def move(self, start, end):
        """
        Move the piece from start to end.
        """
        end_coords = self.get_coords(end)

        piece = self.get(start, _exception=False)
        if piece is None:
            raise ValueError(f"No piece at position {start}")
        
        piece_end = self.get(end, _exception=False)
        if piece_end is not None:
            if piece.color == piece_end.color:
                raise ValueError(f"Cannot capture piece of the same color at position {end}")
            
            if piece.color == Player.WHITE:
                self.black_pieces.remove(piece_end)
            else:
                self.white_pieces.remove(piece_end)

        self.board[start[0]][int(start[1]) - 1] = None
        self.board[end[0]][int(end[1]) - 1] = piece

        piece.pos = end_coords

    def to_FEN(self):
        """
        Convert the board to FEN format.
        """
        fen = ""
        # Loop from rank 8 (index 7) down to rank 1 (index 0)
        for i in range(7, -1, -1):
            empty = 0
            # Make sure to iterate files in order (assuming keys 'a' through 'h')
            for letter in sorted(self.board):
                piece = self.board[letter][i]
                if piece is None:
                    empty += 1
                else:
                    if empty > 0:
                        fen += str(empty)
                        empty = 0
                    # Use lowercase for White and uppercase for Black, adjust as needed
                    fen += piece.name.lower() if piece.color == Player.WHITE else piece.name.upper()
            if empty > 0:
                fen += str(empty)
            # Add a slash between ranks, except after the last one
            if i > 0:
                fen += "/"
        return fen

    def from_FEN(self, fen):
        """
        Set the board from the given FEN string.
        """
        rows = fen.split("/")
        for i, row in enumerate(rows):
            x = 0
            for letter in row:
                if letter.isdigit():
                    x += int(letter)
                else:
                    self.board[chr(97 + x)][7 - i] = p.Piece.from_name(letter)
                    x += 1
    
    def setup(self):
        """
        Setup the board with the initial pieces.
        """
        # self.board['a'][0] = p.Rook(Player.BLACK, (0, 0))
        # self.board['b'][0] = p.Knight(Player.BLACK, (1, 0))
        # self.board['c'][0] = p.Bishop(Player.BLACK, (2, 0))
        # self.board['d'][0] = p.Queen(Player.BLACK, (3, 0))
        # self.board['e'][0] = p.King(Player.BLACK, (4, 0))
        # self.board['f'][0] = p.Bishop(Player.BLACK, (5, 0))
        # self.board['g'][0] = p.Knight(Player.BLACK, (6, 0))
        # self.board['h'][0] = p.Rook(Player.BLACK, (7, 0))

        # for i in range(8):
        #     self.board[chr(97 + i)][1] = p.Pawn(Player.BLACK, (i, 1))
        #     self.board[chr(97 + i)][6] = p.Pawn(Player.WHITE, (i, 6))

        # self.board['a'][7] = p.Rook(Player.WHITE, (0, 7))
        # self.board['b'][7] = p.Knight(Player.WHITE, (1, 7))
        # self.board['c'][7] = p.Bishop(Player.WHITE, (2, 7))
        # self.board['d'][7] = p.Queen(Player.WHITE, (3, 7))
        # self.board['e'][7] = p.King(Player.WHITE, (4, 7))
        # self.board['f'][7] = p.Bishop(Player.WHITE, (5, 7))
        # self.board['g'][7] = p.Knight(Player.WHITE, (6, 7))
        # self.board['h'][7] = p.Rook(Player.WHITE, (7, 7))
        self.from_FEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def __repr__(self):
        board = "  a |b |c |d |e |f |g |h \n"
        for i in range(8, 0, -1):
            row = f"{i}|"
            for letter in self.board:
                piece = self.board[letter][i - 1]
                row += f"{piece or '  '}|"
            board += f"{row}{i}\n"
        board += "  a |b |c |d |e |f |g |h "

        return board
            

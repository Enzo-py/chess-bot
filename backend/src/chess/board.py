from . import piece as p
from .player import Player


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

        self.en_passant = None
        self.available_castling = {
            Player.WHITE: {"king": True, "queen": True},
            Player.BLACK: {"king": True, "queen": True}
        }

        self.king_in_check = {
            Player.WHITE: False,
            Player.BLACK: False
        }

        self.before_simulation_state = None
        self.simulated_moves = []
        self.checkmate = None
        self.draw = None

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
        pos: str
            from 'A1' to 'H8'
        
        return: tuple
            (x, y)
        """
        if type(pos) is tuple:
            return pos
        letter, number = pos
        letter, number = letter.lower(), int(number)
        return ord(letter) - 97, int(number) - 1
    
    def get_box(self, coords):
        """
        Get the box name for the given coordinates.
        """
        x, y = coords
        return chr(x + 97) + str(y + 1)
    
    def remove_piece(self, piece):
        """
        Remove the given piece from the board.
        """
        box = self.get_box(piece.pos)
        self.board[box[0].lower()][piece.pos[1]] = None
        if piece.color == Player.WHITE:
            self.white_pieces.remove(piece)
        else:
            self.black_pieces.remove(piece)

    def global_moves_update(self, start, end, piece: p.Piece, simulated=False):
        """
        Update the global board state after a move.

        start: str
            start position

        end: str
            end position

        piece: Piece
            piece moved

        simulated: bool
            if the move is simulated
        """

        # check if castling was played
        # if piece.name == "K" and self.available_castling[piece.color]["king"]:
        #     if end == "G1" or end == "G8":
        #         if not simulated:
        #             self.move("H" + end[1], "F" + end[1])
        #         else:
        #             self.simulate_move("H" + end[1], "F" + end[1])

        # elif piece.name == "K" and self.available_castling[piece.color]["queen"]:
        #     if end == "C1" or end == "C8":
        #         if not simulated:
        #             self.move("A" + end[1], "D" + end[1])
        #         else:
        #             self.simulate_move("A" + end[1], "D" + end[1])
        x_start, y_start = self.get_coords(start)
        x_end, y_end = self.get_coords(end)

        # check castling
        if piece.name == "K" and abs(x_end - x_start) == 2:
            if x_end == 6: # king side
                if not self.available_castling[piece.color]["king"]:
                    raise ValueError("Cannot castle king side")
                
                if not simulated:
                    self.move("H" + start[1], "F" + start[1])
                else:
                    self.simulate_move("H" + start[1], "F" + start[1])
            elif x_end == 2: # queen side
                if not self.available_castling[piece.color]["queen"]:
                    raise ValueError("Cannot castle queen side")
                
                if not simulated:
                    self.move("A" + start[1], "D" + start[1])
                else:
                    self.simulate_move("A" + start[1], "D" + start[1])

        # check if previous en passant was played
        if self.en_passant is not None:
            en_passant = self.get(self.en_passant, _exception=False)
            if en_passant is not None and en_passant.name == "P":
                killed_pawn = self.en_passant[0], self.en_passant[1] + (1 if piece.color == Player.WHITE else -1)
                killed_pawn = self.get(killed_pawn, _exception=False)
                if killed_pawn is not None and killed_pawn.name == "P" and killed_pawn.color != piece.color:
                    self.remove_piece(killed_pawn)

        # update en passant
        if piece.name == "P" and abs(y_end - y_start) == 2:
            self.en_passant = (x_start, (y_start + y_end) // 2)
        else:
            self.en_passant = None

        # update castling rights
        if piece.name == "K":
            self.available_castling[piece.color]["king"] = False
            self.available_castling[piece.color]["queen"] = False
        elif piece.name == "R":
            if piece.color == Player.WHITE:
                if start.upper() == "A1":
                    self.available_castling[Player.WHITE]["queen"] = False
                elif start.upper() == "H1":
                    self.available_castling[Player.WHITE]["king"] = False
            else:
                if start.upper() == "A8":
                    self.available_castling[Player.BLACK]["queen"] = False
                elif start.upper() == "H8":
                    self.available_castling[Player.BLACK]["king"] = False

    def promote(self, pos, piece_name):
        """
        Promote the pawn at the given position to the given piece.
        """
        piece = self.get(pos)
        if piece is None or piece.name != "P":
            raise ValueError(f"No pawn at position {pos}")
        
        if piece.color == Player.WHITE and pos[1] != "8":
            raise ValueError(f"Cannot promote white pawn at position {pos}")
        elif piece.color == Player.BLACK and pos[1] != "1":
            raise ValueError(f"Cannot promote black pawn at position {pos}")

        if piece_name not in ["Q", "R", "B", "N"]:
            raise ValueError(f"Invalid piece name: {piece_name}")

        x, y = self.get_coords(pos)
        new_piece = None
        if piece_name == "Q":
            new_piece = p.Queen(piece.color, (x, y))
        elif piece_name == "R":
            new_piece = p.Rook(piece.color, (x, y))
        elif piece_name == "B":
            new_piece = p.Bishop(piece.color, (x, y))
        elif piece_name == "N":
            new_piece = p.Knight(piece.color, (x, y))

        self.remove_piece(piece)
        self.board[pos[0].lower()][int(pos[1]) - 1] = new_piece
        if piece.color == Player.WHITE:
            self.white_pieces.add(new_piece)
        else:
            self.black_pieces.add(new_piece)

        self.check_state()

    def check_state(self, simulated=False):
        self.king_in_check[Player.WHITE] = self.check_nb_attackers(self.get_king(Player.WHITE).pos, Player.BLACK) > 0
        self.king_in_check[Player.BLACK] = self.check_nb_attackers(self.get_king(Player.BLACK).pos, Player.WHITE) > 0

        if simulated: return
        # check for checkmate
        white_has_move = False
        black_has_move = False
        
        if self.king_in_check[Player.WHITE]:
            for piece in self.white_pieces:
                if len(piece.get_possible_moves(self)) > 0 or len(piece.get_possible_captures(self)) > 0:
                    white_has_move = True
                    break
        else:
            white_has_move = True

        if self.king_in_check[Player.BLACK]:
            for piece in self.black_pieces:
                if len(piece.get_possible_moves(self)) > 0 or len(piece.get_possible_captures(self)) > 0:
                    black_has_move = True
                    break
        else:
            black_has_move = True

        if not white_has_move:
            self.checkmate = Player.WHITE
        else: 
            self.checkmate = None

        if not black_has_move:
            self.checkmate = Player.BLACK
        else:
            self.checkmate = None

    
    def check_for_draw(self, color, half_moves):
        # 0. check for half moves
        # 1. check insufficient material
        # 2. check 3-fold repetition
        # 3. check stalemate

        if half_moves >= 50:
            self.draw = 'fifty-half-moves'
            return

        if color == Player.WHITE:
            focus_pieces = self.white_pieces
        else:
            focus_pieces = self.black_pieces

        # nb_pieces = {}
        # for piece in focus_pieces:
        #     if piece.name not in nb_pieces:
        #         nb_pieces[piece.name] = 0
        #     nb_pieces[piece.name] += 1

        # if len(focus_pieces) == 1: 
        #     self.draw = 'insufficient-material'
        #     return
        # if len(focus_pieces) == 2:
        #     if "B" in nb_pieces or "N" in nb_pieces:
        #         self.draw = 'insufficient-material'
        #         return
                
        # 2. check 3-fold repetition
        ...

        # 3. check stalemate
        if self.king_in_check[color]:
            self.draw = None
            return
        for piece in focus_pieces:
            if len(piece.get_possible_moves(self)) > 0 or len(piece.get_possible_captures(self)) > 0:
                self.draw = None
                return
            
        self.draw = 'stalemate'
                

    def move(self, start, end):
        """
        Move the piece from start to end.
        """
        end_coords = self.get_coords(end)
        piece = self.get(start, _exception=False)
        self.global_moves_update(start, end, piece)

        if piece is None:
            raise ValueError(f"No piece at position {start}")
        
        piece_end = self.get(end, _exception=False)
        if piece_end is not None:
            if piece.color == piece_end.color:
                raise ValueError(f"Cannot capture piece of the same color at position {end}")
            
            self.remove_piece(piece_end)

        self.board[start[0].lower()][int(start[1]) - 1] = None
        self.board[end[0].lower()][int(end[1]) - 1] = piece

        piece.pos = end_coords
        self.check_state()
        
    def simulate_move(self, start, end):
        """
        Simulate the move from start to end.
        """
        if type(start) is tuple:
            start = self.get_box(start)
        if type(end) is tuple:
            end = self.get_box(end)

        if self.before_simulation_state is None:
            self.before_simulation_state = self.available_castling.copy()

        piece = self.get(start, _exception=False)

        self.global_moves_update(start, end, piece, simulated=True)
            
        if piece is None:
            raise ValueError(f"No piece at position {start}")
        
        piece_end = self.get(end, _exception=False)
        current_move = {
            "from": start,
            "to": end,
            "piece_start": piece,
            "piece_end": piece_end
        }
        self.simulated_moves.append(current_move)

        self.board[start[0].lower()][int(start[1]) - 1] = None
        self.board[end[0].lower()][int(end[1]) - 1] = piece

        piece.pos = self.get_coords(end)
        if piece_end is not None: piece_end.pos = None
        self.check_state(simulated=True)


    def undo_simulated_move(self):
        """
        Undo the last simulated move.
        """
        if len(self.simulated_moves) == 0:
            return
        
        if self.before_simulation_state is not None: # can be wrong
            self.available_castling = self.before_simulation_state
            self.before_simulation_state = None
            
        
        last_move = self.simulated_moves.pop()
        start = last_move["from"]
        end = last_move["to"]
        piece_start = last_move["piece_start"]
        piece_end = last_move["piece_end"]

        self.board[start[0].lower()][int(start[1]) - 1] = piece_start
        self.board[end[0].lower()][int(end[1]) - 1] = piece_end

        piece_start.pos = self.get_coords(start)
        if piece_end is not None:
            piece_end.pos = self.get_coords(end)

        self.global_moves_update(end, start, piece_start, simulated=True)
        self.check_state(simulated=True)

    def get_king(self, color):
        """
        Get the king of the given color.
        """
        for piece in self.white_pieces if color == Player.WHITE else self.black_pieces:
            if isinstance(piece, p.King):
                return piece
        raise Exception(f"No king found for color {color} ... LOL")

    def check_nb_attackers(self, pos, color):
        """
        Check the number of attackers for the given position on a given box.
        """
        attackers = 0
        if color == Player.WHITE:
            for piece in self.white_pieces:
                if pos in piece.get_possible_captures(self, check_attackers=True) or pos in piece.get_possible_moves(self, check_attackers=True):
                    attackers += 1
        else:
            for piece in self.black_pieces:
                if pos in piece.get_possible_captures(self, check_attackers=True) or pos in piece.get_possible_moves(self, check_attackers=True):
                    attackers += 1

        return attackers     

    def get_pieces(self, color) -> list[p.Piece]:
        """
        Get the pieces of the given color.
        """
        return list(self.white_pieces if color == Player.WHITE else self.black_pieces)              

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
                    fen += piece.fen()
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
                    self.board[chr(97 + x)][7 - i] = p.Piece.from_name(letter, (x, 7 - i))
                    if letter.isupper():
                        self.white_pieces.add(self.board[chr(97 + x)][7 - i])
                    else:
                        self.black_pieces.add(self.board[chr(97 + x)][7 - i])
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
            

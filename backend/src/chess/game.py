from .board import Board
from .player import Player

class Game:

    def __init__(self):
        self.board = Board()
        self.current_player = Player.WHITE
        self.player_white = Player(Player.WHITE)
        self.player_black = Player(Player.BLACK)

        self.nb_moves = 0
        self.nb_half_moves = 0
        """Nb moves since last capture or pawn move: used to detect draw"""
        self.available_castling = {
            Player.WHITE: {"king": True, "queen": True},
            Player.BLACK: {"king": True, "queen": True}
        }
        self.en_passant = None

        # Players (can be AI or human)
        self.white = None
        self.black = None

    def to_FEN(self):
        left_part = self.board.to_FEN()
        right_part = f"{'w' if self.current_player == Player.WHITE else 'b'}"
        white_castling = " "
        black_castling = " "
        if self.available_castling[Player.WHITE]["king"]:
            white_castling += "K"
        if self.available_castling[Player.WHITE]["queen"]:
            white_castling += "Q"
        if self.available_castling[Player.BLACK]["king"]:
            black_castling += "k"
        if self.available_castling[Player.BLACK]["queen"]:
            black_castling += "q"
        right_part += white_castling if white_castling else " -"
        right_part += black_castling if black_castling else " -"
        right_part += f" {self.board.get_box(self.en_passant) if self.en_passant is not None else '-'}"
        right_part += f" {self.nb_half_moves} {self.nb_moves}"
        return f"{left_part} {right_part}"
    
    def from_FEN(self, fen):
        left_part = fen.split(" ")[0]
        right_part = fen.split(" ")[1:]

        self.board.from_FEN(left_part)
        self.current_player = Player.WHITE if right_part[0] == "w" else Player.BLACK
        self.available_castling = {
            Player.WHITE: {"king": False, "queen": False},
            Player.BLACK: {"king": False, "queen": False}
        }
        if "K" in right_part:
            self.available_castling[Player.WHITE]["king"] = True
        if "Q" in right_part:
            self.available_castling[Player.WHITE]["queen"] = True
        if "k" in right_part:
            self.available_castling[Player.BLACK]["king"] = True
        if "q" in right_part:
            self.available_castling[Player.BLACK]["queen"] = True
        
        self.en_passant = self.board.get_coords(right_part[3]) if right_part[3] != "-" else None
        self.nb_half_moves = int(right_part[4])
        self.nb_moves = int(right_part[5])

    def move(self, start, end):
        piece = self.board.get(start)
        if piece is None:
            raise Exception("No piece at position", start)

        if piece.color != self.current_player:
            raise Exception("Not your turn")

        end_coords = self.board.get_coords(end)
        if not self.board.is_valid_position(end_coords):
            raise Exception("Invalid position", end)
        
        start_coords = self.board.get_coords(start)
        self._update_state(piece, start_coords, end_coords)

        self.board.move(start, end)

    def _update_state(self, piece, start, end):
        """
        Update the global game state after a move.
            - Update the number of moves
            - Update the number of half moves
            - Update the castling rights
            - Update the en passant square
        """
        self.nb_moves += 1
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        if piece.name == "P" or self.board.get(end) is not None:
            self.nb_half_moves = 0
        else:
            self.nb_half_moves += 1

        # check if previous en passant was played
        if self.en_passant is not None:
            en_passant = self.board.get(self.en_passant, _exception=False)
            if en_passant is not None and en_passant.name == "P":
                killed_pawn = self.en_passant[0], self.en_passant[1] + (1 if piece.color == Player.WHITE else -1)
                killed_pawn = self.board.get(killed_pawn, _exception=False)
                if killed_pawn is not None and killed_pawn.name == "P" and killed_pawn.color != piece.color:
                    self.board.remove_piece(killed_pawn)

        x_start, y_start = start
        x_end, y_end = end
        
        if piece.name == "P" and abs(y_end - y_start) == 2:
            self.en_passant = (x_start, (y_start + y_end) // 2)
        else:
            self.en_passant = None

        self.board.en_passant = self.en_passant

        if piece.name == "K":
            self.available_castling[piece.color]["king"] = False
            self.available_castling[piece.color]["queen"] = False
        elif piece.name == "R":
            if (x_start == 0 and y_start == 0) or (x_start == 0 and y_start == 7):
                self.available_castling[piece.color]["queen"] = False
            elif (x_start == 7 and y_start == 0) or (x_start == 0 and y_start == 7):
                self.available_castling[piece.color]["king"] = False

    def play(self, white, black):
        self.white = white
        self.black = black


if __name__ == "__main__":
    game = Game()
    print(game.board.get('A', 1), "==", game.board.get('A1'))
    print(game.board)

    print(game.board.get('A2').get_possible_moves(game.board))
    print(game.board.get('A7').get_possible_captures(game.board))

    game.move('a7', 'a5')
    game.move('a2', 'a3')
    game.move('a5', 'a4')
    game.move('b2', 'b4')
    print(game.board)
    print(game.board.get('a4').get_possible_captures(game.board))
    print(game.to_FEN())


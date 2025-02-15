from .engine import Engine, Piece

import chess
import numpy as np

class GreedyAI(Engine):
    """
    Greedy AI that try to maximize gain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_move_piece = None

    def play(self) -> dict:
        """
        Return the move played by the AI.
        
        :return: {"from": (int, int), "to": (int, int), ["promotion": str]}
        :rtype: dict
        """

        all_moves: list[chess.Move] = list(self.game.board.legal_moves)
        all_proba = []
        for move in all_moves:
            all_proba.append(self.get_action_score(move))
        
        # to avoid having a probability of 0 or negative (-> sum of probabilities = 0)
        if len(all_proba) == 0: return None

        choice_idx = np.argmax(all_proba)
        choice = all_moves[choice_idx]
        self.last_move_piece = self.game.get_piece(choice.from_square).piece_type
        return choice

            
    def get_action_score(self, move: chess.Move) -> int:
        """
        Get the score of an action.
        
        :param action: action to evaluate
        :type action: dict
        :return: score of the action
        :rtype: int
        """
        _from = self.game.get_coords(move.from_square)
        _to = self.game.get_coords(move.to_square)
        _from_idx = self.game.get_box_idx(_from)
        _to_idx = self.game.get_box_idx(_to)

        end_piece = self.game.get_piece(_to)
        end_piece = end_piece.piece_type if end_piece is not None else None
        value = 0

        ennemy_attackers = len(self.game.board.attackers(chess.WHITE if self.color == chess.BLACK else chess.BLACK, _to_idx))
        from_piece = self.game.get_piece(_from).piece_type

        # Score on the piece taken
        if end_piece is not None:
            value += (self.game.PIECE_VALUES[end_piece]+ 3) ** 2

        # Avoid moving the same piece twice
        if self.last_move_piece == from_piece:
            value -= 10

        # Score on the piece moved (encourage development, and large moves)
        if from_piece == chess.PAWN:
            value += 1 + (_from[1] in [0, 7]) * 3 + (abs(_to[1] - _from[1]) == 2) * 3 + (end_piece is not None) * 3
        elif from_piece == chess.KNIGHT:
            value += 2.5
            # avoid moving the knight on the edge
            if _to[0] in [0, 7] or _to[1] in [0, 7]:
                value -= 6
            if _from[0] in [0, 7] or _from[1] in [0, 7]:
                value += 1
        elif from_piece == chess.BISHOP:
            value += 3 + (abs(_to[0] - _from[0]) + abs(_to[1] - _from[1])) / 3
            # avoid to be in the bottom or top middle
            if _to[0] in [2, 3, 4, 5] and _to[1] in [0, 7]:
                value -= 5
            
            # encourage moving from ...
            if _from[0] in [2, 3, 4, 5] and _from[1] in [0, 7]:
                value += 2

        elif from_piece == chess.ROOK:
            value += 0 + (abs(_to[0] - _from[0]) + abs(_to[1] - _from[1])) / 2


        elif from_piece == chess.QUEEN:
            value += 1.5
        elif from_piece == chess.KING:
            if abs(_to[0] - _from[0]) == 2:
                value += 6
            else:
                value -= 40 # avoid moving the king

        # Score on the promotion
        if move.promotion is not None:
            value += (self.game.PIECE_VALUES[move.promotion] - 1) * 2

        # Avoid to move on an uncontrolled square
        value += (len(self.game.board.attackers(self.color, _to_idx)) - ennemy_attackers) * 2

        # ennemy king possible moves:
        ennemy_king_box_idx = self.game.find_piece_box(chess.KING, chess.WHITE if self.color == chess.BLACK else chess.BLACK)
        ennemy_king_moves = self.game.get_possible_moves(ennemy_king_box_idx)

        # simulate the move
        self.game.board.push(move)

        # get the new possible moves
        ennemy_king_box_idx = self.game.find_piece_box(chess.KING, chess.WHITE if self.color == chess.BLACK else chess.BLACK)
        ennemy_king_moves_after = self.game.get_possible_moves(ennemy_king_box_idx)
        is_check = self.game.king_in_check[chess.WHITE if self.color == chess.BLACK else chess.BLACK]

        # Avoid to move on an uncontrolled square after the move
        value += 4 * (len(self.game.board.attackers(self.color, _to_idx)) - len(self.game.board.attackers(chess.WHITE if self.color == chess.BLACK else chess.BLACK, _to_idx)) - 1)
        
        self.game.board.pop() # cancel the move

        # if we are loosing the piece if we stay, move
        if len(self.game.board.attackers(self.color, _from_idx)) < len(self.game.board.attackers(chess.WHITE if self.color == chess.BLACK else chess.BLACK, _from_idx)):
            value += self.game.PIECE_VALUES[from_piece] * 2
        
        # avoid to let the ennemy take a high value piece
        if ennemy_attackers > 0:
            value -= self.game.PIECE_VALUES[from_piece] * 5
        
        # try to diminue king mobility
        value += (len(ennemy_king_moves) - len(ennemy_king_moves_after)) * 2

        # check for mate or draw
        if not is_check and len(ennemy_king_moves_after) == 0 and self.game.get_score(self.color) > 6:
            value = -200 # draw when we are winning
        elif not is_check and len(ennemy_king_moves_after) == 0 and self.game.get_score(self.color) < 1:
            value += 200 # mate when we are loosing
        elif is_check and len(ennemy_king_moves_after) == 0 and ennemy_attackers == 0:
            value = 1e6

        value += int(is_check)
        # increase the value if the move is a check or reduce king move and piece is rook or queen
        if (is_check or len(ennemy_king_moves_after) - len(ennemy_king_moves) < 0) and from_piece in [chess.ROOK, chess.QUEEN]:
            value += 10

        return value
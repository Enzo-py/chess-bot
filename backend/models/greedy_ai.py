from .algo import Algo, Piece

import numpy as np

class GreedyAI(Algo):
    """
    Greedy AI that try to maximize gain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_move_piece = None

    def play(self, pieces: list[Piece]) -> dict:
        """
        Return the move played by the AI.
        
        :param pieces: list of pieces of the AI
        :type pieces: list[Piece]
        :return: {"from": (int, int), "to": (int, int), ["promotion": str]}
        :rtype: dict
        """

        all_actions = []
        all_proba = []
        for piece in pieces:
            actions = piece.get_possible_actions(self.game.board)
            probabilities = [self.get_action_score(action) for action in actions]
            all_actions.extend(actions)
            all_proba.extend(probabilities)
        
        # to avoid having a probability of 0 or negative (-> sum of probabilities = 0)
        if len(all_proba) == 0: return None
        
        choice_idx = np.argmax(all_proba)
        choice = all_actions[choice_idx]
        self.last_move_piece = self.game.board.get(self.game.board.get_coords(choice["from"]))
        return choice

            
    def get_action_score(self, action: dict) -> int:
        """
        Get the score of an action.
        
        :param action: action to evaluate
        :type action: dict
        :return: score of the action
        :rtype: int
        """
        _from = self.game.board.get_coords(action["from"])
        _to = self.game.board.get_coords(action["to"])

        end_piece = self.game.board.get(_to)
        value = 0
        if end_piece is not None:
            value += (end_piece.value + 3) ** 2

        if self.last_move_piece == self.game.board.get(_from):
            value -= 18

        from_piece = self.game.board.get(_from)
        if from_piece.name == "P":
            value += 2 + (_from[1] in [0, 7]) * 3 + (abs(_to[1] - _from[1]) == 2) * 2 + (end_piece is not None) * 3
        elif from_piece.name == "N":
            value += 3
        elif from_piece.name == "B":
            value += 3
        elif from_piece.name == "R":
            value += 0 + (abs(_to[0] - _from[0]) + abs(_to[1] - _from[1])) / 2
        elif from_piece.name == "Q":
            value += 1.5
        elif from_piece.name == "K":
            if abs(_to[0] - _from[0]) == 2:
                value += 6
            else:
                value -= 40

        if action.get("promote") is not None:
            value += Piece.value_of(action["promote"]) - 1

        value += (self.game.board.check_nb_attackers(_to, self.color) - self.game.board.check_nb_attackers(_to, 'b' if self.color == 'w' else 'w')) * 2

        # ennemy king possible moves:
        ennemy_king = self.game.board.get_king('w' if self.color == 'b' else 'b')
        ennemy_king_moves = ennemy_king.get_possible_actions(self.game.board)

        # simulate the move
        self.game.board.simulate_move(action["from"], action["to"])

        # # if the piece can be taken by a lower value piece
        # for ennemy_piece in self.game.board.get_pieces('w' if self.color == 'b' else 'b'):
        #     if ennemy_piece.value < from_piece.value:
        #         if action["to"].lower() in ennemy_piece.get_possible_captures(self.game.board):
        #             value /= (from_piece.value - ennemy_piece.value + 0.5)
        #             value -= 3

        value += 50 * (self.game.board.check_nb_attackers(_to, self.color) - self.game.board.check_nb_attackers(_to, 'b' if self.color == 'w' else 'w'))

        ennemy_king_moves_after = ennemy_king.get_possible_actions(self.game.board)
        is_check = self.game.board.king_in_check[ennemy_king.color]
        self.game.board.undo_simulated_move()

        # if we are loosing the piece if we stay
        if self.game.board.check_nb_attackers(_from, self.color) < self.game.board.check_nb_attackers(_from, 'b' if self.color == 'w' else 'w'):
            value += from_piece.value * 2
        
        if self.game.board.check_nb_attackers(_to, 'w' if self.color == 'b' else 'b') > 0:
            value -= from_piece.value * 2

        value += (len(ennemy_king_moves) - len(ennemy_king_moves_after)) * 3
        if not is_check and len(ennemy_king_moves_after) == 0 and self.game.get_score(self.color) > 6:
            value = -200
        elif is_check and len(ennemy_king_moves_after) == 0 and self.game.board.check_nb_attackers(_to, 'w' if self.color == 'b' else 'b') == 0:
            value = 1e6

        value += int(is_check)
        return value
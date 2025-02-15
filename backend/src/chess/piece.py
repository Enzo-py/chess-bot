
class Piece:

    def __init__(self, color, pos):
        """
        color: int
            w for white, b for black

        pos: tuple[int, int] 
            (x, y) position on the board
        """
        self.color = color
        """w for white, b for black"""

        self.pos = pos
        self.name = ''
        self.value = 0

        self.strict_movement = False
        """
            If True, only the exact patterns will be considered, otherwise 
            subpatterns will be considered, ex: (0, 8) -> (0, 1), (0, 2), ...
        """

        self.can_jump_over_pieces = False

        self.movement_patterns: list[tuple[int, int]] = []
        """[(x, y), ...] list of possible patterns for movement"""

        self.capture_patterns: list[tuple[int, int]] = []
        """[(x, y), ...] list of possible patterns for capture"""

        self.conditional_patterns: dict[str, tuple[int, int]] = {}
        """{name: (x, y)} list of possible conditional patterns"""


    def __str__(self):
        color = self.color.upper()
        return f'{color}{self.name}'

    def __repr__(self):
        return f'[{self.color}] {self.__class__.__name__} at {self.pos}'

    def fen(self):
        return self.name.lower() if self.color == 'b' else self.name.upper()

    def move(self, new_pos):
        self.pos = new_pos

    def get_possible_moves(self, board, check_attackers=False) -> list[tuple[int, int]]:
        possible_moves = []
        if self.pos is None: return possible_moves
        y_direction = 1 if self.color == 'w' else -1

        # check for pion attack in move, cuz pion is special
        if check_attackers and self.name == 'P':
            for pattern in self.capture_patterns:
                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
                if board.is_valid_position(new_pos): possible_moves.append(new_pos)
            return possible_moves

        for pattern in self.movement_patterns:
            x, y = pattern

            if not self.strict_movement:
                all_subpatterns = set()

                # (8, 8), (8, -8), (-8, 8), (-8, -8): exemple bishop
                x_direction = 1 if x > 0 else -1
                y_direction = 1 if y > 0 else -1

                for idx in range(1, max(abs(x), abs(y))):
                    if x_direction == 1:
                        x_i = min(idx, x)
                    else:
                        x_i = max(-idx, x)

                    if y_direction == 1:
                        y_i = min(idx, y)
                    else:
                        y_i = max(-idx, y)

                    new_x = self.pos[0] + x_i
                    new_y = self.pos[1] + y_i

                    if new_x < 0 or new_x > 7 or new_y < 0 or new_y > 7: continue
                    all_subpatterns.add((new_x, new_y))

                # check if position is valid, if not and not jump over pieces, break the direction
                # print(self.name, sorted([board.get_box(m) for m in all_subpatterns]))

                subpatterns_forwards_left = set()
                subpatterns_forwards_right = set()
                subpatterns_backwards_left = set()
                subpatterns_backwards_right = set()
                subpatterns_forwards = set()
                subpatterns_backwards = set()
                subpatterns_left = set()
                subpatterns_right = set()

                for subpattern in all_subpatterns:
                    if subpattern[0] > self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_forwards_right.add(subpattern)
                    elif subpattern[0] > self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_forwards_left.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_backwards_right.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_backwards_left.add(subpattern)
                    elif subpattern[0] == self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_forwards.add(subpattern)
                    elif subpattern[0] == self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_backwards.add(subpattern)
                    elif subpattern[0] > self.pos[0] and subpattern[1] == self.pos[1] * -y_direction:
                        subpatterns_right.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] == self.pos[1] * -y_direction:
                        subpatterns_left.add(subpattern)
                
                subpatterns = {
                    "f-r": sorted(subpatterns_forwards_right),
                    "f-l": sorted(subpatterns_forwards_left, reverse=y_direction == 1),
                    "b-r": sorted(subpatterns_backwards_right, reverse=y_direction == 1),
                    "b-l": sorted(subpatterns_backwards_left, reverse=y_direction == -1),
                    "f": sorted(subpatterns_forwards, reverse=y_direction == -1),
                    "b": sorted(subpatterns_backwards, reverse=y_direction == -1),
                    "r": sorted(subpatterns_right, reverse=y_direction == 1),
                    "l": sorted(subpatterns_left, reverse=y_direction == -1)
                }

                for subpatterns_list in subpatterns.values():
                    for new_pos in subpatterns_list:
                        if not self._check_move(new_pos, board):
                            if not self.can_jump_over_pieces: break
                            continue
                        possible_moves.append(new_pos)

            # basic movement
            new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
            if self._check_move(new_pos, board):
                possible_moves.append(new_pos)
              
        for name, pattern in self.conditional_patterns.items():
            if check_attackers: break
            if name == 'first_move' and (
                (self.color == 'w' and self.pos[1] == 1) or
                (self.color == 'b' and self.pos[1] == 6)
            ): 

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
                interm_pos = (self.pos[0] + x, self.pos[1] + 1 * y_direction)
                if self._check_move(new_pos, board) and self._check_move(interm_pos, board):
                    possible_moves.append(new_pos)
            
            elif name == 'kingside-castling':
                if self.color == 'w' and board.available_castling[self.color]["king"]:
                    all_free_positions = [(i, 0) for i in range(5, 7)]
                    all_free_positions = [board.check_nb_attackers(pos, 'b') == 0 and board.get(pos) is None for pos in all_free_positions]

                    # check rook is here
                    if board.get((7, 0)) is None or board.get((7, 0)).name != 'R': continue
                    if all(all_free_positions): possible_moves.append((6, 0))

                elif self.color == 'b' and board.available_castling[self.color]["king"]:
                    all_free_positions = [(i, 7) for i in range(5, 7)]
                    all_free_positions = [board.check_nb_attackers(pos, 'w') == 0 and board.get(pos) is None for pos in all_free_positions]

                    # check rook is here
                    if board.get((7, 7)) is None or board.get((7, 7)).name != 'R': continue
                    if all(all_free_positions): possible_moves.append((6, 7))

            elif name == 'queenside-castling':
                if self.color == 'w' and board.available_castling[self.color]["queen"]:
                    all_free_positions = [(i, 0) for i in range(1, 4)]
                    all_free_positions = [board.check_nb_attackers(pos, 'b') == 0 and board.get(pos) is None for pos in all_free_positions]

                    # check rook is here
                    if board.get((0, 0)) is None or board.get((0, 0)).name != 'R': continue
                    if all(all_free_positions): possible_moves.append((2, 0))

                elif self.color == 'b' and board.available_castling[self.color]["queen"]:
                    all_free_positions = [(i, 7) for i in range(1, 4)]
                    all_free_positions = [board.check_nb_attackers(pos, 'w') == 0 and board.get(pos) is None for pos in all_free_positions]

                    # check rook is here
                    if board.get((0, 7)) is None or board.get((0, 7)).name != 'R': continue
                    if all(all_free_positions): possible_moves.append((2, 7))

        if not check_attackers and self.name == 'K':
            for move in possible_moves.copy():
                if board.check_nb_attackers(move, 'b' if self.color == 'w' else 'w') > 0:
                    possible_moves.remove(move)
        
        # check if the king is not in check after the move
        if not check_attackers:
            for move in possible_moves.copy():
                board.simulate_move(start=self.pos, end=move)
                if board.check_nb_attackers(board.get_king(self.color).pos, 'b' if self.color == 'w' else 'w') > 0:
                    possible_moves.remove(move)
                board.undo_simulated_move()

        if not check_attackers and board.king_in_check[self.color]:
            for move in possible_moves.copy():
                board.simulate_move(start=self.pos, end=move)
                if board.king_in_check[self.color]:
                    possible_moves.remove(move)
                board.undo_simulated_move()
        return possible_moves
    
    def get_possible_captures(self, board, check_attackers=False) -> list[tuple[int, int]]:
        possible_captures = []
        if self.pos is None: return possible_captures
        y_direction = 1 if self.color == 'w' else -1

        for pattern in self.capture_patterns:
            x, y = pattern

            if not self.strict_movement:
                all_subpatterns = set()

                # (8, 8), (8, -8), (-8, 8), (-8, -8): exemple bishop
                x_direction = 1 if x > 0 else -1
                y_direction = 1 if y > 0 else -1

                for idx in range(1, max(abs(x), abs(y))):
                    if x_direction == 1:
                        x_i = min(idx, x)
                    else:
                        x_i = max(-idx, x)

                    if y_direction == 1:
                        y_i = min(idx, y)
                    else:
                        y_i = max(-idx, y)

                    new_x = self.pos[0] + x_i
                    new_y = self.pos[1] + y_i

                    if new_x < 0 or new_x > 7 or new_y < 0 or new_y > 7: continue
                    all_subpatterns.add((new_x, new_y))

                # check if position is valid, if not and not jump over pieces, break the direction
                # print(self.name, sorted([board.get_box(m) for m in all_subpatterns]))

                subpatterns_forwards_left = set()
                subpatterns_forwards_right = set()
                subpatterns_backwards_left = set()
                subpatterns_backwards_right = set()
                subpatterns_forwards = set()
                subpatterns_backwards = set()
                subpatterns_left = set()
                subpatterns_right = set()

                for subpattern in all_subpatterns:
                    if subpattern[0] > self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_forwards_right.add(subpattern)
                    elif subpattern[0] > self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_forwards_left.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_backwards_right.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_backwards_left.add(subpattern)
                    elif subpattern[0] == self.pos[0] and subpattern[1] > self.pos[1] * -y_direction:
                        subpatterns_forwards.add(subpattern)
                    elif subpattern[0] == self.pos[0] and subpattern[1] < self.pos[1] * -y_direction:
                        subpatterns_backwards.add(subpattern)
                    elif subpattern[0] > self.pos[0] and subpattern[1] == self.pos[1] * -y_direction:
                        subpatterns_right.add(subpattern)
                    elif subpattern[0] < self.pos[0] and subpattern[1] == self.pos[1] * -y_direction:
                        subpatterns_left.add(subpattern)
                
                subpatterns = {
                    "f-r": sorted(subpatterns_forwards_right),
                    "f-l": sorted(subpatterns_forwards_left, reverse=y_direction == 1),
                    "b-r": sorted(subpatterns_backwards_right, reverse=y_direction == 1),
                    "b-l": sorted(subpatterns_backwards_left, reverse=y_direction == -1),
                    "f": sorted(subpatterns_forwards, reverse=y_direction == -1),
                    "b": sorted(subpatterns_backwards, reverse=y_direction == -1),
                    "r": sorted(subpatterns_right, reverse=y_direction == 1),
                    "l": sorted(subpatterns_left, reverse=y_direction == -1)
                }

                for subpatterns_list in subpatterns.values():
                    for new_pos in subpatterns_list:
                        if self._check_capturable(new_pos, board): possible_captures.append(new_pos)
                        if not self._check_move(new_pos, board):
                            if not self.can_jump_over_pieces: break
                            continue           

            # basic movement
            new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
            if self._check_capturable(new_pos, board):
                possible_captures.append(new_pos)

        for name, pattern in self.conditional_patterns.items():
            if name == 'en_passant-left':
                attack_pos = (self.pos[0] - 1, self.pos[1])
                if not (
                    board.is_valid_position(attack_pos)
                    and board.get(attack_pos) is not None 
                    and board.get(attack_pos).name == 'P' 
                    and board.get(attack_pos).color != self.color
                ): continue

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)

                if not board.en_passant == new_pos: continue

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
                if board.is_valid_position(new_pos) and board.get(new_pos) is None:
                    possible_captures.append(new_pos)

            if name == 'en_passant-right':
                attack_pos = (self.pos[0] + 1, self.pos[1])
                if not (
                    board.is_valid_position(attack_pos)
                    and board.get(attack_pos) is not None 
                    and board.get(attack_pos).name == 'P' 
                    and board.get(attack_pos).color != self.color
                ): continue

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)

                if not board.en_passant == new_pos: continue

                
                if board.is_valid_position(new_pos) and board.get(new_pos) is None:
                    possible_captures.append(new_pos)

        # check if the king is not in check after the move
        if not check_attackers and self.name == 'K':
            for move in possible_captures.copy():
                if board.check_nb_attackers(move, 'b' if self.color == 'w' else 'w') > 0:
                    possible_captures.remove(move)
        
        # check if the king is not in check after the move
        if not check_attackers:
            for move in possible_captures.copy():
                board.simulate_move(start=self.pos, end=move)
                if board.check_nb_attackers(board.get_king(self.color).pos, 'b' if self.color == 'w' else 'w') > 0:
                    possible_captures.remove(move)
                board.undo_simulated_move()

        if not check_attackers and board.king_in_check[self.color]:
            for move in possible_captures.copy():
                board.simulate_move(start=self.pos, end=move)
                if board.king_in_check[self.color]:
                    possible_captures.remove(move)
                board.undo_simulated_move()
        return possible_captures
    
    def get_possible_actions(self, board) -> list[dict]:
        """
        Get ALL possibles actions for this piece at this position.

        :return: list of actions, action: {"from": BoxName, "to": BoxName, ["promotion": str]}
        :rtype: list[dict - Action]
        """

        possible_actions = []
        for move in self.get_possible_moves(board):
            if self.name == "P" and (
                (self.color == 'w' and move[1] == 7) or
                (self.color == 'b' and move[1] == 0)
            ):
                for promotion in ['Q', 'R', 'N', 'B']:
                    possible_actions.append({
                        "from": board.get_box(self.pos),
                        "to": board.get_box(move),
                        "promote": promotion
                    })
            else:
                possible_actions.append({
                    "from": board.get_box(self.pos),
                    "to": board.get_box(move),
                })

        for capture in self.get_possible_captures(board):
            if self.name == "P" and (
                (self.color == 'w' and capture[1] == 7) or
                (self.color == 'b' and capture[1] == 0)
            ):
                for promotion in ['Q', 'R', 'N', 'B']:
                    possible_actions.append({
                        "from": board.get_box(self.pos),
                        "to": board.get_box(capture),
                        "promote": promotion
                    })
            else:
                possible_actions.append({
                    "from": board.get_box(self.pos),
                    "to": board.get_box(capture),
                })

        return possible_actions
    
    def _check_move(self, new_pos, board) -> bool:
        """
        Check if the move is valid.

        WARNING: This doesn't take into account the possibility to jump over pieces (or not).
        WARNING: This doesn't take into account the state of the game (check, checkmate, ...).
        """
        if not board.is_valid_position(new_pos):
            return False
        
        if not board.get(new_pos) is None:
            return False

        return True
    
    def _check_capturable(self, new_pos, board) -> bool:
        """
        Check if the capture is valid.

        WARNING: This doesn't take into account the possibility to jump over pieces (or not).
        WARNING: This doesn't take into account the state of the game (check, checkmate, ...).
        """
        if not board.is_valid_position(new_pos):
            return False
        
        if board.get(new_pos) is None:
            return False

        if board.get(new_pos).color == self.color:
            return False

        return True

    @staticmethod
    def from_name(name, pos):
        color = 'w' if name.isupper() else 'b'
        name = name.upper()
        if name == 'P':
            return Pawn(color, pos)
        if name == 'R':
            return Rook(color, pos)
        if name == 'N':
            return Knight(color, pos)
        if name == 'B':
            return Bishop(color, pos)
        if name == 'Q':
            return Queen(color, pos)
        if name == 'K':
            return King(color, pos)
        
        raise ValueError(f'Invalid piece name: {name}')
    
    @staticmethod
    def value_of(name):
        p = Piece.from_name(name, (0, 0))
        return p.value

class Pawn(Piece):

    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'P'
        self.value = 1
        self.strict_movement = True

        self.movement_patterns.extend([
            (0, 1)
        ])
        
        self.capture_patterns.extend([
            (-1, 1),
            (1, 1)
        ])

        self.conditional_patterns.update({
            'first_move': (0, 2),
            'en_passant-left': (-1, 1),
            'en_passant-right': (1, 1)
        })

class Rook(Piece):

    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'R'
        self.value = 5
        self.strict_movement = False

        self.movement_patterns.extend([
            (0, 8),
            (0, -8),
            (8, 0),
            (-8, 0)
        ])

        self.capture_patterns.extend(self.movement_patterns)

class Knight(Piece):
    
    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'N'
        self.value = 3
        self.strict_movement = True
        self.can_jump_over_pieces = True

        self.movement_patterns.extend([
            (1, 2),
            (2, 1),
            (-1, 2),
            (-2, 1),
            (1, -2),
            (2, -1),
            (-1, -2),
            (-2, -1)
        ])

        self.capture_patterns.extend(self.movement_patterns)

class Bishop(Piece):
    
    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'B'
        self.value = 3
        self.strict_movement = False

        self.movement_patterns.extend([
            (8, 8),
            (8, -8),
            (-8, 8),
            (-8, -8)
        ])

        self.capture_patterns.extend(self.movement_patterns)

class Queen(Piece):
    
    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'Q'
        self.value = 9
        self.strict_movement = False

        self.movement_patterns.extend([
            (0, 8),
            (0, -8),
            (8, 0),
            (-8, 0),
            (8, 8),
            (8, -8),
            (-8, 8),
            (-8, -8)
        ])

        self.capture_patterns.extend(self.movement_patterns)

class King(Piece):
    
    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'K'
        self.value = 100
        self.strict_movement = True

        self.movement_patterns.extend([
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ])

        self.capture_patterns.extend(self.movement_patterns)

        self.conditional_patterns.update({
            'kingside-castling': (2, 0),
            'queenside-castling': (-2, 0)
        })

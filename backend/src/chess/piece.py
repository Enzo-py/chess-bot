
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

    def get_possible_moves(self, board) -> list[tuple[int, int]]:
        possible_moves = []
        y_direction = 1 if self.color == 'w' else -1

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
            if name == 'first_move' and (
                (self.color == 'w' and self.pos[1] == 1) or
                (self.color == 'b' and self.pos[1] == 6)
            ): 

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
                if self._check_move(new_pos, board):
                    possible_moves.append(new_pos)
                
        return possible_moves
    
    def get_possible_captures(self, board) -> list[tuple[int, int]]:
        possible_captures = []
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

        return possible_captures
    
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

class Pawn(Piece):

    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'P'
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

        self.conditional_patterns.update({
            'kingside-castling': (2, 0),
            'queenside-castling': (-2, 0)
        })

class Bishop(Piece):
    
    def __init__(self, color, pos):
        super().__init__(color, pos)
        self.name = 'B'
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

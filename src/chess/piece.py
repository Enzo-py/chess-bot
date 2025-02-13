
class Piece:

    def __init__(self, color, pos):
        """
        color: int
            0 for white, 1 for black

        pos: tuple[int, int] 
            (x, y) position on the board
        """
        self.color = color
        """0 for white, 1 for black"""

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
        color = 'W' if self.color == 0 else 'B'
        return f'{color}{self.name}'

    def __repr__(self):
        return f'[{self.color}] {self.__class__.__name__} at {self.pos}'

    def move(self, new_pos):
        self.pos = new_pos

    def get_possible_moves(self, board):
        possible_moves = []
        y_direction = 1 if self.color == 1 else -1

        for pattern in self.movement_patterns:
            x, y = pattern

            if not self.strict_movement:
                all_subpatterns = set()

                # calc all subpatterns
                cpy_x, cpy_y = abs(x), abs(y)
                for i in range(1, max(cpy_x, cpy_y) + 1):
                    cpy_x -= 1
                    cpy_y -= 1 * y_direction

                    all_subpatterns.add(max(0, cpy_x), max(0, cpy_y))

                # transform subpatterns to position (forwards and backwards)
                forwards_left_positions = set()
                forwards_right_positions = set()
                backwards_left_positions = set()
                backwards_right_positions = set()
                for subpattern in all_subpatterns:
                    sub_x, sub_y = subpattern
                    # align subpattern with current position
                    sub_x -= self.pos[0]
                    sub_y -= self.pos[1]

                    # forwards
                    if y > 0 and x > 0:
                        forwards_right_positions.add((sub_x, sub_y))
                    elif y > 0 and x < 0:
                        forwards_left_positions.add((sub_x, sub_y))
                    # backwards
                    elif y < 0 and x > 0:
                        backwards_right_positions.add((sub_x, sub_y))
                    elif y < 0 and x < 0:
                        backwards_left_positions.add((sub_x, sub_y))

                # check if position is valid, if not and not jump over pieces, break the direction  
                for new_pos in forwards_right_positions:
                    if not self._check_move(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_moves.append(new_pos)

                for new_pos in forwards_left_positions:
                    if not self._check_move(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_moves.append(new_pos)

                for new_pos in backwards_right_positions:
                    if not self._check_move(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_moves.append(new_pos)

                for new_pos in backwards_left_positions:
                    if not self._check_move(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_moves.append(new_pos)

            # basic movement
            new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
            if self._check_move(new_pos, board):
                possible_moves.append(new_pos)
              
        for name, pattern in self.conditional_patterns.items():
            if name == 'first_move' and (
                (self.color == 1 and self.pos[1] == 1) or
                (self.color == 0 and self.pos[1] == 6)
            ): 

                x, y = pattern
                new_pos = (self.pos[0] + x, self.pos[1] + y * y_direction)
                if self._check_move(new_pos, board):
                    possible_moves.append(new_pos)
                
        return possible_moves
    
    def get_possible_captures(self, board):
        possible_captures = []
        y_direction = 1 if self.color == 1 else -1

        for pattern in self.capture_patterns:
            x, y = pattern

            if not self.strict_movement:
                all_subpatterns = set()

                # calc all subpatterns
                cpy_x, cpy_y = abs(x), abs(y)
                for i in range(1, max(cpy_x, cpy_y) + 1):
                    cpy_x -= 1
                    cpy_y -= 1 * y_direction

                    all_subpatterns.add(max(0, cpy_x), max(0, cpy_y))

                # transform subpatterns to position (forwards and backwards)
                forwards_left_positions = set()
                forwards_right_positions = set()
                backwards_left_positions = set()
                backwards_right_positions = set()
                for subpattern in all_subpatterns:
                    sub_x, sub_y = subpattern
                    # align subpattern with current position
                    sub_x -= self.pos[0]
                    sub_y -= self.pos[1]

                    # forwards
                    if y > 0 and x > 0:
                        forwards_right_positions.add((sub_x, sub_y))
                    elif y > 0 and x < 0:
                        forwards_left_positions.add((sub_x, sub_y))
                    # backwards
                    elif y < 0 and x > 0:
                        backwards_right_positions.add((sub_x, sub_y))
                    elif y < 0 and x < 0:
                        backwards_left_positions.add((sub_x, sub_y))

                # check if position is valid, if not and not jump over pieces, break the direction  
                for new_pos in forwards_right_positions:
                    if not self._check_capturable(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_captures.append(new_pos)

                for new_pos in forwards_left_positions:
                    if not self._check_capturable(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_captures.append(new_pos)

                for new_pos in backwards_right_positions:
                    if not self._check_capturable(new_pos, board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_captures.append(new_pos)

                for new_pos in backwards_left_positions:
                    if not self._check_capturable(new_pos
                    , board):
                        if not self.can_jump_over_pieces:
                            break
                        continue
                    possible_captures.append(new_pos)

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
                if self._check_capturable(new_pos, board):
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
    def from_name(name):
        color = 1 if name.isupper() else 0
        name = name.upper()
        if name == 'P':
            return Pawn(color, (0, 0))
        if name == 'R':
            return Rook(color, (0, 0))
        if name == 'N':
            return Knight(color, (0, 0))
        if name == 'B':
            return Bishop(color, (0, 0))
        if name == 'Q':
            return Queen(color, (0, 0))
        if name == 'K':
            return King(color, (0, 0))
        
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

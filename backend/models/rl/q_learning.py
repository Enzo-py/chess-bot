from models.engine import Engine
import chess
import random
import numpy as np
import time
import pickle
import os
from collections import defaultdict

class QLearningAI(Engine):
    """
    Q-Learning based Chess AI that learns to play through experience.
    Uses reinforcement learning to discover effective chess strategies.
    """

    __author__ = "Shashwat SHARMA"
    __description__ = "Chess AI that uses Q-Learning algorithm to learn and select moves."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Q-learning parameters
        self.alpha = 0.1       # Learning rate
        self.gamma = 0.9       # Discount factor
        self.epsilon = 0.1     # Exploration rate (decreases over time)
        self.min_epsilon = 0.01  # Minimum exploration rate
        
        # Q-table - maps state-action pairs to values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Piece values for evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Learning tracking
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0
        self.games_played = 0
        
        # Thinking time
        self.thinking_time = 1.0  # Seconds to think
        
        # Opening book - common first moves
        self.opening_book = {
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -': ['e2e4', 'd2d4', 'g1f3', 'c2c4'],
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
            'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3': ['d7d5', 'g8f6', 'e7e6', 'c7c6'],
        }

        # Try to load Q-table if it exists
        self._load_q_table()
    
    def setup(self):
        """Setup the engine before playing."""
        # Reset learning params for a new game
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0
        self.games_played += 1
        
        # Decay epsilon with more games played
        self.epsilon = max(self.min_epsilon, 0.1 * (0.95 ** self.games_played))
        
        return self

    def play(self) -> chess.Move:
        """
        Select the best move using Q-learning.
        """
        start_time = time.time()
        
        # Check for opening book moves
        board_fen = self.game.board.fen()
        if board_fen in self.opening_book:
            book_moves = self.opening_book[board_fen]
            legal_book_moves = [move for move in book_moves 
                               if chess.Move.from_uci(move) in self.game.board.legal_moves]
            if legal_book_moves:
                move_uci = random.choice(legal_book_moves)
                return chess.Move.from_uci(move_uci)
        
        # Get current state representation
        current_state = self._get_state_representation(self.game.board)
        
        # Learn from previous move if we have one
        if self.previous_state is not None and self.previous_action is not None:
            current_reward = self._get_reward(self.game.board)
            self._update_q_values(self.previous_state, self.previous_action, 
                                 current_reward, current_state)
            self.previous_reward = current_reward
        
        # Get all legal moves
        legal_moves = list(self.game.board.legal_moves)
        if not legal_moves:
            return None
        
        # Special handling for endgame positions
        if self._is_endgame(self.game.board):
            return self._get_endgame_move(legal_moves)
        
        # Select move using epsilon-greedy policy
        if random.random() < self.epsilon:
            # Exploration: choose a random move
            selected_move = random.choice(legal_moves)
        else:
            # Exploitation: choose the best move according to Q-values
            selected_move = self._get_best_action(current_state, legal_moves)
        
        # Update previous state and action for next learning step
        self.previous_state = current_state
        self.previous_action = self._move_to_action_key(selected_move)
        
        # If we're close to out of time, save Q-table
        if time.time() - start_time >= 0.9 * self.thinking_time:
            self._save_q_table()
        
        # Return the selected move
        return selected_move
    
    def _get_state_representation(self, board):
        """
        Create a compact state representation that captures essential features
        while reducing the massive state space of chess.
        """
        # Material count
        material_counts = []
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material_counts.append(len(board.pieces(piece_type, chess.WHITE)))
            material_counts.append(len(board.pieces(piece_type, chess.BLACK)))
        
        # Center control (e4, d4, e5, d5)
        center_control = []
        for square in [chess.E4, chess.D4, chess.E5, chess.D5]:
            piece = board.piece_at(square)
            if piece is None:
                center_control.append(0)
            elif piece.color == chess.WHITE:
                center_control.append(1)
            else:
                center_control.append(2)
        
        # King safety (distance from center for endgame)
        w_king_sq = board.king(chess.WHITE)
        b_king_sq = board.king(chess.BLACK)
        
        w_king_file = chess.square_file(w_king_sq) if w_king_sq is not None else 0
        w_king_rank = chess.square_rank(w_king_sq) if w_king_sq is not None else 0
        b_king_file = chess.square_file(b_king_sq) if b_king_sq is not None else 0
        b_king_rank = chess.square_rank(b_king_sq) if b_king_sq is not None else 0
        
        w_king_center_dist = abs(w_king_file - 3.5) + abs(w_king_rank - 3.5)
        b_king_center_dist = abs(b_king_file - 3.5) + abs(b_king_rank - 3.5)
        
        # Castling rights
        castling_rights = [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK))
        ]
        
        # Game phase (opening, middlegame, endgame)
        total_pieces = sum(1 for _ in board.piece_map())
        game_phase = 0  # opening
        if total_pieces <= 16:
            game_phase = 2  # endgame
        elif total_pieces <= 24:
            game_phase = 1  # middlegame
        
        # Combine all features into a hashable tuple
        features = (
            tuple(material_counts) + 
            tuple(center_control) + 
            (int(w_king_center_dist), int(b_king_center_dist)) + 
            tuple(castling_rights) +
            (game_phase,) +
            (board.turn,)  # Whose turn it is
        )
        
        return features
    
    def _move_to_action_key(self, move):
        """Convert a chess.Move to a hashable key for the Q-table."""
        return move.uci()
    
    def _get_reward(self, board):
        """
        Calculate the immediate reward for the current board state.
        Higher value indicates better position for the AI's color.
        """
        # Check game termination conditions
        if board.is_checkmate():
            return 1000 if board.turn != self.color else -1000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Material advantage
        material_advantage = 0
        for piece_type, value in self.piece_values.items():
            material_advantage += (
                len(board.pieces(piece_type, chess.WHITE)) * value -
                len(board.pieces(piece_type, chess.BLACK)) * value
            )
        
        # Adjust for AI's color
        if self.color == chess.BLACK:
            material_advantage = -material_advantage
        
        # Center control
        center_advantage = 0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == self.color:
                    center_advantage += 10
                else:
                    center_advantage -= 10
        
        # Piece development and activity
        development_score = 0
        
        # Knights and bishops off home rank
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for piece_square in board.pieces(piece_type, self.color):
                # Reward for being off the first rank
                if self.color == chess.WHITE and chess.square_rank(piece_square) > 0:
                    development_score += 5
                elif self.color == chess.BLACK and chess.square_rank(piece_square) < 7:
                    development_score += 5
        
        # Castling status
        castling_score = 0
        if self.color == chess.WHITE:
            # King in castled position
            if board.king(chess.WHITE) in [chess.G1, chess.C1]:
                castling_score += 30
            # Still has castling rights
            elif board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
                castling_score += 15
        else:
            # King in castled position
            if board.king(chess.BLACK) in [chess.G8, chess.C8]:
                castling_score += 30
            # Still has castling rights
            elif board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
                castling_score += 15
        
        # Mobility (number of legal moves)
        mobility_score = 0
        board_copy = board.copy()
        
        # Set the board to AI's turn if it's not already
        if board_copy.turn != self.color:
            board_copy.turn = self.color
        
        # Count legal moves for AI's color
        ai_mobility = len(list(board_copy.legal_moves))
        
        # Switch turn and count opponent's moves
        board_copy.turn = not board_copy.turn
        opponent_mobility = len(list(board_copy.legal_moves))
        
        # Calculate mobility advantage
        mobility_score = (ai_mobility - opponent_mobility) / 10.0
        
        # King safety
        king_safety_score = 0
        king_square = board.king(self.color)
        if king_square:
            # Get surrounding squares
            surrounding_squares = []
            king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
            
            for f in range(max(0, king_file-1), min(8, king_file+2)):
                for r in range(max(0, king_rank-1), min(8, king_rank+2)):
                    if (f, r) != (king_file, king_rank):  # Exclude the king's square
                        surrounding_squares.append(chess.square(f, r))
            
            # Count defenders
            defenders = 0
            for sq in surrounding_squares:
                piece = board.piece_at(sq)
                if piece and piece.color == self.color:
                    defenders += 1
                    
            king_safety_score += defenders * 5
        
        # Normalize and combine scores
        normalized_material = material_advantage / 1000.0  # Normalize to roughly -1 to 1
        normalized_center = center_advantage / 40.0  # Normalize to -1 to 1
        normalized_development = development_score / 20.0  # Normalize to 0 to 1
        normalized_castling = castling_score / 30.0  # Normalize to 0 to 1
        normalized_king_safety = king_safety_score / 20.0  # Normalize to 0 to 1
        
        # Combine with different weights
        total_reward = (
            3.0 * normalized_material +  # Material is most important
            1.0 * normalized_center +    # Center control
            0.5 * normalized_development +  # Development
            1.0 * normalized_castling +  # Castling
            1.0 * mobility_score +       # Mobility
            1.5 * normalized_king_safety  # King safety
        )
        
        return total_reward
    
    def _update_q_values(self, state, action, reward, next_state):
        """
        Update Q-values using the Q-learning formula:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # Get the best value for the next state
        legal_moves = list(self.game.board.legal_moves)
        max_next_q = 0
        
        if legal_moves:
            max_next_q = self._get_max_q_value(next_state, legal_moves)
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def _get_max_q_value(self, state, legal_moves):
        """Get the maximum Q-value for the given state and set of legal moves."""
        max_q = float('-inf')
        
        for move in legal_moves:
            action_key = self._move_to_action_key(move)
            q_value = self.q_table[state][action_key]
            max_q = max(max_q, q_value)
        
        return max_q if max_q != float('-inf') else 0
    
    def _get_best_action(self, state, legal_moves):
        """
        Get the best action (move) for the current state based on Q-values.
        If multiple actions have the same value, use heuristics to break ties.
        """
        best_q = float('-inf')
        best_moves = []
        
        # Find all moves with the highest Q-value
        for move in legal_moves:
            action_key = self._move_to_action_key(move)
            q_value = self.q_table[state][action_key]
            
            if q_value > best_q:
                best_q = q_value
                best_moves = [move]
            elif q_value == best_q:
                best_moves.append(move)
        
        # If we have multiple best moves, use heuristics to break ties
        if len(best_moves) > 1:
            return self._select_best_heuristic_move(best_moves)
        
        # Otherwise, return the single best move
        return best_moves[0] if best_moves else legal_moves[0]
    
    def _select_best_heuristic_move(self, moves):
        """
        Select the best move from a set of candidates using heuristics.
        Used to break ties when Q-values are equal.
        """
        best_move = moves[0]
        best_score = float('-inf')
        
        for move in moves:
            score = 0
            board = self.game.board
            
            # Prioritize captures by piece value difference
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]/2
            
            # Bonus for checks
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_check():
                score += 150
            
            # Bonus for center control
            if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                score += 50
            
            # Bonus for castling
            if board.is_castling(move):
                score += 200
                
            # Bonus for promotion
            if move.promotion:
                score += self.piece_values[move.promotion]
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _is_endgame(self, board):
        """Determine if the current position is in the endgame phase."""
        # Count major pieces
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        minor_pieces = (
            len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
            len(board.pieces(chess.KNIGHT, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.WHITE)) + 
            len(board.pieces(chess.BISHOP, chess.BLACK))
        )
        
        # If there are few major pieces, it's likely an endgame
        return queens == 0 or (queens <= 1 and rooks <= 2 and minor_pieces <= 2)
    
    def _get_endgame_move(self, legal_moves):
        """
        Special move selection for endgame positions.
        In endgames, we prioritize king activity and pawn advancement.
        """
        board = self.game.board
        best_move = legal_moves[0]
        best_score = float('-inf')
        
        for move in legal_moves:
            score = 0
            
            # Capture opponent pieces when possible
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    score += self.piece_values[victim.piece_type]
            
            # Advance pawns toward promotion
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Distance to promotion
                if piece.color == chess.WHITE:
                    score += chess.square_rank(move.to_square) * 10
                else:
                    score += (7 - chess.square_rank(move.to_square)) * 10
                    
                # Bonus for passed pawns
                if self._is_passed_pawn(board, move.to_square, piece.color):
                    score += 50
            
            # Activate king in endgame
            if piece and piece.piece_type == chess.KING:
                # Move king to center in endgame
                center_distance = (
                    abs(chess.square_file(move.to_square) - 3.5) + 
                    abs(chess.square_rank(move.to_square) - 3.5)
                )
                score += (4 - center_distance) * 10
                
                # Get king closer to opponent king in winning positions
                if self._has_material_advantage(board):
                    opponent_king = board.king(not piece.color)
                    if opponent_king:
                        king_distance = chess.square_distance(move.to_square, opponent_king)
                        score += (8 - king_distance) * 5
            
            # Check and checkmate
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                score += 10000
            elif board_copy.is_check():
                score += 200
            
            # Promotion
            if move.promotion:
                score += self.piece_values[move.promotion]
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _is_passed_pawn(self, board, square, color):
        """Check if a pawn is a passed pawn (no enemy pawns ahead of it)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # For white pawns
        if color == chess.WHITE:
            for r in range(rank + 1, 8):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    s = chess.square(f, r)
                    p = board.piece_at(s)
                    if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                        return False
        # For black pawns
        else:
            for r in range(0, rank):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    s = chess.square(f, r)
                    p = board.piece_at(s)
                    if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                        return False
        
        return True
    
    def _has_material_advantage(self, board):
        """Determine if the AI has a material advantage."""
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.piece_values[pt] 
                            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.piece_values[pt] 
                            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        
        if self.color == chess.WHITE:
            return white_material > black_material
        else:
            return black_material > white_material
    
    def _save_q_table(self):
        """Save the Q-table to a file."""
        # Only save significant entries to reduce file size
        filtered_q_table = {}
        for state, actions in self.q_table.items():
            if actions:  # If there are any actions for this state
                filtered_actions = {a: v for a, v in actions.items() if abs(v) > 0.01}
                if filtered_actions:  # Only save if there are significant values
                    filtered_q_table[state] = filtered_actions
        
        try:
            with open('q_table_chess.pkl', 'wb') as f:
                pickle.dump(filtered_q_table, f)
        except Exception as e:
            print(f"Failed to save Q-table: {e}")
    
    def _load_q_table(self):
        """Load the Q-table from a file if it exists."""
        try:
            if os.path.exists('q_table_chess.pkl'):
                with open('q_table_chess.pkl', 'rb') as f:
                    loaded_table = pickle.load(f)
                    
                    # Convert to defaultdict
                    for state, actions in loaded_table.items():
                        for action, value in actions.items():
                            self.q_table[state][action] = value
                    
                    print(f"Loaded Q-table with {len(loaded_table)} states")
        except Exception as e:
            print(f"Failed to load Q-table: {e}")
            # If loading fails, continue with an empty Q-table
    
    def train(self, num_episodes=100, opponent_engine=None):
        """
        Train the Q-learning AI by playing against itself or another engine.
        
        Args:
            num_episodes (int): Number of games to play for training
            opponent_engine (Engine): Optional opponent engine class to play against
        """
        from src.chess.game import Game
        from src.chess.simulation import Simulation
        
        print(f"Starting training for {num_episodes} episodes...")
        
        # If no opponent specified, play against a copy of self with fixed epsilon
        if opponent_engine is None:
            opponent = QLearningAI()
            opponent.epsilon = 0.2  # Fixed exploration rate for opponent
        else:
            opponent = opponent_engine()
        
        wins = 0
        draws = 0
        losses = 0
        
        for episode in range(num_episodes):
            # Create a new game
            game = Game()
            
            # Randomly assign colors
            if random.random() < 0.5:
                self.color = chess.WHITE
                opponent.color = chess.BLACK
                white_player = self
                black_player = opponent
            else:
                self.color = chess.BLACK
                opponent.color = chess.WHITE
                white_player = opponent
                black_player = self
            
            # Set up the game
            self.game = game
            opponent.game = game
            
            # Reset learning state
            self.previous_state = None
            self.previous_action = None
            
            # Play until game over
            while not game.is_game_over():
                # White's turn
                if game.board.turn == chess.WHITE:
                    move = white_player.play()
                else:
                    move = black_player.play()
                
                if move:
                    game.move(move)
            
            # Game result
            if game.checkmate == self.color:
                losses += 1
                result = "Loss"
            elif game.checkmate is not None:
                wins += 1
                result = "Win"
            else:
                draws += 1
                result = "Draw"
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}: {result} - W:{wins} D:{draws} L:{losses}")
            
            # Save Q-table periodically
            if (episode + 1) % 25 == 0:
                self._save_q_table()
        
        # Final save
        self._save_q_table()
        print(f"Training completed. Results: Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"Final Q-table size: {sum(len(actions) for actions in self.q_table.values())} action-values")
        
        return wins, draws, losses
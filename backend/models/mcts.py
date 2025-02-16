from .engine import Engine
import chess
import random
import math
import time
from typing import Dict, List, Optional, Tuple

class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        
    def ucb1(self, exploration_weight: float) -> float:
        """Calculate UCB1 value with tunable exploration weight."""
        if self.visits == 0:
            return float('inf')
        if not self.parent:
            return float('-inf')
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return exploitation + exploration

class MonteCarloAI(Engine):
    """
    Monte Carlo Tree Search AI that uses MCTS algorithm to select moves. 
    Implemented some heuristics and optimizations that guide the search more effectively. Instead of purely random playouts, it uses move selection based on piece values (MVV-LVA principle) and positional advantages . Improved evaluation function. weighted random selection explore/exploit. Check previous commit for the initial simple version.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuration parameters
        self.exploration_weight = 0.7  # Tuned down from sqrt(2) for more exploitation
        self.simulation_depth = 30     # Reduced from 50 for efficiency
        self.thinking_time = 1.0       # Seconds to think
        
        # Piece values for evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def play(self) -> chess.Move:
        """
        Return the move played by the MCTS AI.
        
        :return: Selected chess move or None if no legal moves exist
        :rtype: chess.Move or None
        """
        actions = list(self.game.board.legal_moves)
        if not actions:
            return None

        # Create root node and run MCTS
        root = MCTSNode(self.game.board)
        end_time = time.time() + self.thinking_time

        while time.time() < end_time:
            node = self._select(root)
            if node.untried_moves:
                node = self._expand(node)
            simulation_result = self._simulate(node)
            self._backpropagate(node, simulation_result)

        # If no children were created (time too short), use smart fallback
        if not root.children:
            return self._get_best_fallback_move(actions)

        # Select best child based on visits and win rate
        best_child = max(root.children, 
                        key=lambda c: (c.visits, c.wins/c.visits if c.visits > 0 else 0))
        return best_child.move

    def _get_best_fallback_move(self, moves: List[chess.Move]) -> chess.Move:
        """Choose the best move when MCTS fails using simple heuristics."""
        best_move = moves[0]
        best_score = float('-inf')
        
        for move in moves:
            score = 0
            # Prioritize captures by piece value difference
            if self.game.board.is_capture(move):
                victim = self.game.board.piece_at(move.to_square)
                attacker = self.game.board.piece_at(move.from_square)
                if victim and attacker:
                    score += self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]/2
            
            # Bonus for checks and center control
            if self.game.board.gives_check(move):
                score += 150
            if move.to_square in [27, 28, 35, 36]:  # Center squares
                score += 50
                
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a promising node for expansion using UCB1."""
        while not node.untried_moves and node.children:
            node = max(node.children, key=lambda n: n.ucb1(self.exploration_weight))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the selected node with a new child."""
        move = self._choose_untried_move(node)
        if not move:
            return node

        new_board = node.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, parent=node, move=move)
        node.children.append(child)
        return child

    def _choose_untried_move(self, node: MCTSNode) -> Optional[chess.Move]:
        """Choose an untried move using MVV-LVA principle."""
        if not node.untried_moves:
            return None

        # Score moves based on MVV-LVA and other heuristics
        scored_moves = []
        for move in node.untried_moves:
            score = 0
            # Capturing moves (MVV-LVA)
            if node.board.is_capture(move):
                victim = node.board.piece_at(move.to_square)
                attacker = node.board.piece_at(move.from_square)
                if victim and attacker:
                    score += self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]/2
            
            # Additional heuristics
            if node.board.gives_check(move):
                score += 150
            if move.to_square in [27, 28, 35, 36]:
                score += 50
            
            scored_moves.append((move, score))

        # Select move probabilistically based on scores
        total_score = sum(max(1, score) for _, score in scored_moves)
        choice = random.random() * total_score
        
        current_sum = 0
        for move, score in scored_moves:
            current_sum += max(1, score)
            if current_sum > choice:
                node.untried_moves.remove(move)
                return move
                
        # Fallback to first move if something goes wrong
        move = node.untried_moves.pop()
        return move

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a game from the node's position."""
        board = node.board.copy()
        moves_played = 0

        while not board.is_game_over() and moves_played < self.simulation_depth:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = self._choose_simulation_move(board, legal_moves)
            board.push(move)
            moves_played += 1

        return self._evaluate_position(board, moves_played)

    def _choose_simulation_move(self, board: chess.Board, moves: List[chess.Move]) -> chess.Move:
        """Choose a move for simulation using enhanced heuristics."""
        scored_moves = []
        for move in moves:
            score = 1  # Base score
            
            # Capture value
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]/2
            
            # Positional bonuses
            if move.to_square in [27, 28, 35, 36]:  # Center control
                score += 30
            if board.gives_check(move):
                score += 50
                
            scored_moves.append((move, max(1, score)))
            
        # Weighted random selection
        total_score = sum(score for _, score in scored_moves)
        choice = random.random() * total_score
        
        current_sum = 0
        for move, score in scored_moves:
            current_sum += score
            if current_sum > choice:
                return move
        
        return scored_moves[0][0]  # Fallback

    def _evaluate_position(self, board: chess.Board, depth: int) -> float:
        """Evaluate the position with comprehensive draw handling and material counting."""
        # Checkmate
        if board.is_checkmate():
            return 1.0 if board.turn != self.game.board.turn else 0.0
            
        # Draw conditions
        if (board.is_stalemate() or 
            board.is_insufficient_material() or
            board.can_claim_threefold_repetition() or
            board.can_claim_fifty_moves()):
            return 0.45  # Slightly worse than unclear position
            
        # Material evaluation for non-terminal positions
        material_score = 0
        for piece_type, value in self.piece_values.items():
            material_score += (
                len(board.pieces(piece_type, chess.WHITE)) * value -
                len(board.pieces(piece_type, chess.BLACK)) * value
            )
        
        # Normalize to [0, 1] range, centered at 0.5
        normalized_score = 0.5 + material_score / (self.piece_values[chess.QUEEN] * 4)
        return max(0.0, min(1.0, normalized_score))

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate the simulation result through the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1.0 - result  # Flip result for opponent

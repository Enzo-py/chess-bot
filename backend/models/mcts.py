from .engine import Engine
import chess
import random
import math
import time
from typing import Optional, List

class MCTSNode:
    """Simple node in the MCTS tree."""
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)

    def ucb1(self, exploration: float = 1.414) -> float:
        """Calculate UCB1 value of this node."""
        if self.visits == 0:
            return float('inf')
        if not self.parent:
            return float('-inf')
            
        exploitation = self.wins / self.visits
        exploration = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MonteCarloAI(Engine):
    """
    Simple Monte Carlo Tree Search AI that plays chess.
    """

    def play(self) -> chess.Move:
        """
        Return the move played by the MCTS AI.
        
        :return: Selected chess move or None if no legal moves exist
        :rtype: chess.Move or None
        """
        # Check for legal moves
        actions = list(self.game.board.legal_moves)
        if not actions:
            return None

        # Create root node
        root = MCTSNode(self.game.board)
        
        # Run MCTS for 1 second
        end_time = time.time() + 1.0
        while time.time() < end_time:
            # 1. Selection - Select promising node
            node = self._select(root)
            
            # 2. Expansion - Add a new child node
            if node.untried_moves:
                node = self._expand(node)
            
            # 3. Simulation - Play random moves until game end
            result = self._simulate(node)
            
            # 4. Backpropagation - Update statistics
            self._backpropagate(node, result)

        # Return most visited child's move
        if not root.children:
            return random.choice(actions)
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using UCB1."""
        while not node.untried_moves and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add a new child node to the tree."""
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        # Create new board position
        child_board = node.board.copy()
        child_board.push(move)
        
        # Create and store child node
        child = MCTSNode(child_board, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Play random moves until game ends."""
        board = node.board.copy()
        max_moves = 40  # Limit playout length
        
        while not board.is_game_over() and max_moves > 0:
            moves = list(board.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            board.push(move)
            max_moves -= 1

        # Game result from node's perspective
        if board.is_checkmate():
            return 1.0 if board.turn != node.board.turn else 0.0
        return 0.5  # Draw

    def _backpropagate(self, node: MCTSNode, result: float):
        """Update statistics for all parent nodes."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # Flip result for opponent
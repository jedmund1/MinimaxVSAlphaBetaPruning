from typing import Dict, Tuple, Optional, Union

from adversarial_search_problem import (
    Action,
    State as GameState,
)
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem


def minimax(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth: float = float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the minimax algorithm for adversarial search problems.
    
    Minimax is a recursive algorithm used for finding the optimal move in two-player,
    zero-sum games. It assumes both players play optimally by maximizing their own
    score and minimizing their opponent's score.

    Args:
        asp: A HeuristicAdversarialSearchProblem representing the game.
        cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
                     Uses heuristic evaluation when cutoff is reached.

    Returns:
        Tuple containing:
            - Best action to take from the current state
            - Dictionary with search statistics including 'states_expanded'
    """
    best_action = None
    stats = {
        'states_expanded': 0
    }

    # TODO: Implement the minimax algorithm. Feel free to write helper functions.

    return best_action, stats


def alpha_beta(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth: float = float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implements the alpha-beta pruning algorithm for adversarial search.
    
    Alpha-beta pruning is an optimization of the minimax algorithm that eliminates
    branches that cannot possibly influence the final decision. It maintains two
    values (alpha and beta) representing the best options found so far for the
    maximizing and minimizing players respectively.

    Args:
        asp: A HeuristicAdversarialSearchProblem representing the game.
        cutoff_depth: Maximum search depth (0 = start state, 1 = one move ahead).
                     Uses heuristic evaluation when cutoff is reached.

    Returns:
        Tuple containing:
            - Best action to take from the current state
            - Dictionary with search statistics including 'states_expanded'
    """
    best_action = None
    stats = {
        'states_expanded': 0  # Increase by 1 for every state transition
    }
    
    # TODO: Implement the alpha-beta pruning algorithm. Feel free to use helper functions.
    
    return best_action, stats
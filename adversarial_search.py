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



    start = asp.get_start_state()
    stats['states_expanded'] += 1
    actions = asp.get_available_actions(start)

    current_player = start.player_to_move()
 
    #initialize "first run" of min and maxval

    if current_player == 0:
        best_v = float('-inf')
        for action in actions:
            next_state = asp.transition(start, action)
            v = min_value(asp, next_state, 1, cutoff_depth, stats)
            if v > best_v:
                best_v = v
                best_action = action
    else:
        best_v = float('inf')
        for action in actions:
            next_state = asp.transition(start, action)
            v = max_value(asp, next_state, 1, cutoff_depth, stats)
            if v < best_v:
                best_v = v
                best_action = action


    return best_action, stats

def min_value(asp : HeuristicAdversarialSearchProblem[GameState, Action], state: GameState, depth: int, cutoff_depth: float, stats: Dict[str,int]):
    '''
    Docstring for min_value
    Return the minimum value you can get from the given state
    :param asp: Description
    :type asp: HeuristicAdversarialSearchProblem[GameState, Action]
    :param state: Description
    :type state: GameState
    :param depth: Description
    :type depth: int
    :param cutoff_depth: Description
    :type cutoff_depth: float
    :param stats: Description
    :type stats: Dict[str, int]
    '''

    #check if terminal state
    if asp.is_terminal_state(state):
        return asp.get_result(state)
    
    #when reaches max depth, apply heuristic function to evaluate states
    if depth >= cutoff_depth:
        return asp.heuristic(state)
    
    #we are expanding one state, want to increment from -inf
    stats['states_expanded'] += 1
    v = float('inf')
    actions = asp.get_available_actions(state)

    #for every possible move, compare it to v, if less store the next state and call minimax on it
    for action in actions:
        next_state = asp.transition(state, action)
        v = min(v, max_value(asp, next_state, depth + 1, cutoff_depth, stats))
    return v


def max_value(asp : HeuristicAdversarialSearchProblem[GameState, Action], state: GameState, depth: int, cutoff_depth: float, stats: Dict[str,int]):
    '''
    Docstring for max_value
    Return the maximum value you can get from the given state
    :param asp: Description
    :type asp: HeuristicAdversarialSearchProblem[GameState, Action]
    :param state: Description
    :type state: GameState
    :param depth: Description
    :type depth: int
    :param cutoff_depth: Description
    :type cutoff_depth: float
    :param stats: Description
    :type stats: Dict[str, int]
    '''

    #check if terminal state
    if asp.is_terminal_state(state):
        return asp.get_result(state)
    
    #when reaches max depth, apply heuristic function to evaluate states
    if depth >= cutoff_depth:
        return asp.heuristic(state)
    
    #we are expanding one state, want to increment from -inf
    stats['states_expanded'] += 1
    v = float('-inf')
    actions = asp.get_available_actions(state)

    #for every possible move, compare it to v, if greater store the next state and call minimax on it
    for action in actions:
        next_state = asp.transition(state, action)
        v = max(v, min_value(asp, next_state, depth + 1, cutoff_depth, stats))
    return v


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
        'states_expanded': 0  # Increase by 1 when get_available_actions is called on a state
    }
    

    start = asp.get_start_state()
    stats['states_expanded'] += 1
    actions = asp.get_available_actions(start)

    current_player = start.player_to_move()

    alpha = float('-inf')
    beta = float('inf')
 
    #initialize "first run" of min and maxval

    if current_player == 0:
        best_v = float('-inf')
        for action in actions:
            next_state = asp.transition(start, action)
            v = ab_min_value(asp, next_state, 1, cutoff_depth, alpha, beta, stats)
            if v > best_v:
                best_v = v
                best_action = action
            alpha = max(alpha, best_v)
    else:
        best_v = float('inf')
        for action in actions:
            next_state = asp.transition(start, action)
            v = ab_max_value(asp, next_state, 1, cutoff_depth, alpha, beta, stats)
            if v < best_v:
                best_v = v
                best_action = action
            beta = min(beta, best_v)


    return best_action, stats


def ab_min_value(asp : HeuristicAdversarialSearchProblem[GameState, Action], state: GameState, depth: int, cutoff_depth: float, alpha: float, beta: float, stats: Dict[str,int]) -> float:
    '''
    Docstring for ab_min_value
    Return the minvalue with alpha beta pruning for the given state
    :param asp: Description
    :type asp: HeuristicAdversarialSearchProblem[GameState, Action]
    :param state: Description
    :type state: GameState
    :param depth: Description
    :type depth: int
    :param cutoff_depth: Description
    :type cutoff_depth: float
    :param alpha: Description
    :type alpha: float
    :param beta: Description
    :type beta: float
    :param stats: Description
    :type stats: Dict[str, int]
    :return: Description
    :rtype: float
    '''

    #check if terminal state
    if asp.is_terminal_state(state):
        return asp.get_result(state)
    
    #when reaches max depth, apply heuristic function to evaluate states
    if depth >= cutoff_depth:
        return asp.heuristic(state)
    
    #we are expanding one state, want to increment from -inf
    stats['states_expanded'] += 1
    v = float('inf')
    actions = asp.get_available_actions(state)

    #for every possible move, compare it to v, if less store the next state and call minimax on it
    for action in actions:
        next_state = asp.transition(state, action)
        v = min(v, ab_max_value(asp, next_state, depth + 1, cutoff_depth, alpha, beta, stats))

        #prune if v is less than alpha
        if v <= alpha:
            return v
        
        #update beta
        beta = min(beta, v)
    return v

def ab_max_value(asp : HeuristicAdversarialSearchProblem[GameState, Action], state: GameState, depth: int, cutoff_depth: float, alpha: float, beta: float, stats: Dict[str,int]) -> float:
    '''
    Docstring for ab_max_value
    Return the maxvalue with alpha beta pruning for the given state
    :param asp: Description
    :type asp: HeuristicAdversarialSearchProblem[GameState, Action]
    :param state: Description
    :type state: GameState
    :param depth: Description
    :type depth: int
    :param cutoff_depth: Description
    :type cutoff_depth: float
    :param alpha: Description
    :type alpha: float
    :param beta: Description
    :type beta: float
    :param stats: Description
    :type stats: Dict[str, int]
    :return: Description
    :rtype: float
    '''

    #check if terminal state
    if asp.is_terminal_state(state):
        return asp.get_result(state)
    
    #when reaches max depth, apply heuristic function to evaluate states
    if depth >= cutoff_depth:
        return asp.heuristic(state)
    
    #we are expanding one state, want to increment from -inf
    stats['states_expanded'] += 1
    v = float('-inf')
    actions = asp.get_available_actions(state)

    #for every possible move, compare it to v, if less store the next state and call minimax on it
    for action in actions:
        next_state = asp.transition(state, action)
        v = max(v, ab_min_value(asp, next_state, depth + 1, cutoff_depth, alpha, beta, stats))

        #prune if v is greater than beta
        if v >= beta:
            return v
        
        #update alpha
        alpha = max(alpha, v)
    return v

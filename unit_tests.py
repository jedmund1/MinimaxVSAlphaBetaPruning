import unittest

from asps.game_dag import DAGState, GameDAG
from adversarial_search import alpha_beta, minimax


class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take.
    """

    def _check_result(self, result: object, dag: GameDAG) -> None:
        """
        Tests whether the result is one of the possible actions of the DAG.
        
        Args:
            result: The return value of an adversarial search algorithm (should be an action)
            dag: The GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potential_actions = dag.get_available_actions(start_state)
        self.assertIn(result, potential_actions, "Output should be an available action")

    def test_minimax(self) -> None:
        """
        Test minimax algorithm on a basic GameDAG.
        
        Creates a simple game tree and verifies that minimax returns
        a valid action from the available set.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = minimax(dag)
        self._check_result(result, dag)

    def test_alpha_beta(self) -> None:
        """
        Test alpha-beta pruning algorithm on a basic GameDAG.
        
        Creates a simple game tree and verifies that alpha-beta returns
        a valid action from the available set.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1., 4: -2., 5: -3., 6: -4.}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, _ = alpha_beta(dag)
        self._check_result(result, dag)


if __name__ == "__main__":
    unittest.main()
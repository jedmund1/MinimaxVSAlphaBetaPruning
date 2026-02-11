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

    def test_simple_choice(self):
        '''
        Docstring for test_simple_choice
        We want to test the decision making ability of our minimax function
        We do this in order to verify that max player chooses max values and min player
        chooses min values from direct children
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, X],
                  [_, _, _, _],
                  [_, _, _, _],
                  [_, _, _, _]]
        terminal_evaluations = {1: 3., 2: 7., 3: 5.}

        start_state_max = DAGState(0, 0)
        dag_max = GameDAG(matrix, start_state_max, terminal_evaluations)
        result_max, stats_max = minimax(dag_max)

        self._check_result(result_max, dag_max)
        self.assertEqual(result_max, 2)
        self.assertEqual(stats_max['states_expanded'], 1)

        start_state_min = DAGState(0, 1)
        dag_min = GameDAG(matrix, start_state_min, terminal_evaluations)
        result_min, stats_min = minimax(dag_min)

        self._check_result(result_min, dag_min)
        self.assertEqual(result_min, 1)
        self.assertEqual(stats_max['states_expanded'], 1)

    def test_split_tree_depth2(self):
        '''
        Docstring for test_test_split_tree_depth2
        We want to test the decision making ability of our minimax function for multiple different branches and a longer depth
        We do this in order to verify that max player chooses max values and min player
        chooses min values based on information about what the other player is likely to play, and maximizes
        value from there
        We should expand three states: node 0 (-> 1, 2), node 1(-> 3,4), node2(->5, 6)
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, _, _, _, _],
                  [_, _, _, X, X, _, _],
                  [_, _, _, _, _, X, X],
                  [_, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _]]
        terminal_evaluations = {3: 9., 4: 2., 5: 11., 6: 13}

        #when max chooses first, will choose option 2 with choice 1 to get to 11/13
        start_state_max = DAGState(0, 0)
        dag_max = GameDAG(matrix, start_state_max, terminal_evaluations)
        result_max, stats_max = minimax(dag_max)

        self._check_result(result_max, dag_max)
        self.assertEqual(result_max, 2)
        self.assertEqual(stats_max['states_expanded'], 3)

        #when min chooses first, will choose option 1 with choice 1 to get to 9/2
        start_state_min = DAGState(0, 1)
        dag_min = GameDAG(matrix, start_state_min, terminal_evaluations)
        result_min, stats_min = minimax(dag_min)

        self._check_result(result_min, dag_min)
        self.assertEqual(result_min, 1)
        self.assertEqual(stats_max['states_expanded'], 3)

    
    def test_pos_neg_zero(self):
        '''
        Docstring for test_pos_neg_zero
        We want to test the decision making ability of our minimax function with all 3 types of min and max values
        We do this in order to verify that mininmax handls positive and negative and 0 values the same
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, X],
                  [_, _, _, _],
                  [_, _, _, _],
                  [_, _, _, _]]
        terminal_evaluations_1 = {1: -5., 2: -2., 3: -8.}

        start_state_1 = DAGState(0, 0)
        dag_1 = GameDAG(matrix, start_state_1, terminal_evaluations_1)
        result_1, stats_1 = minimax(dag_1)

        self._check_result(result_1, dag_1)
        self.assertEqual(result_1, 2)

        terminal_evaluations_2 = {1: 0., 2: 0., 3: 0.}
        start_state_2 = DAGState(0, 1)
        dag_2 = GameDAG(matrix, start_state_2, terminal_evaluations_2)
        result_2, stats_2 = minimax(dag_2)

        self._check_result(result_2, dag_2)
        self.assertEqual(result_2, 1)
        
    def test_split_tree_depth_uneven(self):
        '''
        Docstring for test_test_split_tree_depth_uneven
        We want to test the decision making ability of our minimax function for multiple different branches depths
        We do this in order to verify that our min and max functions handle terminating cases at different levels correctly
        and are able to assess cost accurately in accordance with depth
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, X, _, _, _, _],
                  [_, _, _, _, X, _, _, _],
                  [_, _, _, _, _, X, _, _],
                  [_, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, X],
                  [_, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _]]
        terminal_evaluations = {3: 6., 5: 0., 7: 16.}

        start_state_max = DAGState(0, 0)
        dag_max = GameDAG(matrix, start_state_max, terminal_evaluations)
        result, stats = minimax(dag_max)

        self._check_result(result, dag_max)
        #choose path to value 16
        self.assertEqual(result, 1) 
        #expand state 0, 1, 2, 4
        self.assertEqual(stats['states_expanded'], 4)

    def test_branching(self):
        '''
        Docstring for test_branching
        We want to test the functionality of one really long branch
        We do this in order to verify that we still will choose the optimal solution with 10 choices
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, X, X, X, X, X, X, X],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _]]

        terminal_evaluations = {1: 1., 2: 2., 3: 3., 4: 4., 5: 5., 6: 6., 7: 7., 8: 8., 9: 9.}

        start_state_max = DAGState(0, 0)
        dag_max = GameDAG(matrix, start_state_max, terminal_evaluations)
        result_max, stats_max = minimax(dag_max)

        self._check_result(result_max, dag_max)
        self.assertEqual(result_max, 9)
        self.assertEqual(stats_max['states_expanded'], 1)

    def test_tie_behavior(self):
        '''
        Docstring for test_tie_behavior
        We want to test the functionality of if there are ties between branches
        We do this in order to verify that we will choose a random action
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, X, X, X, X, X, X, X, X],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _],
                  [_, _, _, _, _, _, _, _, _, _]]

        terminal_evaluations = {1: 1., 2: 1., 3: 1., 4: 1., 5: 1., 6: 1., 7: 1., 8: 1., 9: 1., 10: 1.}

        start_state_max = DAGState(0, 0)
        dag_max = GameDAG(matrix, start_state_max, terminal_evaluations)
        result_max, stats_max = minimax(dag_max)

        self._check_result(result_max, dag_max)
        self.assertIn(result_max, [1, 2, 3, 4, 5,6 ,7, 8, 9, 10])
        self.assertEqual(stats_max['states_expanded'], 1)

    def test_edge_single_action(self):
        '''
        Docstring for test_edge_single_action
        We want to test the decision making ability of our minimax function when given only one option
        We do this in order to verify that no matter what the game ends in the expected state
        as there is no variability in choice for each opponent
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X],
                  [_, _],]
        terminal_evaluations = {1: 1.}

        start_state = DAGState(0, 0)
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, stats = minimax(dag)

        self._check_result(result, dag)
        self.assertEqual(result, 1)
        self.assertEqual(stats['states_expanded'], 1)

    def test_edge_chain(self):
        '''
        Docstring for test_edge_chain
        We want to test the decision making ability of our minimax function when given only one option 
        for each state in a chain structure, similar to our first edge test but longer path
        We do this in order to verify that no matter what the game ends in the expected state
        as there is no variability in choice for each opponent
        :param self: Description
        '''
        X = True
        _ = False

        matrix = [[_, X, _, _],
                  [_, _, X, _],
                  [_, _, _, X],
                  [_, _, _, _],]
        terminal_evaluations = {3: 1.}

        start_state = DAGState(0, 0)
        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, stats = minimax(dag)

        self._check_result(result, dag)
        self.assertEqual(result, 1)
        self.assertEqual(stats['states_expanded'], 3)




if __name__ == "__main__":
    unittest.main()
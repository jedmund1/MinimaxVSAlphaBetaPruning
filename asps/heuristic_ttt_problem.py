from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from .ttt_problem import TTTState, TTTProblem

SPACE = " "
X = "X"  # Player 0 is X
O = "O"  # Player 1 is O
PLAYER_SYMBOLS = [X, O]


class HeuristicTTTProblem(TTTProblem, HeuristicAdversarialSearchProblem):


    def eval_line(self, line: list) -> float:
        '''
        Docstring for eval_line
        create a score value based exponentially on the idea that you can only win lines you
        solely have control over, and a line is more valuable the more pieces placed there
        :param self: Description
        :param line: Description
        :type line: list
        :return: Description
        :rtype: float
        '''
        x_count = line.count("X")
        o_count = line.count("O")

        if x_count > 0 and o_count > 0:
            return 0.0
        
        if x_count > 0:
            return 10 ** x_count
        
        if o_count > 0:
            return -(10 ** o_count)
        
        return 0.0
    
    def heuristic(self, state: TTTState) -> float:
        """
        TODO: Fill this out with your own heuristic function! You should make sure that this
        function works with boards of any size; if it only works for 3x3 boards, you won't be
        able to properly test ab-cutoff for larger board sizes!
        """

        n_rows = len(state.board)
        score = 0.0

        #rows are already constructed
        for r in range(0, n_rows):
            score += self.eval_line(state.board[r])

        #construct each column and run eval on int
        for c in range(0, n_rows):
            column = [state.board[r][c] for r in range(0, n_rows)]
            score += self.eval_line(column)

        #construct the diagonals
        diagonal_1 = [state.board[i][i] for i in range(0, n_rows)]
        score += self.eval_line(diagonal_1)

        diagonal_2 = [state.board[i][n_rows - 1 - i] for i in range(0, n_rows)]
        score += self.eval_line(diagonal_2)


        return score

    
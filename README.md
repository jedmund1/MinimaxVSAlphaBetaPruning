[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pG7zhQvi)


Task 4: Are you convinced by the experiments?

Yes, I am convinced but only after seeing Figure 3.  The first two experiments don't prove much because both algorithms are deterministic, and use the same heuristic, thus should have similar performance when searching to the same depth.  The fact that minimax won 4 more games initially is just due to randomness and tiebreaking (first move advantages as well).  What Electro Wizard fails to realize at first is that he's unable to prove his point when searching at equal depths.  Since both options are optimal and deterministic, the choose the same moves at the same depth, essentially meaning the algorithms just compete against them sleves, creating easily reproducable results.  Yet, when experiment 3 happens, E-Wiz's point comes to light.  This is because at depths 1 and 2, both algorithms perform similarly with minimal state expansions.  But at depth 3, minimax expands about 60,000 states while alpha-beta only needs 10,000 to both get a similar performance score around 380.  At depth 4, the disparity is even more dramatic, with minimax using 350,000 states to score 420, while alpha-beta does the same with only 25,000 states.  The idea of alpha-beta being "better" cannot be uncovered at the same depth, as it's value lies in computing efficiency.  With a fixed state limit, alpha-beta can search deeper, and produce better results.  In experiments 1 and 2 we saw how it was impossible to uncover this when efficiency was a nonfactor, but it came to light in experiment 3, proving that alpha-beta is indeed "better" as you get to search equally as deep for way cheaper, and more depth for the same cost, which means higher performance in the long run.  

Testing Explanation: 
Minimax Tests:
I created test cases for my minimax implementation to verify correctness across many different scenarios, including board size and seek depth.  I first created tests that checked basic functionality with a simple choice test that ensures the maximizer and minimizer choose the best values from direct children with depth 1.  Then I tested depth 2 trees where max and min had to anticipate each other's moves, verifying that their reasoning works correctly on multiple levels.  Then, I tested edge cases including all negative values, 0 values, equal values, uneven tree depth, and strict branch.  To verify depth limited search, I created a test comparing full search vs a depth 2 cutoff that proved depth limiting can improve efficiency but reduce optimiality.  Lastly I tested terminal start states to verify correct behavior with no moves available.

Alpha-Beta Pruning

I created test cases for alpha-beta pruning forcusing on correctness and efficiency for the algorithm.  I first, similarly to minimax verified simple choice tests on depth 1, then did depth 2 to ensure proper behavior.  Then I tested the pruning of the algorithm with a guaranteed pruning tree structure.  In this test the algorithm was expected to explore the left branch and prune the right's second child as the value of the first child was an improvement (3 to 5) to the left branch, and the second right branch was significantly more (20).  The algorithm performed as expected and reduced state expansion accordingly.  Then I created equivalnece tests for minimax and a-b pruning for varied depths, proving that both reach optimal solutions while a-b expands less states.  I also tested the same edges, including negatives, equal values, and single moves.  Lastly, I tested terminal states to verify that a-b pruning would handle the ending game state correctly.  

TTT Heuristic Implementation:
My heuristic evaluates board states from the first player (X) perspective and analzyes all potential winning conditions.  For each row, column, or diagonal, I evaluated it based on the following rules.  It scored 0 if there was both an X and O move.  It would score positive 2^(X count) if there were only X in that line.  It would score negative 2^(O count) if there were only O in that line.  Then I summed the scores to find the overall value for the game state, which was returned.  The exponential weighting was a strategy that encourages piece placement on lines already close to winning.  The implementation also was adapted to work on all sizes of board.  


Known Problems: N/A

Collaborators: N/A

AI use: N/A
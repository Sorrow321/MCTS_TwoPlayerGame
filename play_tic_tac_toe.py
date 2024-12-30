from MCTS import TwoPlayerMCTS
from tic_tac_toe import TicTacToe


def print_board(state):
    """
    Prints the Tic Tac Toe board in the terminal.
    state: 3x3 matrix with:
        0 -> empty
        1 -> X
        2 -> O
    """
    markers = {0: ".", 1: "X", 2: "O"}
    print("Current board:")
    for row in range(3):
        row_str = " ".join(markers[state[row][col]] for col in range(3))
        print("  " + row_str)
    print()


def play_tictactoe(human_player_idx=0, n_iterations=50, n_rollouts=5):
    """
    Plays a Tic Tac Toe game in the terminal:
      - human_player_idx: 0 for 'X', 1 for 'O'
      - n_iterations: MCTS tree iterations
      - n_rollouts: MCTS rollout simulations
    """
    game = TicTacToe()
    mcts = TwoPlayerMCTS(game=game, exploration_coef=1.0)  # Use your real MCTS class here
    state = game.get_initial_state()

    # Player 0 starts by definition
    current_player = 0

    while True:
        print_board(state)
        
        # Possible actions
        actions = game.get_possible_actions(current_player, state)

        # If no actions left, it's terminal (draw or finished)
        if not actions:
            print("No more moves available. It's a draw or game ended.")
            break

        if current_player == human_player_idx:
            # Human turn
            while True:
                user_input = input(
                    f"Player {human_player_idx} (enter row,col in [1..3]): "
                )
                coords = user_input.strip().split(",")
                if len(coords) != 2:
                    print("  Invalid input. Please enter two numbers separated by a comma.\n")
                    continue
                try:
                    i, j = map(int, coords)
                except ValueError:
                    print("  Invalid input. Must be two integers.\n")
                    continue

                # Check bounds
                if not (1 <= i <= 3 and 1 <= j <= 3):
                    print("  Coordinates must be between 1 and 3.\n")
                    continue

                # Convert (i,j) to the single action index
                action_idx = (i - 1) * 3 + (j - 1)
                if action_idx not in actions:
                    print("  This cell is either occupied or invalid.\n")
                    continue

                # Valid move
                chosen_action = action_idx
                break
        else:
            # Computer turn -> use MCTS
            chosen_action = mcts.get_best_action(
                state,
                current_player,
                n_tree_iterations=n_iterations,
                n_rollout_simulations=n_rollouts,
                dump_tree_to_file=True
            )
            print(f"Computer (player {current_player}) chose action: {chosen_action}")

        # Apply the chosen action
        new_state, reward, done = game.make_transition(current_player, chosen_action, state)
        state = new_state

        # Check if game ended
        if done:
            print_board(state)
            if reward == 1:
                print("Player 0 (X) won!")
            elif reward == -1:
                print("Player 1 (O) won!")
            else:
                print("It's a draw!")
            break

        # Switch player
        current_player = 1 - current_player


###############################################################################
# If you want to run it directly:
###############################################################################
if __name__ == "__main__":
    # Example usage:
    #   - human plays as player 0 (X)
    #   - 50 MCTS iterations
    #   - 5 rollout simulations each
    play_tictactoe(human_player_idx=0, n_iterations=500, n_rollouts=5)

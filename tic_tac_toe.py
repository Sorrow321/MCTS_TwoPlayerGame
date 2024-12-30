import copy
from abstract_game import TwoPlayerGame


class TicTacToe(TwoPlayerGame):
    """
    A Tic Tac Toe game implementation inheriting from TwoPlayerGame.

    Players are represented by indices:
    - Player 0: Represented by 1 on the board
    - Player 1: Represented by 2 on the board
    Empty cells are represented by 0.
    """

    def get_possible_actions(self, player_idx, state):
        """
        Returns a list of possible actions (cell indices) that are empty.

        Args:
            player_idx (int): The index of the current player (0 or 1).
            state (list of lists): The current game state.

        Returns:
            list of int: List of cell indices (0-8) that are empty.
        """
        possible_actions = []
        for row in range(3):
            for col in range(3):
                if state[row][col] == 0:
                    possible_actions.append(row * 3 + col)
        return possible_actions

    def get_initial_state(self):
        """
        Initializes and returns the initial game state.

        Returns:
            list of lists: A 3x3 grid initialized to 0.
        """
        return [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

    def make_transition(self, player_idx, action_idx, state):
        """
        Applies an action to the current state and returns the new state,
        the reward, and whether the game has ended.

        Args:
            player_idx (int): The index of the current player (0 or 1).
            action_idx (int): The cell index (0-8) where the player wants to place their marker.
            state (list of lists): The current game state.

        Returns:
            tuple: (new_state, reward, done)
                - new_state (list of lists): The updated game state after the action.
                - reward (int): 1 if Player 0 wins, -1 if Player 1 wins, 0 otherwise.
                - done (bool): True if the game has ended, False otherwise.

        Raises:
            ValueError: If the chosen action is invalid (cell already occupied or out of bounds).
        """
        row = action_idx // 3
        col = action_idx % 3

        # Validate action
        if not (0 <= action_idx <= 8):
            raise ValueError(f"Invalid action index: {action_idx}. Must be between 0 and 8.")
        if state[row][col] != 0:
            raise ValueError(f"Invalid action: Cell ({row}, {col}) is already occupied.")

        # Create a deep copy of the state to avoid mutating the original
        new_state = copy.deepcopy(state)
        new_state[row][col] = player_idx + 1  # Player 0 -> 1, Player 1 -> 2

        # Check for victory
        if self._check_victory(new_state, player_idx + 1):
            reward = 1 if player_idx == 0 else -1
            return (new_state, reward, True)

        # Check for draw
        if self._is_draw(new_state):
            return (new_state, 0, True)  # 0 represents a draw

        # Game continues
        return (new_state, 0, False)

    def _check_victory(self, state, player_marker):
        """
        Checks if the specified player has won the game.

        Args:
            state (list of lists): The current game state.
            player_marker (int): The marker of the player (1 or 2).

        Returns:
            bool: True if the player has won, False otherwise.
        """
        # Check rows
        for row in state:
            if all(cell == player_marker for cell in row):
                return True

        # Check columns
        for col in range(3):
            if all(state[row][col] == player_marker for row in range(3)):
                return True

        # Check diagonals
        if all(state[i][i] == player_marker for i in range(3)):
            return True
        if all(state[i][2 - i] == player_marker for i in range(3)):
            return True

        return False

    def _is_draw(self, state):
        """
        Checks if the game has ended in a draw.

        Args:
            state (list of lists): The current game state.

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return all(cell != 0 for row in state for cell in row)

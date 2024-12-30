import unittest
from tic_tac_toe import TicTacToe  # Adjust the import path as necessary


class TestTicTacToe(unittest.TestCase):
    def setUp(self):
        """Initialize a new game and the initial state before each test."""
        self.game = TicTacToe()
        self.initial_state = self.game.get_initial_state()

    def test_initial_state(self):
        """Test that the initial state is an empty 3x3 grid."""
        expected_state = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertEqual(self.initial_state, expected_state, "Initial state should be a 3x3 grid of zeros.")

    def test_possible_actions_initial(self):
        """Test that all cells are available at the start of the game."""
        actions = self.game.get_possible_actions(0, self.initial_state)
        expected_actions = list(range(9))  # Cells 0 through 8
        self.assertEqual(actions, expected_actions, "All cells should be available initially.")

    def test_make_transition_valid_move(self):
        """Test making a valid move updates the state correctly."""
        player_idx = 0  # Player 0
        action_idx = 0  # Top-left corner
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, self.initial_state)
        
        expected_state = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        self.assertEqual(new_state, expected_state, "The state should reflect the player's move.")
        self.assertEqual(reward, 0, "No reward should be given for a non-winning move.")
        self.assertFalse(done, "The game should not be done after a single move.")

    def test_make_transition_invalid_move_occupied_cell(self):
        """Test that making a move on an occupied cell raises a ValueError."""
        # First, make a valid move
        player_idx = 0
        action_idx = 0
        new_state, _, _ = self.game.make_transition(player_idx, action_idx, self.initial_state)
        
        # Attempt to make another move on the same cell
        with self.assertRaises(ValueError) as context:
            self.game.make_transition(1, action_idx, new_state)
        
        self.assertIn("already occupied", str(context.exception), "Should raise ValueError for occupied cell.")

    def test_make_transition_invalid_move_out_of_bounds(self):
        """Test that making a move with an out-of-bounds index raises a ValueError."""
        player_idx = 0
        invalid_action_idx = 9  # Invalid index
        with self.assertRaises(ValueError) as context:
            self.game.make_transition(player_idx, invalid_action_idx, self.initial_state)
        
        self.assertIn("Invalid action index", str(context.exception), "Should raise ValueError for invalid action index.")

    def test_victory_row(self):
        """Test that the game detects a victory when a player fills a row."""
        # Set up the state with two markers in the first row
        state = [
            [1, 1, 0],
            [2, 2, 0],
            [0, 0, 0]
        ]
        player_idx = 0  # Player 0
        action_idx = 2  # Completing the first row
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [1, 1, 1],
            [2, 2, 0],
            [0, 0, 0]
        ]
        self.assertEqual(new_state, expected_state, "The first row should be filled with Player 0's markers.")
        self.assertEqual(reward, 1, "Player 0 should receive a reward for winning.")
        self.assertTrue(done, "The game should end after a victory.")

    def test_victory_column(self):
        """Test that the game detects a victory when a player fills a column."""
        # Set up the state with two markers in the first column
        state = [
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 0]
        ]
        player_idx = 0  # Player 0
        action_idx = 6  # Completing the first column
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [1, 2, 0],
            [1, 2, 0],
            [1, 0, 0]
        ]
        self.assertEqual(new_state, expected_state, "The first column should be filled with Player 0's markers.")
        self.assertEqual(reward, 1, "Player 0 should receive a reward for winning.")
        self.assertTrue(done, "The game should end after a victory.")

    def test_victory_diagonal(self):
        """Test that the game detects a victory when a player fills a diagonal."""
        # Set up the state with two markers in the main diagonal
        state = [
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        player_idx = 0  # Player 0
        action_idx = 8  # Completing the main diagonal
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        self.assertEqual(new_state, expected_state, "The main diagonal should be filled with Player 0's markers.")
        self.assertEqual(reward, 1, "Player 0 should receive a reward for winning.")
        self.assertTrue(done, "The game should end after a victory.")

    def test_victory_anti_diagonal(self):
        """Test that the game detects a victory when a player fills the anti-diagonal."""
        # Set up the state with two markers in the anti-diagonal
        state = [
            [0, 2, 1],
            [0, 1, 0],
            [0, 0, 0]
        ]
        player_idx = 0  # Player 0
        action_idx = 6  # Completing the anti-diagonal
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [0, 2, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]
        self.assertEqual(new_state, expected_state, "The anti-diagonal should be filled with Player 0's markers.")
        self.assertEqual(reward, 1, "Player 0 should receive a reward for winning.")
        self.assertTrue(done, "The game should end after a victory.")

    def test_draw_condition(self):
        """Test that the game detects a draw when the board is full without any victories."""
        # Set up a full board with no winner
        state = [
            [1, 2, 1],
            [2, 2, 2],
            [2, 1, 0]
        ]
        player_idx = 0  # Player 0
        action_idx = 8  # Last empty cell
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [1, 2, 1],
            [2, 2, 2],
            [2, 1, 1]
        ]
        self.assertEqual(new_state, expected_state, "The board should be full after the last move.")
        self.assertEqual(reward, 0, "No player should receive a reward in a draw.")
        self.assertTrue(done, "The game should end in a draw.")

    def test_game_continues(self):
        """Test that the game continues when there are empty cells and no winner."""
        # Set up a partially filled board without any winning conditions
        state = [
            [1, 2, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        player_idx = 1  # Player 1
        action_idx = 5  # Place at (1, 2)
        
        new_state, reward, done = self.game.make_transition(player_idx, action_idx, state)
        
        expected_state = [
            [1, 2, 0],
            [0, 1, 2],
            [0, 0, 0]
        ]
        self.assertEqual(new_state, expected_state, "The state should reflect Player 1's move.")
        self.assertEqual(reward, 0, "No reward should be given for a non-winning move.")
        self.assertFalse(done, "The game should continue after a non-winning move.")

if __name__ == '__main__':
    unittest.main()

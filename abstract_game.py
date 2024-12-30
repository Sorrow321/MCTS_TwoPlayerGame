from abc import ABC, abstractmethod


class TwoPlayerGame(ABC):
    @abstractmethod
    def get_initial_state(self):
        """
            Returns:
                Tuple (state, player_idx) - current state of the game, ID of the player that moves next
        """
        pass

    @abstractmethod
    def get_possible_actions(self, player_idx, state):
        """
            Given the player to move and the state, returns the list of possible actions
            Params:
                player_idx - 0 or 1
                state - state of the game, depends on the implementation
            Returns:
                List[int] - list of possible action IDs
        """
        pass

    @abstractmethod
    def make_transition(self, player_idx, action_idx, state):
        """
            Samples the next state from transition function p(s' | s, a). 
            Params:
                player_idx - 0 or 1
                action_idx - action to do from the list of possible actions for this player
                state - the position of the game in which we make the move
            Returns:
                Tuple (next_state, reward, done) - the next state, reward, flag done telling you if the game is finished
        """
        pass

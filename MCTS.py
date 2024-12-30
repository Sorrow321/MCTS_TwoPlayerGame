from __future__ import annotations
import numpy as np
import warnings
from abstract_game import TwoPlayerGame
from dataclasses import dataclass, field
from typing import List, Any, Optional
from collections import deque
from graphviz import Digraph

def ascii_tictactoe_board(state):
    """
        Given a 3x3 state where:
        0 -> empty
        1 -> X
        2 -> O
        returns a multiline ASCII string like:
            X . O
            . X .
            . O .
    """
    markers = {0: ".", 1: "X", 2: "O"}
    rows = []
    for row in state:
        row_str = " ".join(markers[cell] for cell in row)
        rows.append(row_str)
    return "\n".join(rows)

def _build_label(node, node_id):
    """
    Builds a multiline label for a node:
      - Node ID
      - player_to_move
      - V = accumulated_value / visits
      - visits
      - ASCII board
    """
    if node.n_visits > 0:
        v_val = node.accumulated_value / node.n_visits
        v_str = f"{v_val:.2f}"
    else:
        v_str = "NA"

    board_str = ascii_tictactoe_board(node.state)
    label = (
        f"ID={node_id}\n"
        f"Player={node.player_to_move}\n"
        f"V={v_str}\n"
        f"Term={node.terminal}. TR={node.terminal_reward}\n"
        f"Visits={node.n_visits}\n"
        f"{board_str}"
    )
    return label

@dataclass(eq=False)
class MCTSNode:
    state: Any
    player_to_move: int
    accumulated_value: float = 0.0
    n_visits: int = 0
    actions: List[int] = field(default_factory=list)
    children_nodes: List[MCTSNode] = field(default_factory=list)
    parent: Optional[MCTSNode] = None
    terminal: bool = False
    terminal_reward: int = 0.0


class TwoPlayerMCTS:
    def __init__(self, game: TwoPlayerGame, exploration_coef: float = 1.0):
        if exploration_coef < 0:
            raise ValueError("exploration_coef must be greater than 0")
        if np.isclose(exploration_coef, 0):
            warnings.warn('exploration_coef is set to 0')
        self.game = game
        self.exploration_coef = exploration_coef

    def change_player_idx(self, player_idx):
        return 1 - player_idx

    def select_child_node_ucb(self, node):
        if node.n_visits <= 0:
            raise AssertionError('Parent node n_visits must be > 0.')
        action_scores = np.zeros(len(node.children_nodes), dtype=np.float32)
        for child_idx, child in enumerate(node.children_nodes):
            if child.n_visits != 0:
                state_action_value = child.accumulated_value / child.n_visits
                exploration_bonus = (
                    self.exploration_coef
                    * np.sqrt(np.log(node.n_visits) / child.n_visits)
                )
                action_scores[child_idx] = state_action_value - exploration_bonus
            else:
                # not visited yet => infinite priority for exploration
                action_scores[child_idx] = -np.inf

        best_indices = np.where(action_scores == action_scores.min())[0]
        child_node_idx = np.random.choice(best_indices)
        return child_node_idx

    def run(self, state, player_idx_to_move, n_tree_iterations, n_rollout_simulations=10):
        root = MCTSNode(state=state, player_to_move=player_idx_to_move)
        for _ in range(n_tree_iterations):
            # forward pass (select)
            node = root
            player_idx = player_idx_to_move
            while len(node.children_nodes) != 0:
                next_node_idx = self.select_child_node_ucb(node)
                node = node.children_nodes[next_node_idx]
                player_idx = self.change_player_idx(player_idx)

            # expansion
            if not node.terminal:
                children_nodes = []
                actions = []
                possible_actions = self.game.get_possible_actions(player_idx, node.state)
                for action_idx in possible_actions:
                    new_state, reward, done = self.game.make_transition(
                        player_idx, action_idx, node.state
                    )
                    children_nodes.append(MCTSNode(
                        state=new_state,
                        parent=node,
                        terminal=done,
                        terminal_reward=reward if done else 0.0,
                        player_to_move=self.change_player_idx(player_idx)
                    ))
                    actions.append(action_idx)
                node.children_nodes = children_nodes
                node.actions = actions
                node_to_rollout = np.random.choice(node.children_nodes)
            else:
                node_to_rollout = node

            # rollout
            if not node_to_rollout.terminal:
                rollout_reward = self.rollout(node_to_rollout, n_rollout_simulations)
            else:
                rollout_reward = node_to_rollout.terminal_reward

            # backprop
            node = node_to_rollout
            while node is not None:
                if node.player_to_move == 0:
                    node.accumulated_value += rollout_reward
                else:
                    node.accumulated_value -= rollout_reward
                node.n_visits += 1
                node = node.parent

        return root

    def rollout(self, node: MCTSNode, n_rollout_simulations: int):
        total_reward = 0
        for _ in range(n_rollout_simulations):
            pid = node.player_to_move
            game_finished = False
            state = node.state
            while not game_finished:
                possible_actions = self.game.get_possible_actions(pid, state)
                if not possible_actions:
                    raise AssertionError("Rollout: got no possible action in non-terminal state")
                random_action = np.random.choice(possible_actions)
                state, reward, done = self.game.make_transition(pid, random_action, state)
                pid = self.change_player_idx(pid)
                game_finished = done
            total_reward += reward
        avg_reward = total_reward / n_rollout_simulations
        return avg_reward

    def get_best_action(self, state, player_idx,
                        n_tree_iterations=10, n_rollout_simulations=5,
                        dump_tree_to_file=False):
        root = self.run(state, player_idx, n_tree_iterations, n_rollout_simulations)

        if dump_tree_to_file:
            self.visualize_mcts_tree(root)
            self.dump_tree_to_file('tree.txt', root)

        # pick best child
        scores = np.array([
            child.accumulated_value / child.n_visits
            for child in root.children_nodes
        ])
        best_idx = scores.argmin()
        return root.actions[best_idx]

    def dump_tree_to_file(self, file_path: str, root: MCTSNode):
        """
        Traverses the MCTS tree from root (BFS) and dumps each node’s info:
         - Node ID
         - player_to_move
         - accumulated_value
         - V = accumulated_value / visits
         - n_visits
         - children node IDs **with** the associated action IDs
        """
        node2id = {}
        queue = deque([root])
        node2id[root] = 0
        lines = []

        while queue:
            node = queue.popleft()
            nid = node2id[node]

            # Build child info:  child_id(action_id)
            child_info = []
            for i, child in enumerate(node.children_nodes):
                if child not in node2id:
                    node2id[child] = len(node2id)
                    queue.append(child)
                child_id = node2id[child]
                action_id = node.actions[i]  # the move that led from node -> child
                child_info.append(f"{child_id}(action={action_id})")

            # Prepare V for readability
            if node.n_visits > 0:
                v_value = f"{node.accumulated_value / node.n_visits:.2f}"
            else:
                v_value = "NA"

            line = (
                f"NodeID: {nid}, "
                f"PlayerToMove: {node.player_to_move}, "
                f"AccumValue: {node.accumulated_value:.2f}, "
                f"V: {v_value}, "
                f"Visits: {node.n_visits}, "
                f"Children: [{', '.join(child_info)}]"
            )
            lines.append(line)

        # Write lines to file
        with open(file_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def visualize_mcts_tree(self, root_node, filename="mcts_tree", view=False):
        """
        Visualizes the MCTS tree from root_node using graphviz,
        including an ASCII board of each node's 3x3 Tic-Tac-Toe state,
        but only up to 3 layers (depths 0, 1, 2).
        """
        dot = Digraph(comment="MCTS Tree with Board")

        node2id = {}
        # Each queue entry is a tuple: (node, depth)
        queue = deque([(root_node, 0)])
        node2id[root_node] = "0"

        # Create root node label
        dot.node("0", _build_label(root_node, "0"),
                shape="box", style="filled", color="lightblue")

        # BFS up to depth 2 (i.e., 3 layers total: root=depth0, children=1, grandchildren=2)
        while queue:
            current_node, depth = queue.popleft()
            current_id = node2id[current_node]

            # If we've reached depth 2, we do not enqueue children to avoid going deeper.
            if depth == 2:
                # We still show the node at depth 2, but skip its children.
                continue

            # Otherwise, we add the children (which will be depth+1)
            for i, child_node in enumerate(current_node.children_nodes):
                if child_node not in node2id:
                    child_id_str = str(len(node2id))
                    node2id[child_node] = child_id_str

                    dot.node(child_id_str,
                            _build_label(child_node, child_id_str),
                            shape="box")

                    # Enqueue child at the next depth
                    queue.append((child_node, depth + 1))

                # Create an edge with an action label
                action_label = "?"
                if i < len(current_node.actions):
                    action_label = f"action={current_node.actions[i]}"
                dot.edge(current_id, node2id[child_node], label=action_label)
        dot.render(filename, view=view, format="pdf")

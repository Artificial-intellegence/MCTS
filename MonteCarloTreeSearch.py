########################################
# CS63: Artificial Intelligence, Lab 4
# Spring 2022, Swarthmore College
########################################
# NOTE: this should be the only file you need to modify for lab 4
########################################

#NOTE: You will probably want to use these imports. Feel free to add more.
from math import log, sqrt
import random

class Node(object):
    """Node used in MCTS"""
    def __init__(self, state):
        self.state = state
        self.children = {} # maps moves to Nodes
        self.visits = 0    # number of times node was in select/expand path
        self.wins = 0      # number of wins for player +1
        self.losses = 0    # number of losses for player +1
        self.value = 0     # value (from player +1's perspective)
        self.untried_moves = list(state.availableMoves) # moves to try

    def updateValue(self, outcome):
        """
        Increments self.visits.
        Updates self.wins or self.losses based on the outcome, and then
        updates self.value.

        This function will be called during the backpropagation phase
        on each node along the path traversed in the selection and
        expansion phases.

        outcome: Who won the game.
                 +1 for a 1st player win
                 -1 for a 2nd player win
                  0 for a draw
        """
        if outcome == 1:
            self.wins += 1
        elif outcome == -1:
            self.losses += 1
        self.visits += 1
        self.value = 1 + (self.wins-self.losses)/self.visits

    def UCBWeight(self, UCB_const, parent_visits, parent_turn):
        """
        Weight from the UCB formula used by parent to select a child.

        This function calculates the weight for JUST THIS NODE. The
        selection phase, implemented by the MCTSPlayer, is responsible
        for looping through the parent Node's children and calling
        UCBWeight on each.

        UCB_const: the C in the UCB formula.
        parent_visits: the N in the UCB formula.
        parent_turn: Which player is making a decision at the parent node.
           If parent_turn is +1, the stored value is already from the
           right perspective. If parent_turn is -1, value needs to be
           converted to -1's perspective.
        returns the UCB weight calculated
        """
        if parent_turn == 1:
            return self.value + UCB_const*sqrt(log(parent_visits/self.visits))
        else:
            return (2-self.value) + UCB_const*sqrt(log(parent_visits/self.visits))


class MCTSPlayer(object):
    """Selects moves using Monte Carlo tree search."""
    def __init__(self, num_rollouts=1000, UCB_const=1.0):
        self.name = "MCTS"
        self.num_rollouts = int(num_rollouts)
        self.UCB_const = UCB_const
        self.nodes = {} # dictionary that maps states to their nodes

    def getMove(self, game_state):
       # Find or create node for game_state
       key = str(game_state)
       if key in self.nodes:
          curr_node = self.nodes[key]
       else:
          curr_node = Node(game_state)
          self.nodes[key] = curr_node
       # Perform Monte Carlo Tree Search from that node
       self.MCTS(curr_node)
       # Determine the best move from that node
       bestValue = -float("inf")
       bestMove = None
       for move, child_node in curr_node.children.items():
          if game_state.turn == 1:
             value = child_node.value
          else:
             value = 2 - child_node.value
          if value > bestValue:
             bestValue = value
             bestMove = move
       return bestMove

    def status(self, node):
        """
        This method is used solely for debugging purposes. Given a
        node in the MCTS tree, reports on the node's data (wins, losses,
        visits, values), as well as the data of all of its immediate
        children. Helps to verify that MCTS is working properly.
        """
        print("node wins %d, losses %d, visits %d, value %f" % (node.wins, node.losses, node.visits, node.value))
        for move, child in node.children.items():
            print("\n child wins %d, losses %d, visits %d, value %f" % (child.wins, child.losses, child.visits, child.value))

    def MCTS(self, current_node):
        """
        Plays out random games from the root node to a terminal state.
        Each rollout consists of four phases:
        1. Selection: Nodes are selected based on the max UCB weight.
                      Ends when a node is reached where not all children
                      have been expanded.
        2. Expansion: A new node is created for a random unexpanded child.
        3. Simulation: Uniform random moves are played until end of game.
        4. Backpropagation: Values and visits are updated for each node
                     on the path traversed during selection and expansion.
        Returns: None
        """
        print("Completed %d rollouts" % (self.num_rollouts))
        for i in range(self.num_rollouts):
            path = self.selection(current_node)
            selected_node = path[-1]
            if selected_node.state.isTerminal:
                outcome = selected_node.state.winner
            else:
                next_node = self.expansion(selected_node)
                path.append(next_node)
                outcome = self.simulation(next_node)
            self.backpropagation(path, outcome)
        self.status(current_node) #use for debugging

    def expansion(self, current_node):
        """
        Performs expansion.
        A new node is created for a random unexpanded child.
        """
        rand_index = random.randrange(len(current_node.untried_moves))
        rand_move = current_node.untried_moves[rand_index]
        #remove move you are about to make from list of untried moves
        current_node.untried_moves.remove(rand_move)
        new_state = current_node.state.makeMove(rand_move)
        new_node = Node(new_state)
        #update dictionary of children
        current_node.children[rand_move] = new_node
        #update dictionary of all nodes (ie. the tree thus far)
        self.nodes[str(new_state)] = new_node
        return new_node

    def simulation(self, current_node):
        """
        Performs simulation. Uniform random moves are played until end of game.
        """
        current_state = current_node.state
        while not current_state.isTerminal:
            rand_index = random.randrange(len(current_state.availableMoves))
            rand_move = current_state.availableMoves[rand_index]
            next_state = current_state.makeMove(rand_move)
            current_state = next_state
        #if game is over and it would have been player 2's turn next (ie. player 1 wins)
        return current_state.winner

    def backpropagation(self, path, outcome):
        """
        Values and visits are updated for each node on the path traversed
        during selection and expansion.
        """
        for node in path:
            node.updateValue(outcome)

    def selection(self, current_node):
        """
        Performs selection.
        Nodes are selected based on the max UCB weight.
        Ends when a node is reached where not all children
        have been expanded.
        """
        path = []
        path.append(current_node)
        while not current_node.untried_moves and not current_node.state.isTerminal:
            bestValue = -float("inf")
            bestChild = None
            for child_node in current_node.children.values():
                UCB = child_node.UCBWeight(self.UCB_const, current_node.visits, current_node.state.turn)
                if UCB > bestValue:
                    bestValue = UCB
                    bestChild = child_node
            path.append(bestChild)
            #update current node to chosen child node (based on UCB weight)
            current_node = bestChild
        return path

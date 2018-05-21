import numpy as np
from copy import deepcopy
from src.node import node
from src.network import network

class mcts:
    def __init__(self, game_state, network, parameters, train=True):
        self.network = network
        self.max_playouts = parameters['playouts']
        self.temperature = parameters['temperature']['exploration'] if train else parameters['temperature']['exploitation']
        self.c_PUCT = parameters['c_PUCT']
        self.root = node(game_state, 1, self.c_PUCT)
        self.train = train
        
    def getBestMove(self):
        """ Given the game state of the root, find the best move
            using the provided network and max playout number.
        """
        current_playouts = 0
        evalCache = {}
        while current_playouts <= self.max_playouts:
            selected_node = self.root
            while any(selected_node.children):
                children_QU = [child.Q + child.getU() for child in selected_node.children]
                selected_node = selected_node.children[np.argmax(children_QU)]
            
            state = selected_node.state.toArray()
            if selected_node.state in evalCache:
                val = evalCache[selected_node.state]
            else:
                val = self.network.getEvaluation(state)
                evalCache[selected_node.state] = val
            selected_node.backup(val)

            legal_moves = selected_node.state.getLegalMoves()
            if any(legal_moves):
                net_policy = self.network.getPolicy(state, explore=self.train)
                for legal_move in legal_moves:
                    current_game = deepcopy(selected_node.state)
                    current_game.move(legal_move)

                    prior_prob = net_policy[current_game.dim*legal_move[1] + legal_move[0]]
                    child = node(current_game, parent=selected_node, move=legal_move, prior=prior_prob, c_PUCT=self.c_PUCT)
                    selected_node.children.append(child)   

            current_playouts += 1

        child_N_sum = sum([child.N ** (1/self.temperature) for child in self.root.children])

        move_probs = np.zeros((self.root.state.dim ** 2))
        child_evals = np.zeros((self.root.state.dim ** 2))
        for child in self.root.children:
            move_probs[self.root.state.dim * child.move[1] + child.move[0]] = (child.N ** (1/self.temperature)) / child_N_sum
            child_evals[self.root.state.dim * child.move[1] + child.move[0]] = child.Q

        children_probs = [(child.N ** (1/self.temperature)) / child_N_sum for child in self.root.children]
        child = np.random.choice(self.root.children, 1, p=children_probs)[0]

        return child.move, move_probs, (child.Q, child_evals)
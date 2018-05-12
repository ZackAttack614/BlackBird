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
        selected_node = self.root
        current_playouts = 0
        
        while current_playouts < self.max_playouts:
            while any(selected_node.children):
                children_QU = [child.Q + child.U for child in selected_node.children]
                selected_node = selected_node.children[np.argmax(children_QU)]
            
            state = np.append(
                selected_node.state.board,
                np.array([[
                    [[selected_node.state.player] for i in range(selected_node.state.dim)]
                    for j in range(selected_node.state.dim)]]),
                axis=3)
            net_policy = self.network.getPolicy(state, self.train)

            for legal_move in selected_node.state.getLegalMoves():
                current_game = deepcopy(selected_node.state)
                current_game.move(legal_move)

                prior_prob = net_policy[current_game.dim*legal_move[1] + legal_move[0]]
                child = node(current_game, parent=selected_node, move=legal_move, prior=prior_prob, c_PUCT=self.c_PUCT)
                selected_node.children.append(child)            

            for child in selected_node.children:
                current_game = child.state
                state = np.append(current_game.board, np.array([[
                        [[current_game.player] for i in range(current_game.dim)] for j in range(current_game.dim)
                    ]]), axis=3)
                net_eval = self.network.getEvaluation(state)
                child.backup(net_eval)

            for child in selected_node.children:
                child.updateU()

            current_playouts += 1
            
        children_probs = [
            (child.N ** (1/self.temperature)) / (self.root.N ** (1/self.temperature))
            for child in self.root.children]
        return self.root.children[np.argmax(children_probs)].move

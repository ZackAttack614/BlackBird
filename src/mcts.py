import numpy as np
from copy import deepcopy
from src.node import node
from src.network import network

class mcts:
    def __init__(self, game_state, network, playouts=400, temperature=1, c_PUCT=0.85):
        self.network = network
        self.max_playouts = playouts
        self.temperature = temperature
        self.root = node(game_state, prior=1, c_PUCT=c_PUCT)
        self.current_playouts = 0
        self.c_PUCT = c_PUCT
        
    def getBestMove(self, noise=True):
        selected_node = self.root
        
        while self.current_playouts < self.max_playouts:
            while any(selected_node.children):
                children_QU = [child.Q + child.U for child in selected_node.children]
                selected_node = selected_node.children[np.argmax(children_QU)]
            
            state = np.append(
                selected_node.state.board,
                np.array([[
                    [[selected_node.state.player] for i in range(selected_node.state.dim)]
                    for j in range(selected_node.state.dim)]]),
                axis=3)
            net_policy = self.network.getPolicy(state, noise)

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

            self.current_playouts += 1
            
        children_probs = [
            (child.N ** (1/self.temperature)) / (self.root.N ** (1/self.temperature))
            for child in self.root.children]
        return self.root.children[np.argmax(children_probs)].move

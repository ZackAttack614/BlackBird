import numpy as np
import random

from src.network import network
from src.mcts import mcts

class blackbird:
    def __init__(self, game_framework, parameters):
        self.game_framework = game_framework
        self.parameters = parameters
        
        self.network = network(parameters['network'], load_old=True)
        self.positions = []
    
    def selfPlay(self, num_games=1, show_game=False):
        for game_num in range(num_games):
            game_states = []
            new_game = self.game_framework()
            while not new_game.isGameOver():
                treeSearch = mcts(new_game, self.network, self.parameters['mcts'])
                selected_move = treeSearch.getBestMove()
                new_game.move(selected_move)
                
                move_probs = np.zeros((new_game.dim, new_game.dim))
                move_probs[selected_move] = 1
                move_probs = move_probs.reshape(9)
                
                game_states.append({
                    'state':np.append(
                        new_game.board,
                        np.array([[
                            [[new_game.player] for i in range(new_game.dim)]
                            for j in range(new_game.dim)]]),
                        axis=3),
                    'move_probs':move_probs
                })
                
                if show_game:
                    new_game.dumpBoard()
                
            result = new_game.getResult()
            for state in game_states:
                player = state['state'][0,0,0,2]
                if player == result:
                    reward = 1
                elif player == -result:
                    reward = -1
                else:
                    reward = 0
                
                state['reward'] = np.array([reward])
            
            self.positions += game_states
            
    def train(self, learning_rate=0.01):
        for position in self.positions:
            self.network.train(position['state'], position['reward'], position['move_probs'], learning_rate)
            del position
            
    def testNewNetwork(self, num_trials=25, against_random=False):
        new_network_score = 0
        old_network = network(load_old=True)
        
        for trial in range(num_trials):
            new_game = self.game_framework()
            
            old_network_color = random.choice([1, -1])
            new_network_color = -old_network_color
            
            current_player = 1
            while True:
                if current_player == old_network_color:
                    if not against_random:
                        m = mcts(new_game, old_network, self.parameters['mcts'], train=False)
                        move = m.getBestMove()
                    else:
                        move = random.choice(new_game.getLegalMoves())
                    new_game.move(move)
                else:
                    m = mcts(new_game, self.network, self.parameters['mcts'], train=False)
                    move = m.getBestMove()
                    new_game.move(move)
                
                if new_game.isGameOver():
                    break
                    
                current_player *= -1
                
            result = new_game.getResult()
            new_network_score += 1 if result == new_network_color else (-1 if result == old_network_color else 0)
            
        if abs(new_network_score) == num_trials:
            self.network.loadModel()
            return 0
        
        if new_network_score > num_trials*0.05:
            self.network.saveModel()
        
        return new_network_score

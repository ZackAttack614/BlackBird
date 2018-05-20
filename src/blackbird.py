import numpy as np
import random
from copy import deepcopy

from src.network import network
from src.mcts import mcts
from src.logger import *

def format2DArray(a):
    s = ''
    for r in range(a.shape[0]):
        s += '['
        for c in range(a.shape[1]):
            s += '{:<4.1f}'.format(a[r,c])
        s += ']\n'

    return s


class blackbird(logger):
    def __init__(self, game_framework, parameters):
        super().__init__(parameters['logging'])
        self.game_framework = game_framework
        self.parameters = parameters
        
        self.network = network(parameters['network'], load_old=True, writer=True)
        self.game_id = 0
        self.logConfig(parameters)
        
        self.states = []
        self.move_probs = []
        self.rewards = []
     
        
    @canLog(log_file = 'selfPlay.log')
    def selfPlay(self, num_games=1, show_game=False):
        """ Use the current network to generate test games for training.
        """
        for game_num in range(num_games):
            self.log('-'*20)
            self.log('New Game: {}'.format(game_num))
            
            new_game = self.new_game()
            move_num = 0
            states = []
            while not new_game.isGameOver():
                move_num += 1
                tree_search = mcts(new_game, self.network, self.parameters['mcts'])
                selected_move, move_probs, (state_eval, child_evals) = tree_search.getBestMove()
                self.__logMove(self.game_id, move_num, new_game, selected_move, move_probs, True, state_eval, child_evals)
                
                states.append(new_game.toArray())

                self.move_probs.append(move_probs)
                
                new_game.move(selected_move)
                if show_game:
                    new_game.dumpBoard()

            states.append(new_game.toArray())
            self.move_probs.append(np.zeros(self.move_probs[-1].shape))

            result = new_game.getResult()
            for state in states:
                player = state[0,0,0,2]
                reward = player * result
                self.states.append(state)
                self.rewards.append(reward)
        return

    @canLog('training.log')
    def train(self, learning_rate=0.01):
        """ Use the games generated from selfPlay() to train the network.
        """
        batch_size = self.parameters['network']['training']['batch_size']
        num_batches = int(len(self.states) / batch_size)

        for i,s in enumerate(self.states):
            self.log('State')
            self.log(str(s[0,:,:,2]))
            self.log(str(s[0,:,:,0]-s[0,:,:,1]))
            self.log('Value: {}'.format(self.rewards[i]))
        for batch in range(num_batches):
            batch_indices = np.sort(np.random.choice(range(len(self.states)), batch_size, replace=False))[::-1]
            
            batch_states = np.vstack([self.states[ind] for ind in batch_indices])
            batch_move_probs = np.vstack([self.move_probs[ind] for ind in batch_indices])
            batch_rewards = np.array([self.rewards[ind] for ind in batch_indices])

            self.network.train(batch_states, batch_rewards, batch_move_probs, learning_rate=learning_rate)

            for index in batch_indices:
                self.states.pop(index)
                self.move_probs.pop(index)
                self.rewards.pop(index)
                
        self.states = []
        self.move_probs = []
        self.rewards = []

        return
        
    @canLog(log_file = 'testNewNetwork.log')
    def testNewNetwork(self, num_trials=25, against_random=False, against_simple=False):
        """ Test the trained network against an old version of the network
            or against a bot playing random moves.
        """
        new_network_score = 0
        if not against_random:
            old_network = network(self.parameters['network'], load_old=True)
        
        for trial in range(num_trials):
            self.game_id += 1
            if not against_random and not against_simple:
                self.log('-'*20)
                self.log('Playing trial {}'.format(trial))
            new_game = self.new_game()
            
            old_network_color = random.choice([1, -1])
            new_network_color = -old_network_color

            if against_simple:
                simple_parameters = deepcopy(self.parameters)
                simple_parameters['mcts']['playouts'] = 1
            
            current_player = 1
            move_num = 0
            while True:
                move_num += 1
                if current_player == old_network_color:
                    if against_random:
                        move = random.choice(new_game.getLegalMoves())
                    elif against_simple:
                        tree_search = mcts(new_game, self.network, simple_parameters['mcts'], train=False)
                        move, _, _ = tree_search.getBestMove()
                    else:
                        tree_search = mcts(new_game, old_network, self.parameters['mcts'], train=False)
                        move, _, _ = tree_search.getBestMove()
                    new_game.move(move)
                else:
                    tree_search = mcts(new_game, self.network, self.parameters['mcts'], train=False)
                    move, move_probs, (state_eval,child_evals) = tree_search.getBestMove()
                    self.__logMove(self.game_id, move_num, new_game, move, move_probs, True, state_eval, child_evals)
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

        if not against_random:
            old_network.sess.close()
            del old_network
        
        return new_network_score


    def __logMove(self, game_id, move_num, state, move, probabilities, isTraining, state_eval = None, child_evals = None):
        # Log to file
        self.log('\nState - Value: {}'.format(state_eval))
        self.log('{}'.format(str(state)))
        m = np.zeros((3,3))
        m[move] = 1.0
        self.log(format2DArray(probabilities.reshape((3,3)).transpose()))
        if child_evals is not None:
            self.log(format2DArray(child_evals.reshape((3,3)).transpose()))
        self.log(format2DArray(m))

        self.logDecision(move_num, game_id, str(state), list(move), list(probabilities), isTraining, state_eval, list(child_evals))

        return


    def new_game(self):
        self.game_id += 1
        return self.game_framework()

    

     

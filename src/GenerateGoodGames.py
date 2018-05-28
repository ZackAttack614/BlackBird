import pickle
from blackbird import BlackBird
import yaml
import numpy as np

class JustTakeThatOneFunction(BlackBird):
    def __init__(self, parameters):
        return super().__init__(**parameters)

    # Just copied from the default implementation of MCTS
    def GetPriors(self, state):
        """ Gets the array of prior search probabilities. 
            Default is just 1 for each possible move.
        """
        return np.array([1] * len(state.LegalActions()))

    def SampleValue(self, state, player):
        """Samples the value of the state for the specified player.
            Must return the value in [0, 1]
            Default is to randomly playout the game.
        """
        rolloutState = state
        winner = rolloutState.Winner()
        while winner is None:
            actions = np.where(rolloutState.LegalActions() == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self._applyAction(rolloutState, action)
            winner = rolloutState.Winner(action)
        return 0.5 if winner == 0 else int(player == winner)

if __name__ == '__main__':
    with open('parameters.yaml', 'r') as param_file:
        parameters = yaml.load(param_file)
    ai = JustTakeThatOneFunction(parameters)

    examples = ai.GenerateTrainingSamples(1)
    with open('goodGames.txt', 'bw') as goodGames:
        pickle.dump(examples, goodGames)

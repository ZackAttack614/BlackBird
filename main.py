import os
import sys
import yaml
import random
sys.path.insert(0, './src/')

from blackbird import BlackBird
from TicTacToe import BoardState

def TestRandom(blackbirdAI, numTests):
    wins = 0
    draws = 0
    losses = 0
    gameNum = 0

    while gameNum < numTests:
        blackbirdToMove = random.choice([True, False])
        blackbirdPlayer = 1 if blackbirdToMove else 2
        winner = None
        blackbirdAI.DropRoot()
        state = BoardState()
        
        while winner is None:
            if blackbirdToMove:
                (nextState, _, _) = blackbirdAI.FindMove(state)
                state = nextState
                blackbirdAI.MoveRoot([state])

            else:
                legalMoves = state.LegalActions()
                move = random.choice([
                    i for i in range(len(legalMoves)) if legalMoves[i] == 1
                    ])
                state.ApplyAction(move)
                blackbirdAI.MoveRoot([state])

            blackbirdToMove = not blackbirdToMove
            winner = state.Winner()

        gameNum += 1
        if winner == blackbirdPlayer:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses

def main():
    assert os.path.isfile('parameters.yaml'), 'Copy the parameters_template.yaml file into parameters.yaml to test runs.'
    with open('parameters.yaml') as param_file:
        parameters = yaml.load(param_file.read().strip())

    LogDir = parameters.get('logging').get('log_dir')
    if LogDir is not None and os.path.isdir(LogDir):
        for file in os.listdir(os.path.join(os.curdir, LogDir)):
            os.remove(os.path.join(os.curdir, LogDir, file))
            
    TrainingParameters = parameters.get('selfplay')
    BlackbirdInstance = BlackBird(saver=True, tfLog=True,
                                  loadOld=True, **parameters)

    for epoch in range(1, TrainingParameters.get('epochs') + 1):
        print('Starting epoch {0}...'.format(epoch))
        nGames = parameters.get('selfplay').get('training_games')
        examples = BlackbirdInstance.GenerateTrainingSamples(nGames)
        BlackbirdInstance.LearnFromExamples(examples)
        print('Finished training for this epoch!')

        (wins, draws, losses) = TestRandom(BlackbirdInstance, parameters.get('selfplay').get('random_tests'))
        print('Against a random player:')
        print('Wins = {0}'.format(wins))
        print('Draws = {0}'.format(draws))
        print('Losses = {0}'.format(losses))

if __name__ == '__main__':
    main()

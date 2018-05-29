import yaml
from time import time
import os
import json
import sys
sys.path.insert(0, './src/')

from blackbird import BlackBird

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
        nGames = parameters.get('selfplay').get('training_games')
        examples = BlackbirdInstance.GenerateTrainingSamples(nGames)
        BlackbirdInstance.LearnFromExamples(examples)

if __name__ == '__main__':
    main()

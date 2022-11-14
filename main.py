import tracemalloc
tracemalloc.start()
import os
import sys
import yaml
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
sys.path.insert(0, './src/')

# import Blackbird
import LazyBlackbird
from DragonChess import BoardState


def APITest():
    start = time.time()
    if not os.path.isfile('parameters.yaml'):
        raise IOError('Copy parameters_template.yaml into parameters.yaml')
    with open('parameters.yaml') as param_file:
        parameters = yaml.safe_load(param_file)

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    for _ in range(100):
        model = LazyBlackbird.Model(BoardState, parameters['name'], parameters.get(
            'mcts'), parameters.get('network'), parameters.get('tensorflow'))

        LazyBlackbird.GenerateTrainingSamples(model,
                                        1,
                                        parameters.get('mcts').get('temperature').get('exploration'))
        LazyBlackbird.TrainWithExamples(model, batchSize=parameters.get('network').get('training').get('batch_size'), learningRate=parameters.get('network').get('training').get('learning_rate'))

        print('Against a random player:')
        print(LazyBlackbird.TestRandom(model,
                                parameters.get('mcts').get(
                                    'temperature').get('exploitation'),
                                10))

        print('Against the last best player:')
        print(LazyBlackbird.TestPrevious(model,
                                    parameters.get('mcts').get(
                                        'temperature').get('exploitation'),
                                    10))

        print('Against a good player:')
        print(LazyBlackbird.TestGood(model,
                                parameters.get('mcts').get(
                                    'temperature').get('exploitation'),
                                10))

        print('\n')
        print(f'Finished in {time.time()-start} seconds.')
        print('\n')
        del model


if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))
    with tf.device('/GPU:0'):
        APITest()

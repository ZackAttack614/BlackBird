from dragonchess import GameState, BoardState
import time
import numpy as np
import pybind11
from FixedLazyMCTS import FixedLazyMCTS

# start = time.time()
# state = BoardState()
# for i in range(100000):
#     s = state.Copy()
#     s.ApplyAction(188)
# print(time.time()-start)
# exit()

import cProfile, pstats
    
profiler = cProfile.Profile()
profiler.enable()
params = {'maxDepth' : 7, 'explorationRate' : 0.05, 'playLimit' : 500}
player = FixedLazyMCTS(**params)

state = BoardState()
start = time.time()
while state.Winner() == -1:
    print(state)
    print('To move: {}'.format(state.Player))
    print('Number of Legal Moves: {}'.format(state.NumLegalActions()))
    state, v, p = player.FindMove(state, temp=player.ExplorationRate)
    print('Value: {}'.format(v))
    # print('Selection Probabilities: {}'.format(p))
    print('Child Values: {}'.format(player.Root.ChildWinRates()))
    print('Child Exploration Rates: {}'.format(player.Root.ChildPlays()))
    print(f'Time taken: {time.time()-start}')
    player.MoveRoot(state)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.dump_stats('profile.log')
print(state)
print(state.Winner())
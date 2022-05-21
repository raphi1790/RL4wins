import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, unique
from operator import itemgetter
from typing import Tuple, NamedTuple, Hashable, Optional

import gym
import numpy as np
from gym import error
from gym import spaces

from environments.connect_four_env import ConnectFourEnv, Player



class RandomPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='RandomPlayer', seed: Optional[Hashable] = None):
        super().__init__(env, name)
        self._seed = seed
        # For reproducibility of the random
        prev_state = random.getstate()
        random.seed(self._seed)
        self._state = random.getstate()
        random.setstate(prev_state)

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()
        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Next operations are needed for reproducibility of the RandomPlayer when inited with seed
        prev_state = random.getstate()
        random.setstate(self._state)
        action = random.choice(list(available_moves))
        self._state = random.getstate()
        random.setstate(prev_state)
        return action

    def reset(self, episode: int = 0, side: int = 1) -> None:
        # For reproducibility of the random
        random.seed(self._seed)
        self._state = random.getstate()

    def save_model(self, model_prefix: str = None):
        pass


class Negamax(Player):
    def __init__(self, env: 'ConnectFourEnv', name='NegamaxPlayer', max_depth=4):
        super().__init__(env, name)
        # self.__weights = [1, 8, 128, 99999]
        # self.__max_depth = max_depth
        # self.__evaluated = {}
    
    def get_next_action(self, state: np.array) -> int:
        current_player_sign = self.env.current_player
        action = self.find_best_move(current_player_sign,4)
        temp_board = self.env.board
        return action
    
    def find_best_move(self, curr_player_sign, depth):
        opponent_sign = curr_player_sign * (-1)

        # if depth == 0:
        for i in range(4):
            self.env._step(2)
            print("current_player", self.env.current_player)
            print("is_win_state", self.env.is_win_state())
            self.env._inverse_step(2)
            # self.env.change_player_sign()
        return 1

    def _undo_move():
        pass
    def reset(self, episode: int = 0, side: int = 1) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        pass

class HumanPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='HumanPlayer'):
        super().__init__(env, name)

    def get_next_action(self, state: np.ndarray) -> int:
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move! Maybe invoke at the wrong time?')
        print("Choose one of the following moves:",list(available_moves ))
        user_input = int(input())


        return user_input

    def reset(self, episode: int = 0, side: int = 1) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        pass


class SavedPlayer(Player):
    def __init__(self, env, name='SavedPlayer', model_prefix=None):
        super(SavedPlayer, self).__init__(env, name)

        if model_prefix is None:
            model_prefix = self.name

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model = load_model(f"{model_prefix}.h5")

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        q_values = self.model.predict(state)[0]
        vs = [(i, q_values[i]) for i in self.env.available_moves()]
        act = max(vs, key=itemgetter(1))
        return act[0]
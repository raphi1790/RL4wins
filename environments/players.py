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

from environments.connect_four_env import ConnectFourEnv, Player, ResultType



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


class NegamaxPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', name='NegamaxPlayer', max_depth=4):
        super().__init__(env, name)
        self._weights = [1, 8, 128, 99999]
        width = self.env.board_shape[1]
        height = self.env.board_shape[0]
        rows = [[(y, x) for x in range(width)] for y in range(height)]
        columns = [[(x, y) for x in range(height)] for y in range(width)]

        up = [[(x, y) for x in range(height) for y in range(width)
                if x + y == z] for z in range(3, 9)]
        down = [[(x, y) for x in range(height) for y in range(width)
                    if x - y == z] for z in range(-3, 3)]

        segments = rows + columns + up + down
        self._splitted_segments=[segments[x][i:i+4] for x in range(len(segments)) for i in range(len(segments[x])-3)]
        # self.__max_depth = max_depth
        # self.__evaluated = {}
    
    def get_next_action(self, state: np.array) -> int:
        # print("input board", self.env.board)
        current_player_sign = self.env.current_player
        action, score = self.find_best_move( current_player_sign,4)
        # print("action", action, "score", score)
        return action
    
    def find_best_move(self, current_player_sign, depth):
        opponent_player_sign = current_player_sign * (-1)
        best_score = float('-inf')
        best_move = None

        # print("step_result", self.env.StepResult.res_type)
        
        if depth == 0:
            score = self._evaluate_position(current_player_sign,opponent_player_sign)
            # print("temporary_board", )
            return None, score
        
        # available_moves=self.env.available_moves()
        available_moves = [i for i in range(self.env.board_shape[1]) if self.env.is_valid_action(i)]
        # print("available_moves",available_moves)
        for move in available_moves:
            step_result = self.env._step(move)     

            self.env.change_player_sign()
            _, best_subscore = self.find_best_move(opponent_player_sign, depth -1)
            best_subscore *= -1
            
            # if self.env.is_win_state():
            #     best_subscore = current_player_sign*9999999999
            # elif self.env.StepResult.res_type == 'DRAW':
            #     best_subscore = 0

            # else:
               
            self.env._inverse_step(move)

            

            if best_subscore > best_score:
                best_score = best_subscore
                best_move = move

        # Happens when max_depth exceeds number of possible moves
        if best_move is None:
            best_score = self._evaluate_position(current_player_sign, opponent_player_sign)

        # print("end_board", self.env.board)
        return best_move, best_score
        
    
    def _evaluate_position(self, curr, opp):
        """Counts and weighs longest connected checker chains, which can lead to win"""
        board = self.env.board.copy()
        
        curr_score = 0
        opp_score = 0

        for indexes in self._splitted_segments:
            curr_count = 0
            opp_count = 0

            for index in indexes:
                v = board[index[0]][index[1]]
                if v == curr:
                    curr_count += 1
                elif v == opp:
                    opp_count += 1

            if curr_count > 0 and opp_count > 0:
                continue
            elif curr_count > 0:
                curr_score += curr_count * self._weights[curr_count - 1]
            elif opp_count > 0:
                opp_score += opp_count * self._weights[opp_count - 1]

        difference = curr_score - opp_score
        return difference


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
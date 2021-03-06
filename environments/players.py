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
        # segments = rows
        self._splitted_segments=[segments[x][i:i+4] for x in range(len(segments)) for i in range(len(segments[x])-3)]
        self.__max_depth = max_depth
        # self.__evaluated = {}
    
    def get_next_action(self, state: np.array) -> int:
        current_player_sign = self.env.current_player
        action, best_score, evaluation_board = self.find_best_move( current_player_sign,4)
        if current_player_sign != self.env.current_player:
            self.env.change_player_sign()
        return action
    
    def find_best_move(self, current_player_sign, depth):
        opponent_player_sign = current_player_sign * (-1)
        best_score = float('-inf')
        best_board = np.zeros(self.env.board_shape, dtype=int)
        best_move = None

        # print("step_result", self.env.StepResult.res_type)
        if depth == 0 or self.env.is_done_state():
            if self.env.is_win_state():
                score = self._evaluate_position(current_player_sign,opponent_player_sign)
            elif self.env.is_draw_state():
                score = 0
            else:
                score = self._evaluate_position(current_player_sign,opponent_player_sign)
            evaluation_board = self.env.board
            
            # print(evaluation_board)
            return None, score, evaluation_board
        
        # available_moves=self.env.available_moves()
        available_moves = [i for i in range(self.env.board_shape[1]) if self.env.is_valid_action(i)]
        # print("available_moves",available_moves)
        for move in available_moves:
            _ = self.env._step(move)               
 
            self.env.change_player_sign()
            _, best_subscore, evaluation_board = self.find_best_move(opponent_player_sign, depth -1)

            best_subscore *= -1

            self.env._inverse_step(move)
            self.env.change_player_sign()


            if best_subscore > best_score:
                best_score = best_subscore
                best_move = move
                best_board = evaluation_board
    

        # Happens when max_depth exceeds number of possible moves
        if best_move is None:
            best_score = self._evaluate_position(current_player_sign, opponent_player_sign)
            best_board = self.env.board

        # print("end_board", self.env.board)
        return best_move, best_score, best_board
        
    
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
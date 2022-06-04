from random import Random
from environments.connect_four_env import ConnectFourEnv
from environments.players import  HumanPlayer, NegamaxPlayer, RandomPlayer
from random import random

def run_multiple_experiments():
    random_wins = 0
    negamax_wins=0
    num_experiments=50
    for i in range(num_experiments):
        print("round:",i+1)
        env=ConnectFourEnv()
        players=[RandomPlayer(env, 'Dexter-Bot'),NegamaxPlayer(env, 'Deedee-Bot')]
        first_player_index = 0 if random() <0.5 else 1
        second_player_index = 1-first_player_index
        first_player = players[first_player_index]
        second_player = players[second_player_index]

        print("first_player:", first_player, "second_player:",second_player)

        result, board = env.run(first_player, second_player, render=False)
        print(board)
        reward = result.value
        if (reward == 1 and first_player_index == 0) or (reward == -1 and second_player_index == 0):
            random_wins +=1
        if (reward == 1 and first_player_index == 1) or (reward == -1 and second_player_index == 1):
            negamax_wins += 1
            
        print("num_experiments:", num_experiments, "random_wins:", random_wins,"negamax_wins:", negamax_wins )

    print("num_experiments:", num_experiments, "random_wins:", random_wins,"negamax_wins:", negamax_wins )

def run_single_experiment():
    env=ConnectFourEnv()
    player1 = NegamaxPlayer(env, 'Deedee-Bot')
    player2 = HumanPlayer(env, 'Dexter-Bot' )
    result, board = env.run(player1, player2, render=True)
    reward = result.value
    print(reward)
    print(board)

if __name__ == "__main__":
    run_single_experiment()
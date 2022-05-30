from environments.connect_four_env import ConnectFourEnv
from environments.players import  HumanPlayer, NegamaxPlayer, RandomPlayer


if __name__ == "__main__":
    env=ConnectFourEnv()
    player1 = HumanPlayer(env, 'Dexter-Bot')
    player2 = NegamaxPlayer(env, 'Deedee-Bot')
    result = env.run(player1, player2, render=False)
    reward = result.value
    print(reward)
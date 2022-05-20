from environments.connect_four_env import ConnectFourEnv, HumanPlayer, Negamax, RandomPlayer


if __name__ == "__main__":
    env=ConnectFourEnv()
    player1 = RandomPlayer(env, 'Dexter-Bot')
    player2 = Negamax(env, 'Deedee-Bot')
    result = env.run(player1, player2, render=False)
    reward = result.value
    print(reward)
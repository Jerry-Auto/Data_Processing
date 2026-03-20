
""" conda install -c conda-forge libstdcxx-ng """
def agent_play(env,agent):
    observation, info = env.reset()
    done=False
    for _ in range(1000):
        while not done:
            action = agent.take_action(observation)
            next_state, reward, done, truncated, _= env.step(action)
            observation = next_state
        if done or truncated:
            observation, info = env.reset()
    env.close()
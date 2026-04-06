def main():
    print("Hello from craftium-with-td3!")

    import gymnasium as gym
    from craftium import CraftiumEnv

    env = gym.make("Craftium/ChopTree-v0")

    observation, info = env.reset()

    for t in range(100):
        action = env.action_space.sample()  # get a random action
        observation, reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()

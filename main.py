def main():
    print("Hello from craftium-with-td3!")

    import matplotlib.pyplot as plt
    import numpy as np
    import gymnasium as gym
    import craftium

    env = gym.make("Craftium/ChopTree-v0")

    observation, info = env.reset()

    for t in range(20):
        action = env.action_space.sample()

        # plot the observation
        plt.clf()
        plt.imshow(np.transpose(observation, (1, 0, 2)))
        plt.savefig(f"results/observation_{t}.png")  # saves observations in results folder

        observation, reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    print("Done!")


if __name__ == "__main__":
    main()

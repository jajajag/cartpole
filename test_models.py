import argparse
import copy
import gym
import numpy as np
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=8)
# Name of the vulnerable model
parser.add_argument("--vul_model", type=str, default="dqn_l")
parser.add_argument("--model", type=str, default="dqn_l")
parser.add_argument("--neg_reward", type=str, default="False")
args = parser.parse_args()
# Temperature constant
T = 1
ENV_NAME = "CartPole-v1"


# Calculate the c function that maximize the difference between e values
def get_c(model, state):
    q_values = model.predict(state.reshape(1, -1))[0]
    e_values = np.exp(q_values / T)
    s = np.sum(e_values)
    return np.max(e_values / s) - np.min(e_values / s)

# Return the best action of the vulnerable / attack model
def get_action(model, state, flag=True):
    q_values = model.predict(state.reshape(1, -1))[0]
    # Return the maximum if we want to use the vulnerable model,
    # or the attack model with negative reward
    if flag:
        return np.argmax(q_values)
    # Return minimum if we want to use the attack model with positive reward
    else:
        return np.argmin(q_values)

# Return the best attacking frame
def get_frame(env, model, state, num_frames, neg_reward):
    best_c, frame = 0, 0
    for i in range(num_frames):
        c = get_c(model, state)
        # If this c value is the best, save the frame
        if best_c < c:
            best_c = c
            frame = i
        action = get_action(model, state, not neg_reward)
        state, reward, terminal, info = env.step(action)
        # Break the loop if the game ends
        if terminal: break
    return frame

# Return the number of steps of the game
def play(env, vul_model, model, num_frames, neg_reward):
    step, terminal = 0, False
    state = env.reset()
    while not terminal:
        # Calculate the frame if this frame is the first
        if step % num_frames == 0:
            frame = get_frame(copy.deepcopy(env),
                              model, state, num_frames, neg_reward)
        # Check if it is the attack frame
        if step % num_frames == frame:
            action = get_action(model, state, neg_reward)
        else:
            action = get_action(vul_model, state)
        step += 1
        state, reward, terminal, info = env.step(action)
    return step


if __name__ == "__main__":
    # Load vulnerable model and attack model from path
    vul_model = load_model("models/" + args.vul_model + "/model")
    model = load_model("models/" + args.model + "/model")
    env = gym.make(ENV_NAME)
    env.seed(233)
    neg_reward = False
    if args.neg_reward.upper() == "TRUE": neg_reward = True
    steps = play(env, vul_model, model, args.num_frames, neg_reward)
    # Print out the result
    print("%s model with %d frames and neg_reward = %s ends in %d steps." %
          (args.model,
           args.num_frames,
           args.neg_reward,
           steps))


import matplotlib.pyplot as plt

# Plot of result_models
x = [2 ** i for i in range(1, 7)]
dqn_l = [29, 160, 171, 161, 281, 342]
dqn_s = [39, 143, 314, 283, 296, 343]
dqn_nl = [22, 149, 174, 313, 287, 359]
dqn_ns = [29, 500, 500, 383, 386, 446]
baseline = [357] * 6
plt.plot(x, dqn_l, x, dqn_s, x, dqn_nl, x, dqn_ns)
plt.plot(x, baseline, ls="--")
plt.xlabel("Attack every x frames")
plt.ylabel("Score when game ends")
plt.legend(["dqn_l", "dqn_s", "dqn_nl", "dqn_ns"])
plt.title("Overall performance of 4 different models")
plt.show()

# Plot of result_frame
x = [2 ** i for i in range(1, 7)]
dqn_l = [29, 160, 171, 161, 281, 342]
fixed_frame = [29, 438, 280, 493, 351, 500]
random_frame = [20, 182, 323, 329, 178, 400]
baseline = [357] * 6
plt.plot(x, dqn_l, x, fixed_frame, x, random_frame)
plt.plot(x, baseline, ls="--")
plt.xlabel("Attack every x frames")
plt.ylabel("Score when game ends")
plt.legend(["dqn_l", "fixed_frame", "random_frame"])
plt.title("Comparison of dqn_l vs. fixed / random frame")
plt.show()

# Plot of result_action
x = [2 ** i for i in range(1, 7)]
dqn_l = [29, 160, 171, 161, 281, 342]
dqn_s = [39, 143, 314, 283, 296, 343]
fixed_action = [29, 257, 426, 335, 315, 325]
random_action = [120, 135, 300, 345, 350, 343]
baseline = [357] * 6
plt.plot(x, dqn_l, x, dqn_s, x, fixed_action, x, random_action)
plt.plot(x, baseline, ls="--")
plt.xlabel("Attack every x frames")
plt.ylabel("Score when game ends")
plt.legend(["dqn_l", "dqn_s", "fixed_action", "random_action"])
plt.title("Comparison of dqn_l vs. fixed / random action")
plt.show()

# Plot of result_inverse
x = [2 ** i for i in range(1, 7)]
dqn_s = [29, 11, 9, 9, 9, 9]
dqn_l = [29, 154, 158, 167, 168, 170]
dqn_ns = [29, 155, 162, 168, 170, 170]
baseline = [170] * 6
plt.plot(x, dqn_s, x, dqn_l, x, dqn_ns)
plt.plot(x, baseline, ls="--")
plt.xlabel("Attack every x frames")
plt.ylabel("Score when game ends")
plt.legend(["dqn_s", "dqn_l", "dqn_ns"])
plt.title("Inverse performance on smaller DNN")
plt.show()


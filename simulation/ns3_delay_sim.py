import numpy as np
import matplotlib.pyplot as plt

rounds = 10
delay = []

for r in range(rounds):
    d = np.random.normal(loc=50, scale=10)
    delay.append(d)

plt.plot(delay, marker="o")
plt.title("UAV Communication Delay (NS-3 Style)")
plt.xlabel("Round")
plt.ylabel("Delay (ms)")
plt.grid()
plt.savefig("../results/latency_plot.png")
plt.show()

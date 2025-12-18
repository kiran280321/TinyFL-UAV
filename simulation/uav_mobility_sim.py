import numpy as np
import matplotlib.pyplot as plt

num = 6
area = 200

pos = np.random.rand(num,2)*area

plt.scatter(pos[:,0], pos[:,1], c="red")
for i,p in enumerate(pos):
    plt.text(p[0]+2,p[1]+2,f"UAV {i+1}")

plt.grid()
plt.title("UAV Swarm Mobility Simulation")
plt.savefig("../results/mobility_map.png")
plt.show()

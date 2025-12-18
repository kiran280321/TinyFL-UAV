import matplotlib.pyplot as plt

rounds = [1,2,3,4,5,6,7,8,9,10]
acc = [0.72,0.75,0.79,0.82,0.84,0.87,0.89,0.91,0.93,0.94]

plt.plot(rounds, acc, marker="o")
plt.title("Federated Learning Accuracy Convergence")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("../results/accuracy_plot.png")
plt.show()

import matplotlib.pyplot as plt

tpr = [1, 1, 1]
fpr = [1, 0.0603282, 0.00877532]
stages = [0, 1, 2]

plt.plot(stages, tpr, label='True Positive Rate')
plt.plot(stages, fpr, label='False Positive Rate')
plt.xlabel('Training Stage')
plt.ylabel('Rate')
plt.title('TPR and FPR across Training Stages')
plt.legend()
plt.show()

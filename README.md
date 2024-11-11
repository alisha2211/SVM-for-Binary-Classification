# SVM-for-Binary-Classification

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset
# Class 0: Points centered around (2, 2)
# Class 1: Points centered around (8, 8)
X = np.array([[1, 2], [2, 3], [3, 3], [8, 8], [9, 10], [10, 9]])
y = np.array([0, 0, 0, 1, 1, 1])  # Labels for the points

# Initialize SVM classifier with a linear kernel
svm = SVC(kernel='linear')
svm.fit(X, y)

# Visualize the result
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Plot decision boundary
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary for Binary Classification")
plt.show()

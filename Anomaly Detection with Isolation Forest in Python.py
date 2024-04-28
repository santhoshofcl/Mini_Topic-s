import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=0.5, size=(1000, 2))
anomaly_data = np.random.uniform(low=-5, high=5, size=(50, 2))

data = np.vstack([normal_data, anomaly_data])

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(data)

outliers = clf.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=outliers, cmap='viridis', edgecolors='k')
plt.colorbar(label='Outlier Score')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

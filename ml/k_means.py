import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

colors = ['black', 'green', 'yellow']
ports = [80, 443, 8080]
packet_size = [10000, 200000, 300000]

# Create data using above features
listing = []
for i in range(10):
    data = [ports[np.random.randint(0, 3)], packet_size[np.random.randint(0, 3)]]
    listing.append(data)

samples = np.array(listing)
kmeans = KMeans(n_clusters=3, random_state=0).fit(samples)
print(kmeans.labels_)

plt.title('My IP Graph')
plt.xlabel('Ports')
plt.ylabel('Packet Size')
labels = kmeans.labels_
for l in labels:
    for i in np.where(kmeans.labels_ == l)[:][0].tolist():
        item = samples[i]
        plt.plot([item[0]], [item[1]], 'o', color=colors[l])
plt.show()


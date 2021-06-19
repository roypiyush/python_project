import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


genes = ['gene' + str(i) for i in range(1, 101)]
wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]

data = pd.DataFrame(columns=[*wt, *ko], index=genes)
for gene in data.index:
    data.loc[gene, 'wt1': 'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1': 'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

print(data.head())
print(data.shape)

# Scale the sample data. Scaling method expects samples to be at rows rather than at columns which we did earlier
# while creating sample data. Here, we are also centering and scaling the data.
scaled_data = preprocessing.scale(data.T)

# Let's do PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

percentage_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
print("Percentage Variance ", percentage_variance)
labels = ['PC' + str(x) for x in range(1, len(percentage_variance) + 1)]

plt.bar(x=range(1, len(percentage_variance) + 1), height=percentage_variance, tick_label=labels)
for idx, v in enumerate(percentage_variance):
    plt.text(idx + 1, v + 1, str(v))
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
print("PCA data frame ", pca_df.head(10))

print("PC1 Data Frame ", pca_df.PC1.head(10))
print("PC2 Data Frame ", pca_df.PC2.head(10))
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(percentage_variance[0]))
plt.ylabel('PC2 - {0}%'.format(percentage_variance[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()


#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################
# get the name of the top 10 measurements (genes) that contribute
# most to pc1.
# first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes)
# now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values

# print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

data = ['Web-server-1', 'Web-server-2', 'App-server-1', 'App-server-2']
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(data).todense()
print(vectorizer.vocabulary_)

for i in range(len(data)):
    print('Calculating variance for {}'.format(data[i]))
    for f in features:
        distances = euclidean_distances(features[i], f)
        print(distances)

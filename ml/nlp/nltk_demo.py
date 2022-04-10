from nltk.book import text1
from nltk.book import sent1
from nltk.book import FreqDist


if __name__ == '__main__':
    fdist1 = FreqDist(text1)
    fdist1_keys = fdist1.keys()
    fdist1_values = fdist1.values()
    fdist1.plot(50, cumulative=True)
    text1.vocab()

    print(sent1)
from sklearn import tree

SPECIAL_CHARACTER = 'special character'
LETTER = 'letter'
NUMBER = 'number'


class NumberLetterPredictor(object):
    def __init__(self):
        self.classifier = tree.DecisionTreeClassifier()
        self._train()

    def _train(self):
        features = [[48], [57], [65], [90], [97], [122], [64], [91], [47], [58], [96], [123]]
        labels = [NUMBER, NUMBER, LETTER, LETTER, LETTER, LETTER, SPECIAL_CHARACTER, SPECIAL_CHARACTER,
                  SPECIAL_CHARACTER, SPECIAL_CHARACTER, SPECIAL_CHARACTER, SPECIAL_CHARACTER]
        self.classifier = self.classifier.fit(features, labels)

    def predict(self, data):
        data = ord(str(data))
        result = self.classifier.predict([[data]])
        return result[0]


def show_result(data):
    try:
        number_letter_predictor = NumberLetterPredictor()
        predict = number_letter_predictor.predict(data)
        print("%s is %s" % (str(data), predict))
    except KeyError as ke:
        print('Item not found', ke)

    except ValueError as ve:
        print("%s" % ve)


def main():
    show_result(7)
    show_result(3)
    show_result(4)
    show_result(9)
    show_result('$')
    show_result('&')
    show_result('l')
    show_result('L')
    show_result('K')
    show_result('k')
    show_result('R')


if __name__ == '__main__':
    main()

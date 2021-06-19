from sklearn import tree


def prediction(data, classifier_tree):
    try:
        predict = classifier_tree.predict(data)
        return predict
    except ValueError as ve:
        print("%s", ve)
        return ['']


def main():
    try:
        items = {1: 'apple', 0: 'orange'}
        # weight
        # texture = [Rough = 1, Smooth = 0]
        features = [[140, 1], [130, 1], [150, 0], [170, 0]]
        labels = [0, 0, 0, 1]
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(features, labels)

        predict_1 = prediction([[165, 0]], clf)
        predict_2 = prediction([[141, 0]], clf)

        print("Result: %s" % (items[predict_1[0]]))
        print("Result: %s" % (items[predict_2[0]]))

    except KeyError as ke:
        print('Item not found', ke)

    except ValueError as ve:
        print("%s" % ve)


if __name__ == '__main__':
    main()

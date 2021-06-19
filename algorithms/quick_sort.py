

def quick_sort(array, p, r):

    """
    Performs quick sort on given array from start position p 
    till end position r
    """

    if p < r:
        q = __partition(array, p, r)
        quick_sort(array, p, q - 1)  # Sort items to the left side of partition q
        quick_sort(array, q + 1, r)  # Sort items to the right side of partition q


def __partition(array, p, r):

    i = p - 1  # from 0 till i have elements which are less than pivot
    j = p
    while j < r:
        if array[j] < array[r]:
            i = i + 1
            array[i], array[j] = array[j], array[i]
        j = j + 1

    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1


def main():
    limit = 1000
    array = []
    for i in range(0, limit):
        import random
        array.append(random.randint(1, limit * 100))
    from datetime import datetime
    start = datetime.now()
    quick_sort(array, 0, len(array) - 1)
    done = datetime.now()
    if __name__ == '__main__':
        print array
        print "Time taken %d ms" % (int((done - start).microseconds) / 1000)


if __name__ == '__main__':
    main()
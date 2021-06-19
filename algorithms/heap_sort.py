
def left(i):
    return i * 2 + 1


def right(i):
    return i * 2 + 2


def heap_sort(array):
    __build_max_heap(array)
    heap_size = len(array)
    i = heap_size - 1
    while i > 0:
        array[i], array[0] = array[0], array[i]
        i = i - 1
        heap_size = heap_size - 1
        __max_heapify(array, 0, heap_size)


def __max_heapify(array, i, heap_size):

    largest = i

    if left(i) < heap_size and array[left(i)] > array[i]:
        largest = left(i)

    if right(i) < heap_size and array[right(i)] > array[largest]:
        largest = right(i)

    if largest != i:
        array[largest], array[i] = array[i], array[largest]
        __max_heapify(array, largest, heap_size)


def __build_max_heap(array):

    i = len(array) / 2
    while i >= 0:
        __max_heapify(array, i, heap_size=len(array))
        i = i - 1


def main():
    limit = 1000
    array = []
    for i in range(0, limit):
        import random
        array.append(random.randint(1, limit * 100))
    from datetime import datetime
    start = datetime.now()
    heap_sort(array)
    done = datetime.now()
    if __name__ == '__main__':
        print array
    print "Time taken %d ms" % (int((done - start).microseconds)/1000)


if __name__ == '__main__':
    main()

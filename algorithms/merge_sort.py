
def __merge(array, i, mid, j):

    l_arr = array[i:mid + 1]
    r_arr = array[mid + 1:j + 1]

    l = 0
    r = 0
    k = i
    while k <= j:
        if l < len(l_arr) and r < len(r_arr) and l_arr[l] <= r_arr[r]:
            array[k] = l_arr[l]
            l = l + 1
            k = k + 1
        elif l < len(l_arr) and r < len(r_arr) and r_arr[r] < l_arr[l]:
            array[k] = r_arr[r]
            r = r + 1
            k = k + 1
        elif l < len(l_arr):
            array[k] = l_arr[l]
            l = l + 1
            k = k + 1
        elif r < len(r_arr):
            array[k] = r_arr[r]
            r = r + 1
            k = k + 1


def merge_sort(array, i, j):
    if i < j:
        mid = (i + j) >> 1
        merge_sort(array, i, mid)
        merge_sort(array, mid + 1, j)
        __merge(array, i, mid, j)


def main():
    limit = 1000
    array = []
    for i in range(0, limit):
        import random
        array.append(random.randint(1, limit * 100))
    from datetime import datetime
    start = datetime.now()
    merge_sort(array, 0, len(array) - 1)
    done = datetime.now()
    if __name__ == '__main__':
        print array
    print "Time taken %d ms" % (int((done - start).microseconds)/1000)


if __name__ == '__main__':
    main()
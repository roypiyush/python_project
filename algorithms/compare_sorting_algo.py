import merge_sort
import quick_sort
import heap_sort

from datetime import datetime
import random


def main():

    print "Creating data samples"
    t_start = datetime.now()

    limit = 100000
    array = []
    for i in range(0, limit):
        array.append(random.randint(1, limit * 100))

    print "Starting algorithms"
    # Start of Quick Sort
    q_start = datetime.now()
    quick_sort.quick_sort(array, 0, len(array) - 1)
    q_done = datetime.now()
    # End of Quick Sort

    # Start of Merge Sort
    m_start = datetime.now()
    merge_sort.merge_sort(array, 0, len(array) - 1)
    m_done = datetime.now()
    # End of Merge Sort

    # Start of Heap Sort
    h_start = datetime.now()
    heap_sort.heap_sort(array)
    h_done = datetime.now()
    # End of Heap Sort

    t_done = datetime.now()

    print "Quick Sort Time taken %d ms" % ((q_done - q_start).microseconds / 1000)
    print "Merge Sort Time taken %d ms" % ((m_done - m_start).microseconds / 1000)
    print "Heap Sort Time taken %d ms" % ((h_done - h_start).microseconds / 1000)
    print "Total Time taken %d sec" % (t_done - t_start).seconds


if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        print "Caught Exception %s" % e.message

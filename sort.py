# -*- coding: utf-8 -*-


"""
几种常用的排序方法
"""


def bubble_sort(array):
    """冒泡排序

    步骤：

    1.比较相邻的元素。如果第一个比第二个大，就交换它们两个；
    2.对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
    3.针对所有的元素重复以上的步骤，除了最后一个；
    4.重复步骤1~3，直到排序完成。

    """
    lengths = len(array)
    for i in range(lengths-1):
        for j in range(lengths-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array


def select_sort(array):
    """选择排序
    n个记录的直接选择排序可经过n-1趟直接选择排序得到有序结果。具体算法描述如下：

     1.初始状态：无序区为R[1..n]，有序区为空；
     2.第i趟排序(i=1,2,3…n-1)开始时，当前有序区和无序区分别为R[1..i-1]和R(i..n）。该趟排序从当前无序区中-选出关键字最小的记录 R[k]，
     将它与无序区的第1个记录R交换，使R[1..i]和R[i+1..n)分别变为记录个数增加1个的新有序区和记录个数减少1个的新无序区；
     3.n-1趟结束，数组有序化了。

    """
    lengths = len(array)
    for i in range(lengths-1):
        min_index = i
        for j in range(i+1, lengths-1):
            if array[j] < array[min_index]:
                min_index = j
        array[i], array[min_index] = array[min_index], array[i]
    return array


def insert_sort(array):
    """插入排序

    插入排序（insertion sort）又称直接插入排序（staright insertion sort），其是将未排序元素一个个插入到已排序列表中。对于未排序元素，
            在已排序序列中从后向前扫描，找到相应位置把它插进去；在从后向前扫描过程中，需要反复把已排序元素逐步向后挪，为新元素提供插入空间。

    步骤

    1.从第一个元素开始，该元素可以认为已经被排序；
    2.取出下一个元素（未排序），在已经排序的元素序列中从后向前扫描；
    3.如果该元素（已排序）大于新元素，将该元素移到下一位置（往前移动）；
    4.重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
    5.将新元素插入到该位置后；
    6.重复步骤2~5。
    """
    lengths = len(array)
    for i in range(1, lengths):
        current_val = array[i]
        null_index = i
        for j in range(i-1)[::-1]:
            if array[j] > current_val:
                array[null_index] = array[j]
                null_index = j
            else:
                array[null_index] = current_val
                break
    return array


def quick_sort(array):
    """快速排序

    快速排序（quick sort）：通过一趟排序将待排列表分隔成独立的两部分，其中一部分的所有元素均比另一部分的所有元素小，
    则可分别对这两部分继续重复进行此操作，以达到整个序列有序。（这个过程，我们可以使用递归快速实现）

    步骤

    快速排序使用分治法来把一个串（list）分为两个子串（sub-lists）。具体算法描述如下：

    从数列中挑出一个元素，称为 “基准”（pivot），这里我们通常都会选择第一个元素作为prvot；
    重新排序数列，将比基准值小的所有元素放在基准前面，比基准值大的所有元素放在基准的后面（相同的数可以到任一边）。
    这样操作完成后，该基准就处于新数列的中间位置，即将数列分成了两部分。这个操作称为分区（partition）操作；
    递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数按上述操作进行排序。这里的递归结束的条件是序列的大小为0或1。
    此时递归结束，排序就完成了。

    """
    if len(array) <= 1:
        return array
    left = []
    right = []
    middle = array[0]
    for item in array[1:]:
        if item >= middle:
            right.append(item)
        else:
            left.append(item)
    return quick_sort(left)+[middle]+quick_sort(right)

def merge_sort(array):
    """
    归并排序：
    步骤：
    1. 将列表分割成相等两份，每份递归，直到每份长度>1
    2. 将分割后的小列表两两比较进行递归合并
    """
    # 设置递归结束条件
    if len(array) <= 1:
        return array
    # 递归分割数组
    mid = len(array) // 2
    left = array[:mid]
    right = array[mid:]

    # 递归合并
    merged = []
    # 比较左右两部分
    while left and right:
        if left[0] < right[0]:
            merged.append(left.pop(0))
        else:
            merged.append(right.pop(0))
    # 如果左右比较完之后还有剩余，直接添加到mergerd
    if left:
        merged.extend(left)
    if right:
        merged.extend(right)
    return merged

if __name__ == '__main__':
    array = [5, 4, 7, 1, 6, 2]
    print('冒泡排序：', bubble_sort(array))
    print('选择排序：', select_sort(array))
    print('插入排序：', insert_sort(array))
    print('快速排序：', quick_sort(array))
    print('归并排序：', merge_sort(array))
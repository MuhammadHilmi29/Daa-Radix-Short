def counting_sort(arr, min_val=None, max_val=None):
    if not arr:
        return arr

    if min_val is None:
        min_val = min(arr)
    if max_val is None:
        max_val = max(arr)

    key_range = max_val - min_val + 1
    count = [0] * key_range
    output = [0] * len(arr)

    for num in arr:
        count[num - min_val] += 1

    for i in range(1, key_range):
        count[i] += count[i - 1]

    for num in reversed(arr):
        idx = num - min_val
        output[count[idx] - 1] = num
        count[idx] -= 1

    for i in range(len(arr)):
        arr[i] = output[i]

    return arr
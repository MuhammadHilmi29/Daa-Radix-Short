def counting_sort(arr):
    if len(arr) <= 1:
        return list(arr)

    min_val = min(arr)
    max_val = max(arr)
    k = max_val - min_val + 1

    count = [0] * k
    for x in arr:
        count[x - min_val] += 1

    for i in range(1, k):
        count[i] += count[i - 1]

    output = [0] * len(arr)
    for x in reversed(arr):
        count[x - min_val] -= 1
        output[count[x - min_val]] = x

    return output
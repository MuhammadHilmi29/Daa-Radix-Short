def _counting_sort_by_digit(arr, exp, base):
    n = len(arr)
    output = [0] * n
    count = [0] * base

    if base == 256:
        for num in arr:
            digit = (num >> exp) & 0xFF
            count[digit] += 1

        for i in range(1, base):
            count[i] += count[i - 1]

        for num in reversed(arr):
            digit = (num >> exp) & 0xFF
            output[count[digit] - 1] = num
            count[digit] -= 1
    else:
        for num in arr:
            digit = (num // exp) % base
            count[digit] += 1

        for i in range(1, base):
            count[i] += count[i - 1]

        for num in reversed(arr):
            digit = (num // exp) % base
            output[count[digit] - 1] = num
            count[digit] -= 1

    for i in range(n):
        arr[i] = output[i]


def _radix_sort_non_negative(arr, base=10):
    if len(arr) <= 1:
        return arr

    max_val = max(arr)
    if base == 256:
        shift = 0
        while (max_val >> shift) > 0:
            _counting_sort_by_digit(arr, shift, 256)
            shift += 8
    else:
        exp = 1
        while max_val // exp > 0:
            _counting_sort_by_digit(arr, exp, base)
            exp *= base

    return arr


def radix_sort(arr, key_range=None):
    if len(arr) <= 1:
        return arr

    if key_range is None:
        key_range = max(arr) - min(arr) + 1

    base = 256 if key_range > 1_000_000 else 10

    negatives = [-x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    sorted_neg = [-x for x in reversed(_radix_sort_non_negative(negatives, base))]
    sorted_pos = _radix_sort_non_negative(positives, base)

    combined = sorted_neg + sorted_pos
    for i in range(len(arr)):
        arr[i] = combined[i]

    return arr
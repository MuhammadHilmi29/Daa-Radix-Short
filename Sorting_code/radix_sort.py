def _counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for x in arr:
        count[(x // exp) % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for x in reversed(arr):
        digit = (x // exp) % 10
        count[digit] -= 1
        output[count[digit]] = x

    return output


def _radix_sort_non_negative(arr):
    if not arr:
        return []

    max_val = max(arr)
    exp = 1

    while max_val // exp > 0:
        arr = _counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def radix_sort(arr):
    if len(arr) <= 1:
        return list(arr)

    negatives = [-x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    sorted_neg = _radix_sort_non_negative(negatives)
    sorted_pos = _radix_sort_non_negative(positives)

    # Negatif: urutan menaik nilai mutlak -> dibalik & dikembalikan tanda minus
    return [-x for x in reversed(sorted_neg)] + sorted_pos
import numpy as np

_ar = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 10, 20], [30, 40, 50], [60, 70, 80]]])
_ar = _ar[::-1, :, :]
print(_ar)

print(range(-5))
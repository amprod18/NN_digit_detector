import numpy as np

ar1 = np.eye(3)
ar2 = np.ones((3, 3))
ar3 = np.ones((3, 3))
dot = np.dot(np.dot(ar1, ar2), ar3)
print(dot, np.dot(ar2, ar3))
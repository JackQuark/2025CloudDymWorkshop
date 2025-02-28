import numpy as np

A = np.arange(1, 1001).reshape(10, 10, 10)
B = np.random.randint(0, 9, size=(10, 10))

print(A.shape)
print(B.shape)

C = A[B, np.arange(A.shape[1])[:, None], np.arange(A.shape[2])]
print(C.shape)

print(B)
print(C)
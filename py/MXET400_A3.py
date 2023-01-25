import numpy as np

Tdb = np.array([
    [0, 0, -1, 200],
    [0, -1, 0, -200],
    [-1, 0, 0, 50],
    [0, 0, 0, 1],
])

Tde = np.array([
    [0, 0, -1, 250],
    [0, -1, 0, 50],
    [-1, 0, 0, 100],
    [0, 0, 0, 1],
])

# Tad = np.array([
#     [0, 0, -1, 350],
#     [0, -1, 0, 50],
#     [-1, 0, 0, 250],
#     [0, 0, 0, 1],
# ])

Tbc = np.array([
    [-1/np.sqrt(2), -1/np.sqrt(2), 0, 50],
    [1/np.sqrt(2), -1/np.sqrt(2), 0, -50],
    [0, 0, 1, 20],
    [0, 0, 0, 1],
])

Tdc = np.matmul(Tdb, Tbc)
print(f"\nTdc:{Tdc}")

Tcd = np.linalg.inv(Tdc)
print(f"\nTcd:{Tcd}")

Tce = np.matmul(Tcd, Tde)
print(f"\nTce:{Tce}")


# print(Tdb)
# print('\n')
# print(Tbc)
# print('\n')
# print()
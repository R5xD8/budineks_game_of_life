import numpy as np

a = 0
b = 4001

for size in range(a, b):   
    if (size % 100 == 0) and (size != 0):
        arr = np.random.randint(0, 2, (size, size))
        np.savetxt(f"tests/data/input-board-data/boards/Array_{size}-x-{size}.txt", arr, fmt="%.1i")

print("done")
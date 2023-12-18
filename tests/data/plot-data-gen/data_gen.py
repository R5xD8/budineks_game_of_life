import numpy as np
import glob
import GOL_jit as JIT
import GOL_numpy as NUM
import time

paths = glob.glob("tests/data/input-board-data/boards/Array_*.txt")
print(*paths, sep="\n")

number_of_repetitions = 1000

names = {
    0 : "_JIT_HAL_",
    1 : "_JIT_MOD_",
    2 : "_JIT_PAR-HAL_",
    3 : "_JIT_PAR-MOD_",
    4 : "_NUM_HAL_",
    5 : "_NUM_MOD_"
}

print("starting...")

for i_path in range(len(paths)):
    path = paths[i_path]
    board = np.loadtxt(path, dtype=np.int8)

    simulators = np.array([
        JIT.GOL_with_JIT_hal(board),
        JIT.GOL_with_JIT_mod(board),
        JIT.GOL_with_JIT_par_hal(board),
        JIT.GOL_with_JIT_par_mod(board),
        NUM.GOL_just_numpy_hal(board),
        NUM.GOL_just_numpy_mod(board)
    ], dtype=object)

    for i_sim in range(len(simulators)):

        rep = 0
        dt_arr = np.zeros(number_of_repetitions)
        sum_arr = np.zeros(number_of_repetitions + 1)

        while rep < number_of_repetitions:

            if (i_sim == 0) or (i_sim == 2) or (i_sim == 4):
                cell_sum = np.sum((simulators[i_sim]).board[1:-1, 1:-1])
            else:
                cell_sum = np.sum((simulators[i_sim]).board)
            
            t1 = time.time_ns()
            
            simulators[i_sim].step()

            dt = time.time_ns() - t1

            if (dt >= 1000000000) and (i_sim > 0):
                rep = number_of_repetitions - 1
                print(f"{names[i_sim]} took too long (more than 60s/it), skipping the rest of the simulation...")

            dt_arr[rep] = dt

            sum_arr[rep] = cell_sum

            rep += 1
        
        if (i_sim == 0) or (i_sim == 2) or (i_sim == 4):
            cell_sum = np.sum((simulators[i_sim]).board[1:-1, 1:-1])
        else:
            cell_sum = np.sum((simulators[i_sim]).board)
        sum_arr[rep] = cell_sum

        np.savetxt(f"tests/plotting-data/sum/sum{names[i_sim]}{board.shape[0]}x{board.shape[1]}.txt", sum_arr)
        np.savetxt(f"tests/plotting-data/time/time-in-ns{names[i_sim]}{board.shape[0]}x{board.shape[1]}.txt", dt_arr)

        print(f"Just done Path {i_path}, simulator {names[i_sim]}.")

print("Done")
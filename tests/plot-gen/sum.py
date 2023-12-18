import numpy as np
import matplotlib.pyplot as plt
import glob

# get paths
paths = glob.glob("tests/data/plotting-data/sum/*.txt")

# create dictionary for names and index of data
names_dict = {
              "JIT_HAL": 0,
              "JIT_MOD": 1,
              "JIT_PAR-HAL": 2,
              "JIT_PAR-MOD": 3,
              "NUM_HAL": 4,
              "NUM_MOD": 5
              }

# create list of names
just_names = list(names_dict.keys())

# create array for data
data = np.zeros((len(names_dict),
                 len(paths)//len(names_dict),
                 len(np.loadtxt(paths[0]))))

# create array for x axis
x_size = np.arange(100, 4001, 100)

# load data
for i_path in range(len(paths)):
    pos = 0
    for i in range((len(paths)//6) + 1):
        some_pos = int(100*i)
        if ("x" + str(some_pos) + ".") in paths[i_path]:
            pos = int(i - 1)

    for name in names_dict.keys():
        if name in paths[i_path]:
            data[names_dict[name], pos] = np.loadtxt(paths[i_path], dtype=np.float64)

# find averages
data_av = np.average(data, axis=2)

# create plot
fig, ax = plt.subplots()
plt.xlabel("number of cells")
plt.ylabel("number of alive cells")
for i_data in range(len(data_av)):
    # for now I dont know how to filter out the NUM_HAL and NUM_MOD
    if just_names[i_data] == "NUM_HAL" or just_names[i_data] == "NUM_MOD":
        continue
    plt.plot(x_size**2, data_av[i_data], label = just_names[i_data])

# plt.scatter()
plt.legend()
plt.show()
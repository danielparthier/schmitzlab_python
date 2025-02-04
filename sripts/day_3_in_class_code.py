
# fixing errors
## syntax error
def extract_max(a)
    return max(a)

def solve_problem():
    b = ((1+1)*3))+(4/2)
    c = b * 2


# indentation error
def extract_max(a):
    if len(a) == 0:
        print("Empty list")
    else:
    return max(a)


## name error
a = [1,2,3]
def extract_max(a):
    return maxa(a)

def extract_double_max(a):
    import numpy as np
    c = np.array(a) * 2
    return max(c)

# type error
extract_double_max(2)

print("We can connect strings with numbers like" + 2)

# index error
a = [1,2,3]
print(a[3])

for i in range(10):
    print(a[i])

# attribute error
a = [1,2,3]
a.add(4)

a.split("_")

string_example = "Some_file.txt"
string_example.append("day_2")

# fix errors

import numpy as np
import os

def AP_check(folder):
    AP_sweep_count = 0
    for filename in os.listdir(folde):
        if filename.endsith('.csv'):
            with open(os.path.join(folder, filename), 'r') as file:
            data = np.loadtxt(file, skiprows=1)
                if any(data > 20):
                    AP_sweep_count += 1
                    print("AP found in " + filename)
                else:
                    print('No AP in ' + filename)
    return AP_sweep_count

file_path = 'data/sweeps_csv/'
sweep_count = AP_check(file_path)
print(sweep_count)

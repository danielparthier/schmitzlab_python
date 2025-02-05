# Solutions to the homework day 2


# local variable placement

RMPs = [-81, -76, -71, -78]
# make RMPs = [-80, -76, -71, -78] local

def rmp_change_local():
    # add code
    RMPs = [-80, -76, -71, -78]
    print(RMPs)

rmp_change_local()
print(RMPs)


# global variables
RMPs = [-81, -76, -71, -78]
def rmp_change_global():
    RMPs[0] = -80

rmp_change_global()
print(RMPs)

# local variables

RMPs = [-81, -76, -71, -78]
def rmp_change_local_modification(RMPs):
    RMPs_local = RMPs.copy()
    RMPs_local[0] = -80
    print(RMPs_local)

rmp_change_local_modification(RMPs)
print(RMPs)

RMPs = [-81, -76, -71, -78]
def rmp_change_local_modification():
    RMPs = [-80, -76, -71, -78]
    print(RMPs)

rmp_change_local_modification()
print(RMPs)

# if statement
RMP = -30
threshold = -34
if RMP > threshold:
    RMP = 10
print(RMP)


RMP = -30

def spike_simulation(RMP, threshold):
    if RMP == 10:
        RMP = -70    
    elif RMP > threshold:
        RMP = 10
    else:
        RMP = RMP
    return RMP

RMP = spike_simulation(RMP, threshold)

print(RMP)

RMP = spike_simulation(RMP, threshold)

print(RMP)

# for loop
input_list = [True, False, False, False, True, False, True, False, True, False] * 4
print(input_list)

number_of_inputs = 0
for input in input_list:
    if input:
        number_of_inputs += 1
        print("input")

print('The number of inputs is', number_of_inputs)

# for loop with random numbers

import numpy as np
np.random.seed(2025)
np.random.normal(5, 1)

# for loop with random numbers

np.random.seed(2025)

RMP = -70
for input in input_list:
    # add your code here
    if input:
        RMP += np.random.normal(5, 1)
    else:
        RMP += 0

print(np.round(RMP, 4))


# spike example
np.random.seed(2025)

threshold = -34
RMP = -70
RMP_list = []
spike_counter = 0

for input in input_list:
    # add your code here
    if input:
        RMP += np.random.normal(5, 1)
    else:
        RMP += 0
    RMP = spike_simulation(RMP, threshold)
    RMP_list.append(RMP)
    if RMP > 0:
        spike_counter += 1


print(spike_counter)

print(np.round(RMP_list[-1], 4))

# plot RMP_list
import matplotlib.pyplot as plt
plt.plot(RMP_list)
plt.show()

# spike location
spike_location = 0
for RMP in RMP_list:
    if RMP > 0:
        print(spike_location)
    spike_location += 1

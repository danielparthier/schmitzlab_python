# simple build-in functions

RMPs = [-70.1, -73.3, -69.8, -68.5, -71.2]
len(RMPs)

# simple functions - numpy

from fcntl import F_DUPFD
import numpy as np

zeros = np.zeros(12)  # one or more elements of the same type
ones = np.ones(12)
random = np.random.randint(12, None, 12) # low, high, size
ordered = np.arange(12)
A = np.stack((zeros, ones))
B = np.stack((random, ordered))

# attributes
print(A)
print(A.shape)
print(B)
type(A) # type of the variable
print(A.dtype) # type of the data inside the array

# more functions
A.flatten()
print(A + B)
print(B - A)

# find the mean
np.mean(A)
np.mean(B)
np.mean(A+B)
np.mean(B[1])

# other funcs
np.amax(B[1])
np.amin(B[0])

list_a = [2,3,4,5]
list_b = [1,3,5,7]



# define your own functions
# look for pieces that are changing
import json
import numpy as np

path = 'data/'
resting_membrane = [-70.1, -73.3, -69.8, -68.5, -71.2]

resting_membrane_avg = np.mean(resting_membrane)
sweeps = [1, 2, 3, 4, 5]
voltage_unit = 'mV'
sweep_count = len(resting_membrane)

output_file = path + 'data_info.json'


output_data = {
    'author': 'Doe, John',
    'date': '2025-01-10',
    'resting_membrane_avg': resting_membrane_avg,
    'unit': voltage_unit,
    'sweep_count': sweep_count
}

with open(output_file, 'w') as f:
    json.dump(output_data, f)

def create_meta_data_json(patcher, date_of_rec, RMPs, save_path, save_filename):
  avg_RMP = np.mean(RMPs)
  num_sweeps = len(RMPs)

  output_data = {
    'author': patcher,
    'date': date_of_rec,
    'resting_membrane_avg': avg_RMP,
    'unit': 'mV',
    'sweep_count': num_sweeps
  }

  print('saving the file ', save_filename, 'in', save_path)
  with open(save_path + save_filename, 'w') as f:
    json.dump(output_data, f)
  
  return output_data


# define own functions
def convert_miliseconds_to_minutes(ms):
  minutes = ms / 60_000
  return minutes

# default parameters in a function
def convert_time_to_ms(minutes, seconds = 0):
  ms = minutes * 60_000 + seconds * 1000
  return ms


# documentation
def convert_miliseconds_to_minutes(ms):
  '''
  this is the doc string
  '''
  minutes = ms / 60_000
  return minutes

help(convert_miliseconds_to_minutes)

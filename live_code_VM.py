## simple build-in functions

RMPs = [-70.1, -73.3, -69.8, -68.5, -71.2]
# find the length

## simple functions - numpy
import numpy as np

# zeros, ones, random, ordered
random = np.random.randint(12, None, 12) # low, high, size

# np.stack
#A = 
#B = 

# attributes: shape and dtype

# more functions: flatten, +, -

# mean, median, min, max

# index

## methods
# look functions up

animal_list = ['SNA 0254581', 'DSC 035576', 'SNA 0954581','SNA 0856662','DSC 024504']

# using the function sorted()
sorted_animal_list = sorted(animal_list)

# same thing but using a method .sort()
animal_list.sort() # sorting in ascending order
print(animal_list)

animal_list.sort(reverse = True) # sorting in descending order
print(animal_list)


num_animals = len(animal_list)
print("I've analyzed the data of", num_animals, "animals.")


# define your own functions
















# define own functions
def convert_miliseconds_to_minutes(ms):
  minutes = ms / 60_000
  return minutes

# default parameters in a function
def convert_time_to_ms(minutes, seconds = 0):
  ms = minutes * 60_000 + seconds * 1000
  return ms

# local and global variables

# starting script

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





# look for pieces that are changing

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




# documentation
def convert_miliseconds_to_minutes(ms):
  '''
  this is the doc string
  '''
  minutes = ms / 60_000
  return minutes

help(convert_miliseconds_to_minutes)

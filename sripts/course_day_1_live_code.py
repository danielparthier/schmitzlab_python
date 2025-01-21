# assign variables

cell_count, cell_density, cell_size = 3, 0.4, 4.1

## basic operators
recording_time = 93.5 # minutes
recording_time_seconds = recording_time * 60
recording_time_hours = recording_time / 60 # how to get only whole hours?

# how many seconds are left after 1 hour?
seconds_after_hour_1 = recording_time_seconds % (60*60)

# compare outcomes with different calculations (watch brackets)
seconds_after_hour_2 = ((recording_time_hours-1)*60*60)
seconds_after_hour_1 == seconds_after_hour_2

# working with strings
file_name = 'exp_data/sub-01/ses-02/func/sub-01_ses-02_task-run_data.csv'
file_name.split('/')

# get the last part of the file name and split it
file_split = file_name.split('/')[-1].split('_')
file_split[2].split('-')[1]

# working with lists
RMPs = [-70.1, -73.3, -69.8, -68.5, -71.2]
## get last element
RMPs[-1]

## which position is -69.8?
RMPs.index(-69.8)
## append a new value
RMPs.append(-72.1)
## sort
RMPs.sort()
## from lower to higher values
RMPs.sort(reverse=True)


# make dictionaries
author_list = ['Verjinia', 'Daniel']
new_dict = {
    "project": 'course',
    "date": '2025-01-21',
    "authors": author_list
}

## subset the dictionary and add more information
new_dict['authors']
new_information = {
    "duration": 120,
    "unit": 'minutes',
    "participants": 10
}
new_dict.update(new_information)

## get feedback for missing information
new_dict["location"]
new_dict.get('location', 'missing')
new_dict['location'] = 'Fenster der Wissenschaft'
new_dict.get('location', 'missing')



## simple build-in functions

RMPs = [-70.1, -73.3, -69.8, -68.5, -71.2]
RMPs.sort()
sorted()

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

convert_time_to_ms(minutes = 4, seconds = 10)
convert_time_to_ms(4,10)
convert_time_to_ms(5)

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


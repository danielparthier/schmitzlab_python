import json
import numpy as np

def create_meta_data_json(patcher, date_of_rec, RMPs, save_path, save_filename):
    '''
    Returns and saves a dictionary with metadata.

    Args:       patcher (str): name of experimenter
                date_of_rec (str): date of experiment
                RMPs (list): list of recorded resting membrane potentials
                save_path (str): destination folder
                save_filename (str): name of the file

    Returns:    output_data (dictionary): containing the metadata
    '''

    resting_membrane_avg = np.mean(RMPs)
    voltage_unit = 'mV'

    # assuming each RMPs values comes from a different sweep
    sweep_count = len(RMPs)

    # defining a dicctionary
    output_data = {
    'author': patcher,
    'date': date_of_rec,
    'resting_membrane_avg': resting_membrane_avg,
    'unit': voltage_unit,
    'sweep_count': sweep_count
    }

    output_file = save_path + save_filename
    print('Saving data into ', output_file)

    with open(output_file, 'w') as f:
        json.dump(output_data, f)

    return output_data







save_path = 'data/'
resting_membrane = [-70.1, -73.3, -69.8, -68.5, -71.2]
output_file = 'data_info.json'

out_data = create_meta_data_json('Verji', '25-01-21', [-70.1, -73.3, -69.8, -68.5, -71.2], 'data/', 'data_info.json')

out_data['resting_membrane_avg']

help(create_meta_data_json)
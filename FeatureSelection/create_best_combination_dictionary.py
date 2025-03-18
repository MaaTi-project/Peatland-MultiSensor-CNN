import os
import numpy as np


def create_best_combination_dictionary(path):
    best_dictionary = {}
    
    for type_of_inputs in os.listdir(path):
        for raster in os.listdir(os.path.join(path, type_of_inputs)):
            
            full_path = os.path.join(path, type_of_inputs, raster)
            
            if type_of_inputs == "Derived":
                if 'channel_3' not in best_dictionary.keys():
                    best_dictionary['channel_3'] = list()
                    
                best_dictionary['channel_3'].append(full_path)

            elif type_of_inputs == "Optical":
                if 'channel_1' not in best_dictionary.keys():
                    best_dictionary['channel_1'] = list()
                    
                best_dictionary['channel_1'].append(full_path)
            else:
                if 'channel_2' not in best_dictionary.keys():
                    best_dictionary['channel_2'] = list()
                best_dictionary['channel_2'].append(full_path)
    return best_dictionary
 


def load_the_dictionary_with_best_combination(path):
    best_combination = {}
    opener = open(path, 'r')
    
    for row in opener.readlines():
        splitted = row.strip().split(',')
        
        if splitted[1] not in best_combination.keys():
            best_combination[splitted[1]] = list() 
        
        best_combination[splitted[1]].append(splitted[0])

    return best_combination
    

if __name__ == '__main__':

    windows_size = 5
    no_data_value = 255
    no_data_sub = 0
    min_range = 0
    max_range = no_data_value - 1
    bands_needed = "all"
    type_of_data = "drained"
    
    PATH = os.path.join("..", "Windows", "TestArea" , type_of_data)
    best_combination = create_best_combination_dictionary(path=PATH)
    
    opener = open(os.path.join("..", "Configuration", "Masterfiles", "best_combination.csv"), 'w')

    for key in best_combination.keys():
        for elements in best_combination[key]:
            opener.write(f"{elements},{key},{windows_size},{no_data_value},{no_data_sub},{min_range},{max_range}:{bands_needed}\n")
    
    opener.close() 


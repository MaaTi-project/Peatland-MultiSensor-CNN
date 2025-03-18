import os
import numpy as np

def create_dataset_summary():
    handler = open(os.path.join("..", "Configuration", "Masterfiles", f"dataset_test_area.csv"), 'w')

    base_path = os.path.join("..", "Datasets")
    for area in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path, area)):
            full_path = os.path.join(base_path, area, file)
            
            if file == "Derived":
                row = f"{full_path},channel_3,255,0"
            elif file == "Optical":
                row = f"{full_path},channel_1,255,0"
            else:
                row = f"{full_path},channel_2,255,0"

            handler.write(f"{row}\n")
    
    handler.close()

if __name__ == '__main__':

    handler = open(os.path.join("..", "Configuration", "Masterfiles", f"dataset_test_area.csv"), 'w')


    base_path = os.path.join("..", "Datasets")
    for area in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path, area)):
            full_path = os.path.join(base_path, area, file)
            
            if file == "Derived":
                row = f"{full_path},channel_3,255,0,5"
            elif file == "Optical":
                row = f"{full_path},channel_1,255,0,5"
            else:
                row = f"{full_path},channel_2,255,0,5"

            handler.write(f"{row}\n")
    
    handler.close()
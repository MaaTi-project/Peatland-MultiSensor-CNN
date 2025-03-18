import multiprocessing

import sys 
import os 

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Functions.pixelwise_classification_utilities import *
from classification_configuration import (load_best_dictionary_from_input_selection_results, 
                                          pixelwise_classification_options, 
                                          define_the_target_raster, 
                                          check_the_bounds, 
                                          create_sub_raster_multiprocessing)
import argparse

def load_the_dictionary_with_best_combination(path):
    best_combination = {}
    opener = open(path, 'r')
    
    for row in opener.readlines():
        splitted = row.strip().split(',')
 
        if splitted[1] not in best_combination.keys():
            best_combination[splitted[1]] = list() 
        
        best_combination[splitted[1]].append(splitted[0])

    return best_combination


if __name__ == "__main__":
    try:
        
        if config["NOGPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            for i in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[i], True)

        # add arg parse is better than previous
        parser = argparse.ArgumentParser()

        parser.add_argument("-area",
                            help="The area to make the classification",
                            required=True)

        parser.add_argument("--opix",
                            help="Override the spatial pixel resolution",
                            required=False)

        parser.add_argument("--ocpu",
                            help="Override the number of cpu",
                            required=False)
        
        parser.add_argument("--path_to_raster",
                            help="Override the number of cpu",
                            required=False)

        # parse
        args = parser.parse_args()

        if args.area == "TESTAREA" or args.area is None:
            area = "TestArea"
        else:
            area = args.area


        if not os.path.isdir(args.path_to_raster):
            path_to_raster = os.path.join("..", "Datasets", area)
        else:
            path_to_raster = args.path_to_raster


        # we do not need the type because it is always the same geo area
        BEST_DICTIONARY_PATH = os.path.join("..", area, "Results", f"input_selection_second_stage_DRAINED", "summary_best.txt") 

        if not os.path.isfile(BEST_DICTIONARY_PATH):
            BEST_DICTIONARY_PATH = os.path.join("..", area, "Results", f"input_selection_second_stage_UNDRAINED", "summary_best.txt") 

        best_combination = load_best_dictionary_from_input_selection_results(input_selection_results_path=BEST_DICTIONARY_PATH)


        # rebuilt the command
        launch_cmd = f"launch_commander_{args.area.lower()}.txt"

        # get the pix (override)
        if args.opix and args.opix is not False:
            pixelwise_classification_options["OVERRIDE_RASTER_SIZE"] = int(args.opix)

        # get the number of cpu (override)
        if args.ocpu and args.ocpu is not False:
            number_of_cpus = int(args.ocpu)
        else:
            number_of_cpus = multiprocessing.cpu_count()

        print(f'[INFO] CPU COUNT {number_of_cpus}')

        # here I have to get the target raster
        target_raster = define_the_target_raster(dictionary_of_combination=best_combination,
                                                 dataset_path=path_to_raster)

        # check each bound if it is divisible by pix
        target_raster = check_the_bounds(target_raster=target_raster)

        
        # create the actual cmds
        list_to_compute = create_sub_raster_multiprocessing(target_raster=target_raster,
                                                            number_of_worker=number_of_cpus,
                                                            write_parser=True,
                                                            launch_cmd=launch_cmd,
                                                            area=area)
       
        print(f'[TARGET RASTER] Raster after checking the bound {target_raster}')
        
    except Exception as ex:
        print(f"Pixelwise classification v3 throws exception {ex}")

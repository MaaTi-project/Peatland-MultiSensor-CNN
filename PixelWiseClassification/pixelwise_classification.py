import argparse
import gc
import psutil
import numpy as np
import rasterio
import tensorflow as tf

import os
import sys 

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import traceback
from joblib import Parallel, delayed
import time
import pickle
import shutil
from classification_configuration import *



def pixelwise_classification_prediction(config,
                                        spatial_pixel_resolution,
                                        undrained_best_combination,
                                        drained_best_combination,
                                        current_east,
                                        current_north,
                                        worker,
                                        from_which_row,
                                        from_which_col,
                                        height_size_in_pixel,
                                        width_size_in_pixel,
                                        undrained_weights_folder,
                                        drained_weights_folder,
                                        class_id_list,
                                        no_data_raster,
                                        nodata_value,
                                        window_no_data_dictionary):
    try:
        
        
        # load the undrained model
        undrained_model = tf.keras.models.load_model(undrained_weights_folder)
        undrained_model.compile(loss="categorical_crossentropy", optimizer="adam")

        # load the drained model
        drained_model = tf.keras.models.load_model(drained_weights_folder)
        drained_model.compile(loss="categorical_crossentropy", optimizer="adam")
       
        # create two
        undrained_predicted_sector = np.full((height_size_in_pixel, width_size_in_pixel), nodata_value, dtype='uint8')
        drained_predicted_sector = np.full((height_size_in_pixel, width_size_in_pixel), nodata_value, dtype='uint8')
        
        # we need the specific colors
        # class_id_list = list(colors.keys())

        # load the no data tif
        #if no_data_raster is not None:
        loaded_tif_no_data = rasterio.open(no_data_raster)

        # load the undrained dataset
        undrained_stored_input = routine_load_the_dataset(data_to_load=undrained_best_combination, 
                                                          dataset_path=config['dataset_path'])

        # load the drained dataset
        drained_stored_input = routine_load_the_dataset(data_to_load=drained_best_combination, 
                                                        dataset_path=config['dataset_path'])

        print(f'[MEMORY worker {worker}] RAM Used (GB): {psutil.virtual_memory()[3] / 1000000000}')
        
        for R in range(from_which_row, height_size_in_pixel):
            start_timer = time.time()
            print(f"[Walking worker -> block_R {R}] ROW {R} over {height_size_in_pixel}")
            # slide for each pixel in col
            for C in range(from_which_col, width_size_in_pixel):

                # here let s decide if we have to put no value or not:
                point_E_to_analyse = current_east + (C * spatial_pixel_resolution)
                point_N_to_analyse = current_north - (R * spatial_pixel_resolution)


                # get if it is an undrained or drained point
                is_a_point_to_analyse = get_if_there_is_no_data(current_E=point_E_to_analyse,
                                                                current_N=point_N_to_analyse,
                                                                loaded_novalue_tif=loaded_tif_no_data)
                # do the undrained sector
                if is_a_point_to_analyse == 2:
                    data_to_predict = routine_prepare_data_for_prediction(configuration=configuration,
                                                                          stored_input=undrained_stored_input,
                                                                          point_N_to_analyse=point_N_to_analyse,
                                                                          point_E_to_analyse=point_E_to_analyse,
                                                                          spatial_pixel_resolution=spatial_pixel_resolution,
                                                                          window_no_data_dictionary=window_no_data_dictionary)

                    prediction = undrained_model.predict(data_to_predict, verbose=config["VERBOSE"])
                    prediction = np.squeeze(prediction)
                    prediction = np.argmax(prediction, axis=-1)

                    if config["STDOUT"]:
                        # if class_id_list[prediction] != 91:
                        print(f"[PREDICTION worker {worker}] Row -> {R} Col -> {C} is type {is_a_point_to_analyse}, the value is {class_id_list[prediction]}")

                    # save each pixel in the placeholder
                    undrained_predicted_sector[R, C] = class_id_list[prediction]
                    # for the other one put no data
                    # drained_predicted_sector[R, C] = nodata_value

                # do the drained sector
                elif is_a_point_to_analyse == 1:
                    data_to_predict = routine_prepare_data_for_prediction(configuration=configuration,
                                                                          stored_input=drained_stored_input,
                                                                          point_N_to_analyse=point_N_to_analyse,
                                                                          point_E_to_analyse=point_E_to_analyse,
                                                                          spatial_pixel_resolution=spatial_pixel_resolution,
                                                                          window_no_data_dictionary=window_no_data_dictionary)

                    prediction = drained_model.predict(data_to_predict, verbose=config["VERBOSE"])
                    prediction = np.squeeze(prediction)
                    prediction = np.argmax(prediction, axis=-1)

                    if config["STDOUT"]:
                        # if class_id_list[prediction] != 91:
                        print(
                            f"[PREDICTION worker {worker}] Row -> {R} Col -> {C} is type {is_a_point_to_analyse}, the value is {class_id_list[prediction]}")

                    # save each pixel in the placeholder
                    drained_predicted_sector[R, C] = class_id_list[prediction]
                    # same as before
                    # undrained_predicted_sector[R, C] = nodata_value
                else:
                    if config["STDOUT"]:
                        # insert no-data value
                        print(f"[NO DATA] No data value for C {C} and R {R} is type is {is_a_point_to_analyse}")
                    # undrained_predicted_sector[R, C] = nodata_value
                    # drained_predicted_sector[R, C] = nodata_value
                    continue

            print(f"[TIMER worker {worker}] Finished a row in {time.time() - start_timer} s")

        # remove stored input from garbage
        print(f"[MEMORY] I deleted stored input and data to predict in garbage collection")

        # close all opened raster
        routine_clean_the_raster(stored_input=undrained_stored_input)
        routine_clean_the_raster(stored_input=drained_stored_input)

        # close opened rasters
        loaded_tif_no_data.close()
        # unload the stuff from memory
        del loaded_tif_no_data
        gc.collect()
        
        return [undrained_predicted_sector, drained_predicted_sector]

    except Exception as ex:
        print(f"Make pixelwise classification throws exception {ex}")
        print(traceback.format_exc())

configuration = {
    "VERBOSE": 0,
    "STDOUT": True,
    "NOGPU": True,
    "DUMP": True,
    "channel_1": 5,
    "channel_2": 5,
    "channel_3": 5,
    "OFFSET": 0.0,
    "number_of_cpus": 20,
    "no_data_value": 255,
    "channel_list": ["channel_1", "channel_2", "channel_3"],
    "drained_weights": os.path.join("..", "TestArea", "Results", "BestCombination", "drained", "weights", "best_model_drained.keras"),
    "undrained_weights": os.path.join("..", "TestArea", "Results", "BestCombination", "undrained", "weights", "best_model_undrained.keras"),
    "no_data": os.path.join("no_data", "no_data.img"),
    "dataset_path": os.path.join("..", "Datasets", "TestArea")
}


if __name__ == "__main__":
    try:
        CSC = False

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        window_no_data_dictionary = {}
        window_no_data_dictionary['value_to_change'] = 0

        # LOCAL VAR
        dump_all_results_in_on_array = list()
        # In memory rebuilt
        reconstructed_undrained_col, reconstructed_drained_col = None, None

        # get the colors and config
        colors  = return_configuration()

        number_of_worker = configuration["number_of_cpus"]
        print(f'[INFO] Number of worker {number_of_worker}')

        parser = argparse.ArgumentParser()
        parser.add_argument("--row",
                            help="row of the block",
                            type=int,
                            required=True)

        parser.add_argument("--col",
                            help="col of the block",
                            type=int,
                            required=True)

        parser.add_argument("--maxy",
                            help="maxy in meters",
                            type=float,
                            required=True)

        parser.add_argument("--miny",
                            help="miny in meters",
                            type=float,
                            required=True)

        parser.add_argument("--minx",
                            help="minx E in meters",
                            type=float,
                            required=True)

        parser.add_argument("--maxx",
                            help="maxx in meters",
                            type=float,
                            required=True)

        parser.add_argument("--width_split_size_in_pixel",
                            type=int,
                            help="width split size in pixel",
                            required=True)

        parser.add_argument("--height_split_size_in_pixel",
                            type=int,
                            help="height split size in pixel",
                            required=True)

        parser.add_argument("--width_split_size_in_meters",
                            type=float,
                            help="width split size in meters",
                            required=True)

        parser.add_argument("--height_split_size_in_meters",
                            type=float,
                            help="height split size in meters",
                            required=True)

        parser.add_argument("--raster_size",
                            type=int,
                            help="raster size in meters",
                            required=True)

        parser.add_argument("--area",
                            type=str,
                            default="TESTAREA",
                            help="zone",
                            required=False)

        # parse
        args = parser.parse_args()

        # build the target
        main_target = {
            'row': args.row,
            'col': args.col,
            'target_raster_max_y': args.maxy,
            'target_raster_min_y': args.miny,
            'target_raster_min_x': args.minx,
            'target_raster_max_x': args.maxx,
            'height_split_size_in_pixel': args.height_split_size_in_pixel,
            'width_split_size_in_pixel': args.width_split_size_in_pixel,
            'height_split_size_in_meters': args.height_split_size_in_meters,
            'width_split_size_in_meters': args.height_split_size_in_meters,
            'target_raster_spatial_x_pixel_size_meters': args.raster_size,
            'zone': args.area
        }

        best_combination_drained = load_best_dictionary_from_input_selection_results(input_selection_results_path=os.path.join("..", args.area, "Results", f"input_selection_second_stage_DRAINED", "summary_best.txt") )
        best_combination_undrained = load_best_dictionary_from_input_selection_results(input_selection_results_path=os.path.join("..", args.area, "Results", f"input_selection_second_stage_UNDRAINED", "summary_best.txt") )

        print(f"[TARGET] Target acquired {main_target}")
        
        # divide and conquer the box
        list_of_tasks = divide_and_conquer_the_raster(main_target=main_target, how_big_I_want_the_chop=10)

        print(f"[LOADING] The input")
        # load the undrained dataset
        undrained_stored_input = routine_load_the_dataset(data_to_load=best_combination_undrained,
                                                          dataset_path=configuration["dataset_path"])

        # load the drained dataset
        drained_stored_input = routine_load_the_dataset(data_to_load=best_combination_drained,
                                                        dataset_path=configuration["dataset_path"])

        window_no_data_dictionary = routine_get_no_data_values(current_dictionary_of_tif=undrained_stored_input,
                                                               no_data_dictionary=window_no_data_dictionary)

        window_no_data_dictionary = routine_get_no_data_values(current_dictionary_of_tif=drained_stored_input,
                                                               no_data_dictionary=window_no_data_dictionary)

        del undrained_stored_input
        del drained_stored_input

        # create the folder structure
        main_undrained_path = os.path.join(f"output_{main_target['zone']}", f"worker_{main_target['row']}", f"worker_{main_target['row']}_{main_target['col']}_undrained")
        main_drained_path = os.path.join(f"output_{main_target['zone']}", f"worker_{main_target['row']}", f"worker_{main_target['row']}_{main_target['col']}_drained")
        
        dump = f'output/worker_{main_target["row"]}'

        if not os.path.exists(f'{dump}'):
            os.makedirs(f'{dump}', exist_ok=True)

        if not os.path.exists(main_undrained_path):
            os.makedirs(main_undrained_path, exist_ok=True)
            
        if not os.path.exists(main_drained_path):
            os.makedirs(main_drained_path, exist_ok=True)
        

        # get how many rounds to do
        number_of_rounds = len(list_of_tasks)
        print(f"[ROUND] Number of round is {number_of_rounds}")
        # decompose the lists and start the training
        for current_round in range(int(number_of_rounds)):

            # get the ref to N and the lists of E and the distances
            North = list_of_tasks[current_round]['North']
            East = list_of_tasks[current_round]['East']
            # I have fast indexing to not complicate the data too much
            round_distances = list_of_tasks[current_round]['distances']
            print(f"[MAPPING] Round {current_round} over {number_of_rounds}")
            print(f"[MAPPING round {current_round}] current N {North} - current list of E {East} len of the round {len(East)}")
            print(f"[MAPPING distances {current_round}] Distances {round_distances} {len(round_distances)}")

            timer = time.time()
       
            result = Parallel(n_jobs=len(East), backend="multiprocessing", max_nbytes='100M')(
                delayed(pixelwise_classification_prediction)(
                    config=configuration,
                    spatial_pixel_resolution=main_target["target_raster_spatial_x_pixel_size_meters"],
                    undrained_best_combination=best_combination_undrained,
                    drained_best_combination=best_combination_drained,
                    current_east=East[i],
                    current_north=North,
                    worker=i,
                    from_which_row=0,
                    from_which_col=0,
                    height_size_in_pixel=round_distances[i][0],
                    width_size_in_pixel=round_distances[i][1],
                    undrained_weights_folder=configuration["undrained_weights"],
                    drained_weights_folder=configuration["drained_weights"],
                    class_id_list=list(colors.keys()),
                    no_data_raster=configuration["no_data"],
                    nodata_value=configuration["no_data_value"],
                    window_no_data_dictionary=window_no_data_dictionary,
                ) for i in range(len(East)))

            print(f"[TIMER] Timer -> {time.time() - timer}")
            print(f"[POST PROCESS] Process complete rebuilt the matrix")

            # let s add the result in the list
            if configuration['DUMP']:
                dump_all_results_in_on_array.append(result)

            # do the rows
            reconstructed_undrained_row, reconstructed_drained_row = routine_rebuilt_the_rows(result=result)

            # Do the same job row wise
            if reconstructed_undrained_col is None:
                reconstructed_undrained_col = reconstructed_undrained_row
                # put none so we re-do again another row
                reconstructed_undrained_row = None
            else:
                reconstructed_undrained_col = np.concatenate((reconstructed_undrained_col, reconstructed_undrained_row),
                                                             axis=0)

            if reconstructed_drained_col is None:
                reconstructed_drained_col = reconstructed_drained_row
                # put none so we re-do again another row
                reconstructed_drained_row = None
            else:
                reconstructed_drained_col = np.concatenate((reconstructed_drained_col, reconstructed_drained_row),
                                                           axis=0)
            print(f"[MATRIX COL WISE] Shape drained {reconstructed_drained_col.shape} Shape undrained {reconstructed_undrained_col.shape}")

        # save the final matrix
        np.save(f'{main_undrained_path}', reconstructed_undrained_col)
        np.save(f'{main_drained_path}', reconstructed_drained_col)
        
    except Exception as ex:
        print(f"Main pixelwise classification on csc throws exception {ex}")
        print(traceback.format_exc())


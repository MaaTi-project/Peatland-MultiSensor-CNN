import numpy as np
import tensorflow as tf
import os
import traceback
import rasterio
import gc
import shutil
from osgeo import gdal 
from Configuration.input_selection_placeholder import InputSelectionPlaceHolder

pixelwise_classification_options = {
    "TRAIN": False,
    "OVERRIDE_RASTER_SIZE": 10
}


def return_configuration():
    colors = {
        11: (255, 0, 0),
        12: (0, 255, 0),
        13: (188, 143, 143),
        14: (0, 0, 255),
        21: (255, 255, 0),
        22: (0, 255, 255),
        23: (0, 244, 244),
        31: (255, 0, 255),
        32: (192, 192, 192),
        33: (128, 128, 128),
        41: (128, 0, 0),
        42: (128, 128, 0),
        43: (0, 128, 0),
        44: (128, 0, 128),
        45: (0, 128, 128),
        46: (0, 0, 128),
        47: (255, 165, 0),
        48: (176, 196, 222),
        51: (238, 232, 170),
        52: (154, 205, 50),
        53: (175, 238, 238),
        54: (65, 105, 225),
        55: (138, 43, 226),
        56: (221, 160, 221),
        61: (255, 105, 180),
        62: (255, 250, 205),
        63: (240, 255, 255),
        64: (205, 133, 63),
        65: (178, 34, 34),
        71: (255, 228, 225),
        72: (245, 255, 250),
        81: (230, 230, 250),
        82: (240, 248, 255),
        83: (255, 20, 147),
        84: (254, 19, 146),
        91: (105, 105, 105),
        92: (104, 104, 104),
        101: (30, 144, 255),
        102: (220, 220, 220),
        103: (127, 255, 212),
        104: (126, 254,211),
        120: (34, 139, 34),
        130: (220, 20, 60),
        140: (33, 138, 33)
    }

    return colors



def check_raster_divisibility(number_to_check, spatial_pixel_resolution):
    try:
        # check the mod
        if np.mod(number_to_check, spatial_pixel_resolution) != 0:
            number_to_check = round(number_to_check / spatial_pixel_resolution) * spatial_pixel_resolution
        # assert just in case
        assert np.mod(number_to_check, spatial_pixel_resolution) == 0, \
            f"[ASSERTION ERROR] {number_to_check} is not divisible by {spatial_pixel_resolution}"

        return number_to_check
    except Exception as ex:
        print(f"[EXCEPTION] Check raster divisibility throws exception {ex}")


def create_sub_raster_multiprocessing(target_raster, number_of_worker, write_parser=False, launch_cmd=None, area=None):
    try:

        if write_parser:
            handler = open(os.path.join("launch_commander", f"{launch_cmd}"), 'w')

        list_to_compute, distances = list(), list()

        maxy = float(target_raster['target_raster_max_y'])
        miny = float(target_raster['target_raster_min_y'])
        minx = float(target_raster['target_raster_min_x'])
        maxx = float(target_raster['target_raster_max_x'])

        spatial_pixel_resolution = target_raster['target_raster_spatial_x_pixel_size_meters']
        print(f"[Target raster] Maxy = {maxy} miny = {miny} minx = {minx} maxx = {maxx} pi_x {spatial_pixel_resolution}")

        # define the target of each sub raster -- height
        target_height_in_meters = check_raster_divisibility(number_to_check=round(maxy - miny),
                                                            spatial_pixel_resolution=spatial_pixel_resolution)

        distance_height_in_meters = check_raster_divisibility(number_to_check=(target_height_in_meters / float(number_of_worker)),
                                                              spatial_pixel_resolution=spatial_pixel_resolution)

        # assert that all the number produced are ok
        assert np.mod(target_height_in_meters - (distance_height_in_meters * (
                    number_of_worker - 1)), spatial_pixel_resolution) == 0, f"[ASSERTION ERROR] Checking all the number are ok (HEIGHT SIDE)"

        # define the target of each sub raster -- width
        target_width_in_meters = check_raster_divisibility(number_to_check=round(maxx - minx),
                                                           spatial_pixel_resolution=spatial_pixel_resolution)

        distance_width_in_meters = check_raster_divisibility(number_to_check=(target_width_in_meters / float(number_of_worker)),
                                                             spatial_pixel_resolution=spatial_pixel_resolution)

        # assert that all the number produced are ok
        assert np.mod(target_width_in_meters - (distance_width_in_meters * (
                    number_of_worker - 1)), spatial_pixel_resolution) == 0, f"[ASSERTION ERROR] Checking all the number are ok (WIDTH SIDE)"

        for row in range(0, number_of_worker):
            # calculate the start point in meters (y wise)
            start_N = maxy - (row * distance_height_in_meters)

            # add normal distance
            if row != number_of_worker - 1:
                end_N = (start_N - distance_height_in_meters)
            else:
                # last element is = to miny the end of the raster
                end_N = miny

            # reset the list for each row
            temp_E = list()
            for col in range(0, number_of_worker):
                # calculate the start point meters (x wise)
                start_E = minx + (col * distance_width_in_meters)

                # same here but col wise 4

                if col != number_of_worker -1:
                    end_E = (start_E + distance_width_in_meters)
                else:
                    end_E = maxx
                # I do not know If I need list let s keep it for now
                temp_E.append(start_E)

                # add distances to the distances arrays (y,x)
                distances.append([round((start_N - end_N) / spatial_pixel_resolution), round((end_E - start_E) / spatial_pixel_resolution)])

                if write_parser:
                    print(f"[SUMMARY SPLITS] Row {row} Col {col}")
                    print(f"[height split] Meters -> {start_N - end_N} Pixels -> {round((start_N - end_N) / spatial_pixel_resolution)}")
                    print(f"[width split] Meters -> {end_E - start_E} Pixels -> {round((end_E - start_E) / spatial_pixel_resolution)}")
                    print("------")

                    if area is None:
                        # prepare the string builder
                        string_builder = f"python pixelwise_classification.py --row {row} --col {col} --maxy {start_N} " \
                                        f"--miny {end_N} --minx {start_E} --maxx {end_E} " \
                                        f"--height_split_size_in_pixel {round((start_N - end_N) / spatial_pixel_resolution)} " \
                                        f"--width_split_size_in_pixel {round((end_E - start_E) / spatial_pixel_resolution)} " \
                                        f"--height_split_size_in_meters {round((start_N - end_N))} " \
                                        f"--width_split_size_in_meters {round((end_E - start_E))} " \
                                        f"--raster_size {spatial_pixel_resolution}" 
                    else:
                        # prepare the string builder
                        string_builder = f"python pixelwise_classification.py --row {row} --col {col} --maxy {start_N} " \
                                        f"--miny {end_N} --minx {start_E} --maxx {end_E} " \
                                        f"--height_split_size_in_pixel {round((start_N - end_N) / spatial_pixel_resolution)} " \
                                        f"--width_split_size_in_pixel {round((end_E - start_E) / spatial_pixel_resolution)} " \
                                        f"--height_split_size_in_meters {round((start_N - end_N))} " \
                                        f"--width_split_size_in_meters {round((end_E - start_E))} " \
                                        f"--raster_size {spatial_pixel_resolution} " \
                                        f"--area {area}" 


                    handler.write(string_builder)
                    handler.write('\n')

            # add all the N and E we in the list to compute
            list_to_compute.append([start_N, temp_E])

        if write_parser:
            handler.close()
        #print(f"[GRID] {list_to_compute}")
        #print(f"[DISTANCES] Distance in R and C are {distances}")

        print(f"[GRID] The len of the list to compute is {len(list_to_compute)} each element contains {len(list_to_compute[0][1])} "
              f"within a total of {len(list_to_compute) * len(list_to_compute[0][1])} grids"
              f"The distances of each grid element are {len(distances)}")

        return list_to_compute, distances

    except Exception as ex:
        print(f"[EXCEPTION] Create sub raster for multirocessing throws exception {ex}")


def yield_the_sliding_windows(tif, E, N, window_dimension, raster_size, current_channel, window_no_data_dictionary):
    try:
        # create the windows:
        if current_channel == "channel_1":
            E = E - 20
            N = N + 20

        elif current_channel == "channel_2":
            E = E - 24
            N = N + 24

        else:
            E = E - 32
            N = N + 32

        coordinates = []
        for y in range(0, window_dimension):
            N_point = N - y * raster_size
            for x in range(0, window_dimension):
                E_point = E + x * raster_size
                coordinates.append((E_point, N_point))

        values_with_need = [tiff_val for tiff_val in tif.sample(coordinates)]
        # create the window I m testing with float
        window = np.array(values_with_need).astype('uint8').reshape((window_dimension, window_dimension, -1))

        # put the needed value to the window
        no_data_window = window_no_data_dictionary[current_channel]

        for v in no_data_window:
            window[window == v] = window_no_data_dictionary['value_to_change']

        # normalize
        window = window / 255.0

        return window
    except Exception as ex:
        print(f"[EXCEPTION] Yield the sliding windows throws exception {ex}")
        print(traceback.print_exc())


def get_if_there_is_no_data(current_N, current_E, loaded_novalue_tif):
    try:
        coordinates = []
        coordinates.append((current_E, current_N))
        values_with_need = [tiff_val for tiff_val in loaded_novalue_tif.sample(coordinates)]
        return values_with_need[0][0]
    except Exception as ex:
        print(f"[EXCEPTION] Get if there is no data throws exception {ex}")


def routine_prepare_data_for_prediction(configuration, stored_input, point_N_to_analyse, point_E_to_analyse, window_no_data_dictionary,spatial_pixel_resolution):
    try:
        data_to_predict = {}
        # let s start with the channels:
        for channel in stored_input.keys():

            # we have to stack the windows from the same inputs
            concatenated = np.zeros((configuration[channel], configuration[channel], 0))
            for ctn, obj in enumerate(stored_input[channel]):
                # create the windows
                window = yield_the_sliding_windows(tif=obj,
                                                   E=point_E_to_analyse,
                                                   N=point_N_to_analyse,
                                                   window_dimension=configuration[channel],
                                                   raster_size=spatial_pixel_resolution,
                                                   current_channel=channel,
                                                   window_no_data_dictionary=window_no_data_dictionary)

                if len(stored_input[channel]) > 1:
                    concatenated = np.concatenate((concatenated, window), axis=2)
                else:
                    concatenated = window

                if concatenated.shape[0] != configuration[channel] or concatenated.shape[1] != configuration[channel]:
                    print(f"[DIMENSION] Windows dimension for channel {channel} does not match")
                    continue

            # here the data is ready to be predicted
            data_to_predict[f"input_{channel}"] = np.expand_dims(concatenated, axis=0)
        return data_to_predict
    except Exception as ex:
        print(f'[EXCEPTION] routine_prepare_data_for_prediction throws exception {ex}')


def routine_load_the_dataset(data_to_load, dataset_path):
    try:
        stored_input = {}
        # load the file once
        for channel in data_to_load.keys():

            for set_of_input in data_to_load[channel]:
                path_to_image = os.path.join(dataset_path, f"{set_of_input.satellite_name.split('_')[0]}", f"{set_of_input.image_name}.tif")
                if channel not in stored_input.keys():
                    stored_input[channel] = list() 

                stored_input[channel].append(rasterio.open(path_to_image))
            # remove the list to save space
            print(f"[MEMORY] I deleted the temp list in garbage collection")
            gc.collect()
            
        return stored_input
    except Exception as ex:
        print(f"[EXCEPTION] routine load the dataset throws exception {ex}")


def routine_clean_the_raster(stored_input):
    try:
        for key, values in stored_input.items():
            temp = stored_input[key]
            for element in temp:
                element.close()
        del stored_input
    except Exception as ex:
        print(f"[EXCEPTION] routine clean the rasters throws exception {ex}")


def divide_and_conquer_the_raster(main_target, how_big_I_want_the_chop=None):
    try:
        list_to_compute = list()

        spatial_pixel_resolution = main_target['target_raster_spatial_x_pixel_size_meters']
        height_split_size_in_pixel = main_target['height_split_size_in_pixel']
        width_split_size_in_pixel = main_target['width_split_size_in_pixel']

        maxy = float(main_target['target_raster_max_y'])
        miny = float(main_target['target_raster_min_y'])
        minx = float(main_target['target_raster_min_x'])
        maxx = float(main_target['target_raster_max_x'])

        if how_big_I_want_the_chop is None:
            how_big_I_want_the_chop = spatial_pixel_resolution
            cunk_I_want = spatial_pixel_resolution
        else:
            cunk_I_want = how_big_I_want_the_chop

        # if we have not perfect box
        if np.mod(height_split_size_in_pixel,  cunk_I_want) != 0:
            number_of_split_height_size_in_pixel = np.floor(height_split_size_in_pixel / cunk_I_want).astype(int) + 1
        else:
            number_of_split_height_size_in_pixel = np.floor(height_split_size_in_pixel / cunk_I_want).astype(int)

        # if we have not perfect box
        if np.mod(width_split_size_in_pixel,  cunk_I_want) != 0:
            number_of_split_width_size_in_pixel = np.floor(width_split_size_in_pixel / cunk_I_want).astype(int) + 1
        else:
            number_of_split_width_size_in_pixel = np.floor(width_split_size_in_pixel / cunk_I_want).astype(int)

        for row in range(1, number_of_split_height_size_in_pixel + 1):

            # calculate the start point in meters (y wise)
            start_N = maxy - (row - 1) * spatial_pixel_resolution * how_big_I_want_the_chop
            # reset the list for each row
            # add normal distance
            if row != number_of_split_height_size_in_pixel:
                end_N = (start_N - spatial_pixel_resolution * how_big_I_want_the_chop)
            else:
                # last element is = to miny the end of the raster
                end_N = miny

            temp_E, distances = list(), list()
            for col in range(1, number_of_split_width_size_in_pixel + 1):
                # calculate the start point meters (x wise)
                start_E = minx + (col - 1) * spatial_pixel_resolution * how_big_I_want_the_chop

                # same here but col wise
                if col != number_of_split_width_size_in_pixel:
                    end_E = (start_E + spatial_pixel_resolution * how_big_I_want_the_chop)
                else:
                    end_E = maxx

                temp_E.append(start_E)

                # add distances to the distances arrays (y,x)
                distances.append([round((start_N - end_N) / spatial_pixel_resolution), round((end_E - start_E) / spatial_pixel_resolution)])

            # check the len
            assert len(distances) == len(temp_E), f"[ASSERTION ERROR] The len of distance {len(distances)} is different than the len of E {len(temp_E)}"

            # add all the N and E we in the list to compute
            #list_to_compute.append([start_N, temp_E])
            list_to_compute.append({'North': start_N,
                                    'East': temp_E,
                                    'distances': distances,
                                    'length_of_E': len(temp_E),
                                    'length_of_distances': len(distances)})

        #print(f"[GRID] {list_to_compute}")

        return list_to_compute

    except Exception as ex:
        print(f"[EXCEPTION] Divide and conquer the raster throws exception {ex}")


def routine_rebuilt_the_rows(result):
    try:
        reconstructed_undrained_row, reconstructed_drained_row = None, None
        undrained_ctn, drained_ctn = 0, 0
        for mini_array_undrained, mini_array_drained in result:
            if undrained_ctn == 0:
                undrained_ctn += 1
                reconstructed_undrained_row = np.array(mini_array_undrained)
            else:
                reconstructed_undrained_row = np.concatenate((reconstructed_undrained_row, np.array(mini_array_undrained)), axis=1)
            #print(f"[MATRIX ROW WISE] Shape undrained {reconstructed_undrained_row.shape}")

            if drained_ctn == 0:
                drained_ctn +=1
                reconstructed_drained_row = np.array(mini_array_drained)
            else:
                reconstructed_drained_row = np.concatenate((reconstructed_drained_row, np.array(mini_array_drained)), axis=1)
            #print(f"[MATRIX ROW WISE] Shape drained {reconstructed_drained_row.shape}")
        return reconstructed_undrained_row, reconstructed_drained_row
    except Exception as ex:
        print(f"[EXCEPTION] Routine rebuilt the rows throws exception {ex}")


def routine_get_no_data_values(current_dictionary_of_tif, no_data_dictionary):
    try:
        for k, v in current_dictionary_of_tif.items():
            # get the channel
            for values in current_dictionary_of_tif[k]:
                # check if the key is already in the dictionary of input
                if k not in no_data_dictionary.keys():
                    temp = list()
                else:
                    temp = no_data_dictionary[k]
                for no_data in values.nodatavals:
                    if no_data not in temp:
                        temp.append(no_data)
                no_data_dictionary[k] = temp
                values.close()
        print(f"[NO DATA] no_data_dictionary found window no data value {no_data_dictionary}")
        return no_data_dictionary
    except Exception as ex:
        print(f"[EXCEPTION] Routine get no data value {ex}")


def load_the_dictionary_with_best_combination(path):
    best_combination = {}
    opener = open(path, 'r')
    
    for row in opener.readlines():
        splitted = row.strip().split(',')
 
        if splitted[1] not in best_combination.keys():
            best_combination[splitted[1]] = list() 
        
        best_combination[splitted[1]].append(splitted[0])

    return best_combination

def load_the_data(list_of_pathes):

    concatenated = None 
    for element in list_of_pathes:
        temp_windows = list() 
        label_set = list() 
        for windows in os.listdir(element):
            label_set.append(int(windows.split('_')[1]))
            full_path = os.path.join(element, windows)
            temp_windows.append(np.load(full_path))
        if concatenated is None:
            concatenated = np.array(temp_windows)
            
        else:
            concatenated = np.concatenate((concatenated, np.array(temp_windows)), axis=3)
    concatenated = concatenated / 255. 
    label_set = np.array(label_set)
    return concatenated, label_set 

def check_the_bounds(target_raster):
    try:

        # Check that the x-min border is divisible by spatial resolution
        if np.mod(target_raster['target_raster_min_x'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['target_raster_min_x'] = round(float(target_raster['target_raster_min_x']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])
        # Double-check that the borders are divisible by spatial resolution
        assert np.mod(target_raster['target_raster_min_x'],float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        # Check that the x-max border is divisible by spatial resolution
        if np.mod(target_raster['target_raster_max_x'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['target_raster_max_x'] = round(float(target_raster['target_raster_max_x']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])
        # Double-check that the borders are divisible by spatial resolution
        assert np.mod(target_raster['target_raster_max_x'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        # Check that the y-min border is divisible by spatial resolution
        if np.mod(target_raster['target_raster_min_y'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['target_raster_min_y'] = round(float(target_raster['target_raster_min_y']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])
        # Double-check that the borders are divisible by spatial resolution
        assert np.mod(target_raster['target_raster_min_y'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        # Check that the y-max border is divisible by spatial resolution
        if np.mod(target_raster['target_raster_max_y'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['target_raster_max_y'] = round(float(target_raster['target_raster_max_y']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])
        # Double-check that the borders are divisible by spatial resolution
        assert np.mod(target_raster['target_raster_max_y'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        if np.mod(target_raster['number_of_x_pixels'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) !=0:
            target_raster['number_of_x_pixels'] = round(float(target_raster['number_of_x_pixels']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])

        assert np.mod(target_raster['number_of_x_pixels'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        if np.mod(target_raster['number_of_y_pixels'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['number_of_y_pixels'] = round(float(target_raster['number_of_y_pixels']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_y_pixel_size_meters'])

        assert np.mod(target_raster['number_of_y_pixels'], float(target_raster['target_raster_spatial_y_pixel_size_meters'])) == 0

        if np.mod(target_raster['raster_width'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['raster_width'] = round(float(target_raster['raster_width']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])

        assert np.mod(target_raster['raster_width'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        if np.mod(target_raster['raster_height'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) != 0:
            target_raster['raster_height'] = round(float(target_raster['raster_height']) / float(target_raster['target_raster_spatial_x_pixel_size_meters'])) * float(target_raster['target_raster_spatial_x_pixel_size_meters'])

        assert np.mod(target_raster['raster_height'], float(target_raster['target_raster_spatial_x_pixel_size_meters'])) == 0

        return target_raster
    except Exception as ex:
        print(f"Check the bounds throws exception {ex}")


def define_the_target_raster(dictionary_of_combination, dataset_path):
    try:
        target_raster_info = {}
        target_raster_info['target_raster_min_x'] = np.NINF
        target_raster_info['target_raster_max_x'] = np.PINF
        target_raster_info['target_raster_min_y'] = np.NINF
        target_raster_info['target_raster_max_y'] = np.PINF
        target_raster_info['number_of_x_pixels'] = 0
        target_raster_info['number_of_y_pixels'] = 0

        if pixelwise_classification_options["OVERRIDE_RASTER_SIZE"] != None:
            target_raster_info['target_raster_spatial_x_pixel_size_meters'] = pixelwise_classification_options[
                "OVERRIDE_RASTER_SIZE"]
            target_raster_info['target_raster_spatial_y_pixel_size_meters'] = pixelwise_classification_options[
                "OVERRIDE_RASTER_SIZE"]
        else:
            target_raster_info['target_raster_spatial_x_pixel_size_meters'] = np.PINF
            target_raster_info['target_raster_spatial_y_pixel_size_meters'] = np.PINF

        for channel in dictionary_of_combination.keys():

            for set_of_input in dictionary_of_combination[channel]:
                print(f'[WALK] I m walking on {channel} with input {set_of_input}')
                path_to_image = os.path.join(dataset_path, f"{set_of_input.satellite_name.split('_')[0]}", f"{set_of_input.image_name}.tif")
                
                tiff_data = gdal.Open(f"{path_to_image}", gdal.GA_ReadOnly)

                # get the tif profile
                geoprofile = tiff_data.GetGeoTransform()

                minx = geoprofile[0]  # x-koordinaatin minimi (lansi) W
                maxy = geoprofile[3]  # y-koordinaatin maksimi (pohjoinen) N
                pix_x = geoprofile[1]  # pikselikoko x-suunnassa; positiivinen (kasvaa lanteen)
                pix_y = geoprofile[5]  # pikselikoko y-suunnassa; negatiivinen (pienenee etelaan)
                x_ext = tiff_data.RasterXSize  # rasterin koko (pikselia) x-suunnassa
                y_ext = tiff_data.RasterYSize  # rasterin koko (pikselia) y-suunnassa
                maxx = minx + pix_x * x_ext  # x-koordinaatin maksimi (ita) E
                miny = maxy + pix_y * y_ext  # y-koordinaatin minimi (etela) S

                print(f"[PROFILE {channel}] image: {set_of_input} has profile:\n" 
                      f"[min_x] {minx}\n"
                      f"[max_y] {maxy}\n"
                      f"[pix_x] {pix_x}\n"
                      f"[pix_y] {pix_y}\n"
                      f"[x_ext] {x_ext}\n"
                      f"[y_ext] {y_ext}\n"
                      f"[max_x] {maxx}\n"
                      f"[min_y] {miny}\n")

                # load the georef into a dict
                current_tif = {}
                current_tif['x_min_boundary_coord'] = minx
                current_tif['x_max_boundary_coord'] = maxx
                current_tif['y_min_boundary_coord'] = miny
                current_tif['y_max_boundary_coord'] = maxy
                current_tif['number_of_x_pixels'] = x_ext
                current_tif['number_of_y_pixels'] = y_ext
                current_tif['spatial_pixel_size_meters_x'] = np.abs(pix_x)
                current_tif['spatial_pixel_size_meters_y'] = np.abs(pix_x)

                if pixelwise_classification_options["OVERRIDE_RASTER_SIZE"] == None:
                    # Step 2: Add target raster info
                    if target_raster_info['target_raster_spatial_x_pixel_size_meters'] > current_tif['spatial_pixel_size_meters_x']:
                        target_raster_info['target_raster_spatial_x_pixel_size_meters'] = current_tif['spatial_pixel_size_meters_x']

                    if target_raster_info['target_raster_spatial_y_pixel_size_meters'] > current_tif['spatial_pixel_size_meters_y']:
                        target_raster_info['target_raster_spatial_y_pixel_size_meters'] = current_tif['spatial_pixel_size_meters_y']

                if target_raster_info['target_raster_min_x'] < current_tif['x_min_boundary_coord']:
                    target_raster_info['target_raster_min_x'] = current_tif['x_min_boundary_coord']

                if target_raster_info['target_raster_max_x'] > current_tif['x_max_boundary_coord']:
                    target_raster_info['target_raster_max_x'] = current_tif['x_max_boundary_coord']

                if target_raster_info['target_raster_min_y'] < current_tif['y_min_boundary_coord']:
                    target_raster_info['target_raster_min_y'] = current_tif['y_min_boundary_coord']

                if target_raster_info['target_raster_max_y'] > current_tif['y_max_boundary_coord']:
                    target_raster_info['target_raster_max_y'] = current_tif['y_max_boundary_coord']

                # get the size of the pixels
                target_raster_info['number_of_x_pixels'] = round(target_raster_info['target_raster_max_x'] - target_raster_info['target_raster_min_x']) / float(target_raster_info['target_raster_spatial_x_pixel_size_meters'])
                target_raster_info['number_of_y_pixels'] = round(target_raster_info['target_raster_max_y'] - target_raster_info['target_raster_min_y']) / float(target_raster_info['target_raster_spatial_x_pixel_size_meters'])

                target_raster_info['raster_height'] = round(target_raster_info['target_raster_max_y'] - target_raster_info['target_raster_min_y'])
                target_raster_info['raster_width'] = round(target_raster_info['target_raster_max_x'] - target_raster_info['target_raster_min_x'])

                target_raster_info['number_of_x_pixels'] = int(target_raster_info['number_of_x_pixels'])
                target_raster_info['number_of_y_pixels'] = int(target_raster_info['number_of_y_pixels'])
        return target_raster_info
    
    except Exception as ex:
        print(f"Define the target raster throws exception {ex}")


def load_best_dictionary_from_input_selection_results(input_selection_results_path):
    list_of_stuff = list()
    current_handler = {}
    best_combination = {}

    handler = open(input_selection_results_path, 'r')

    for line in handler.readlines():
        current_row = line.strip().split(' ')
        
        if current_row[0] != 'Mean':

            if current_row[0] == '+' or current_row[0] == '=': 
                continue

            list_of_stuff.append(line.strip().split(','))
        else:
            accuracy = float(current_row[-1])
            current_handler[accuracy] = list_of_stuff
            list_of_stuff = list() 
    
    handler.close() 

    best_accuracy = max(current_handler.keys()) 
    temp = current_handler[best_accuracy]
    for element_ctn, element in enumerate(temp):
        channel = element[0]
        path = element[1]
        
        if channel not in best_combination.keys():
            best_combination[channel] = list()
            
        temp_win = list() 
        temp_label = list () 

        for windows in os.listdir(path):
            temp_label.append(int(windows.split('_')[1]))
            
            temp_win.append(
                np.load((os.path.join(path, windows)))
            )
                
        
        best_combination[channel].append(InputSelectionPlaceHolder(
            channel=channel,
            id=element_ctn,
            satellite_name=path.split('/')[-2],
            image_name=path.split('/')[-1],
            full_path=path,
            label_set=np.array(temp_label, dtype='int'),
            data=np.array(temp_win, dtype='float'),
            shape=np.array(temp_win, dtype='float').shape
        ))
    
    print(f"[LOAD] Best combination is loaded {best_combination}")
    return best_combination

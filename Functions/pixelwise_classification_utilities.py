import traceback
from osgeo import gdal
import numpy as np
import rasterio
from Functions.Utilities import *
from Configuration.configuration import *

"""
def get_the_correct_area(path_area_sample, path_area_to_change, output_path):
    try:
        # load the image we need to copy the profile
        tiff_data = gdal.Open(f"{path_area_sample}")

        driver = gdal.GetDriverByName("GTiff")

        # get the tif profile
        geotransform = tiff_data.GetGeoTransform()

        print(f"[GEO] Sample area geo trans {geotransform}")

        minx = geotransform[0]  # x-koordinaatin minimi (lansi)
        maxy = geotransform[3]  # y-koordinaatin maksimi (pohjoinen)
        pix_x = geotransform[1]  # pikselikoko x-suunnassa; positiivinen (kasvaa lanteen)
        pix_y = geotransform[5]  # pikselikoko y-suunnassa; negatiivinen (pienenee etelaan)
        x_ext = tiff_data.RasterXSize  # rasterin koko (pikselia) x-suunnassa
        y_ext = tiff_data.RasterYSize  # rasterin koko (pikselia) y-suunnassa
        maxx = minx + pix_x * x_ext  # x-koordinaatin maksimi (ita)
        miny = maxy + pix_y * y_ext  # y-koordinaatin minimi (etela)

        new_raster = gdal.Warp("", path_area_to_change, outputBounds=(minx, miny, maxx, maxy), format="vrt")
        arr = new_raster.ReadAsArray()

        [cols, rows] = arr.shape
        outdata = driver.Create(output_path, rows, cols)
        outdata.SetGeoTransform(new_raster.GetGeoTransform())
        outdata.SetProjection(new_raster.GetProjection())
        outdata.WriteArray(arr)

        print(f"[GEO] New raster geo trans {new_raster.GetGeoTransform()}")

        return output_path
    except Exception as ex:
        print(f"Get the correct area throws exception {ex}")

def do_the_pixelwise_classification(configuration, input_data, weights_folder, save_folder, worker_number,
                                    total_height, total_width):
    try:
        stored_input = {}

        # load the file once
        for channel in configuration["channel_list"]:
            temp = list()

            for obj in input_data[channel]:
                temp.append(rasterio.open(obj.path))

            # stored the file we need in a dict so we do not need to load it each time
            stored_input[channel] = temp

        # data_to_predict = {}
        class_id_list = list(colors.keys())
        # load the model
        model = tf.keras.models.load_model(weights_folder)
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        placeholder = np.zeros((total_height, total_width))
        # slide for each pixel in row
        for R in range(total_height):
            if R % 5 == 0:
                print(f"[SAVE] Saving")
                np.save(f"{save_folder}/slice_{worker_number}", placeholder)
            # slide for each pixel in col
            for C in range(total_width):

                data_to_predict = {}
                # let s start with the channels:
                for channel in configuration["channel_list"]:
                    # we have to stack the windows from the same inputs
                    concatenated = np.zeros((input_data[channel][0].window_dimension, input_data[channel][0].window_dimension, 0))
                    for ctn, obj in enumerate(input_data[channel]):

                        # create the windows
                        window = yield_the_sliding_windows(tif=stored_input[channel][ctn],
                                                           E=obj.list_E[worker_number][0] + (C + configuration["OFFSET"]) * obj.raster_size,
                                                           N=obj.list_N[worker_number][0] - (R + configuration["OFFSET"]) * obj.raster_size,
                                                           window_dimension=obj.window_dimension,
                                                           raster_size=obj.raster_size,
                                                           current_channel=channel)

                        if len(input_data[channel]) > 1:
                            concatenated = np.concatenate((concatenated, window), axis=2)
                        else:
                            concatenated = window

                        if concatenated.shape[0] != configuration[channel] or concatenated.shape[1] != configuration[
                            channel]:
                            print(f"[DIMENSION] Windows dimension for channel {channel} does not match")
                            continue

                    # if configuration["STDOUT"]:
                    # print(f"[WINDOWS DIMENSION] R -> {R} C -> {C} -> {channel} -- {concatenated.shape} -> {input_data[channel][0].raster_size}")

                    # here the data is ready to be predicted
                    data_to_predict[f"input_{channel}"] = np.expand_dims(concatenated, axis=0)

                # predict
                prediction = model.predict(data_to_predict, verbose=configuration["VERBOSE"])
                prediction = np.squeeze(prediction)
                prediction = np.argmax(prediction, axis=-1)

                if configuration["STDOUT"]:
                    print(
                        f"[PREDICTION] Worker -> {worker_number} Row -> {R} Col -> {C} is {class_id_list[prediction]}")
                # save each pixel in the placeholder
                placeholder[R, C] = class_id_list[prediction]

        # save
        print(f"[SAVE] Saving")
        np.save(f"{save_folder}/slice_{worker_number}", placeholder)
        # end

    except Exception as ex:
        print(f"[EXCEPTION] Do the pixelwise classification throws exception {ex}")
        print(traceback.print_exc())

def define_the_target_raster_v2(dictionary_of_combination, configuration, dataset_path):
    try:
        target_raster_info = {}
        target_raster_info['target_raster_min_x'] = 0
        target_raster_info['target_raster_max_x'] = 0
        target_raster_info['target_raster_min_y'] = 0
        target_raster_info['target_raster_max_y'] = 0
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

        ctn = 0

        for channel in configuration['channel_list']:
            for set_of_input in dictionary_of_combination[channel]:
                print(f'[WALK] I m walking on {channel} with input {set_of_input}')

                # load the data with gdal
                if channel != "channel_3":
                    path_to_image = f"{dataset_path}/{set_of_input}.tif"
                    tiff_data = gdal.Open(f"{path_to_image}", gdal.GA_ReadOnly)
                else:
                    path_to_image = '/media/luca/My Passport' + f"/{set_of_input}.tif"
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

                if ctn == 0:
                    target_raster_info['target_raster_min_x'] = minx
                    target_raster_info['target_raster_max_x'] = maxx
                    target_raster_info['target_raster_min_y'] = miny
                    target_raster_info['target_raster_max_y'] = maxy
                    target_raster_info['number_of_x_pixels'] = x_ext
                    target_raster_info['number_of_y_pixels'] = y_ext


                    if pixelwise_classification_options["OVERRIDE_RASTER_SIZE"] != None:
                        target_raster_info['target_raster_spatial_x_pixel_size_meters'] = pixelwise_classification_options["OVERRIDE_RASTER_SIZE"]
                        target_raster_info['target_raster_spatial_y_pixel_size_meters'] = pixelwise_classification_options["OVERRIDE_RASTER_SIZE"]
                    else:
                        target_raster_info['target_raster_spatial_x_pixel_size_meters'] = np.abs(pix_x)
                        target_raster_info['target_raster_spatial_y_pixel_size_meters'] = np.abs(pix_x)
                    ctn+= 1

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
                target_raster_info['raster_width'] = target_raster_info['target_raster_max_x'] - target_raster_info['target_raster_min_x']
                target_raster_info['raster_height'] = target_raster_info['target_raster_max_y'] - target_raster_info['target_raster_min_y']

                target_raster_info['number_of_x_pixels'] = int(target_raster_info['raster_width'] / float(target_raster_info['target_raster_spatial_x_pixel_size_meters']))
                target_raster_info['number_of_y_pixels'] = int(target_raster_info['raster_height'] / float(target_raster_info['target_raster_spatial_y_pixel_size_meters']))


        return target_raster_info

    except Exception as ex:
        print(f"Define the target raster throws exception {ex}")

def define_the_target_raster_zone(path_to_zone):
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

        tiff_data = gdal.Open(f"{path_to_zone}", gdal.GA_ReadOnly)

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

        print(f"[PROFILE] has profile:\n"
              f"[min_x] {minx}\n"
              f"[max_y] {maxy}\n"
              f"[pix_x] {pix_x}\n"
              f"[pix_y] {pix_y}\n"
              f"[x_ext] {x_ext}\n"
              f"[y_ext] {y_ext}\n"
              f"[max_x] {maxx}\n"
              f"[min_y] {miny}\n")

        # load the georef into a dict
        target_raster_info['target_raster_min_x'] = minx
        target_raster_info['target_raster_max_x'] = maxx
        target_raster_info['target_raster_min_y'] = miny
        target_raster_info['target_raster_max_y'] = maxy
        target_raster_info['number_of_x_pixels'] = x_ext
        target_raster_info['number_of_y_pixels'] = y_ext
        target_raster_info['target_raster_spatial_x_pixel_size_meters'] = np.abs(pix_x)
        target_raster_info['target_raster_spatial_y_pixel_size_meters'] = np.abs(pix_x)

        # get the size of the pixels
        target_raster_info['number_of_x_pixels'] = round(
            target_raster_info['target_raster_max_x'] - target_raster_info['target_raster_min_x']) / float(
            target_raster_info['target_raster_spatial_x_pixel_size_meters'])
        target_raster_info['number_of_y_pixels'] = round(
            target_raster_info['target_raster_max_y'] - target_raster_info['target_raster_min_y']) / float(
            target_raster_info['target_raster_spatial_x_pixel_size_meters'])

        target_raster_info['raster_height'] = round(
            target_raster_info['target_raster_max_y'] - target_raster_info['target_raster_min_y'])
        target_raster_info['raster_width'] = round(
            target_raster_info['target_raster_max_x'] - target_raster_info['target_raster_min_x'])

        target_raster_info['number_of_x_pixels'] = int(target_raster_info['number_of_x_pixels'])
        target_raster_info['number_of_y_pixels'] = int(target_raster_info['number_of_y_pixels'])

        return target_raster_info

    except Exception as ex:
        print(f"[EXCEPTION] Define the target raster throws exception {ex}")

def make_the_pixelwise_classification_v2(best_combination, configuration, target_raster, weights_folder, save_folder,
                                         worker_number, dataset_path, nfi_dataset_path=f"/media/luca/Luca",
                                         override_height=None, override_width=None, from_which_row=0, from_which_col=0):
    try:
        stored_input = {}

        # load the file once
        for channel in configuration["channel_list"]:
            temp = list()

            for path in best_combination[channel]:
                if channel != "channel_3":
                    path_to_image =f"{dataset_path}/{path}.tif"
                else:
                    path_to_image = f"{nfi_dataset_path}/{path}.tif"

                temp.append(rasterio.open(path_to_image))

            # stored the file we need in a dict so we do not need to load it each time
            stored_input[channel] = temp

        # we need the specific colors
        class_id_list = list(colors.keys())

        # load the model
        model = tf.keras.models.load_model(weights_folder)
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        # sliding windows start here
        placeholder = np.zeros((override_height, override_width))

        for R in range(from_which_row, override_height):
            if R % 5 == 0:
                print(f"[SAVE] Saving")
                np.save(f"{save_folder}/slice_{worker_number}", placeholder)

            # slide for each pixel in col
            for C in range(from_which_col, override_width):
                data_to_predict = {}
                # let s start with the channels:
                for channel in configuration["channel_list"]:
                    # we have to stack the windows from the same inputs
                    concatenated = np.zeros((configuration[channel], configuration[channel], 0))
                    for ctn, obj in enumerate(stored_input[channel]):

                        # create the windows
                        window = yield_the_sliding_windows(tif=obj,
                                                           E=target_raster["list_of_E"][worker_number][0] + (C + configuration["OFFSET"]) * target_raster["target_raster_spatial_y_pixel_size_meters"],
                                                           N=target_raster["list_of_N"][worker_number][0] - (R + configuration["OFFSET"]) * target_raster["target_raster_spatial_y_pixel_size_meters"],
                                                           window_dimension=configuration[channel],
                                                           raster_size=target_raster["target_raster_spatial_y_pixel_size_meters"],
                                                           current_channel=channel)

                        if len(best_combination[channel]) > 1:
                            concatenated = np.concatenate((concatenated, window), axis=2)
                        else:
                            concatenated = window

                        if concatenated.shape[0] != configuration[channel] or concatenated.shape[1] != configuration[channel]:
                            print(f"[DIMENSION] Windows dimension for channel {channel} does not match")
                            continue

                    #print(f"[WINDOWS DIMENSION] R -> {R} C -> {C} -> {channel} -- {concatenated.shape}")
                    # here the data is ready to be predicted
                    data_to_predict[f"input_{channel}"] = np.expand_dims(concatenated, axis=0)
                #print(data_to_predict)
                # predict
                prediction = model.predict(data_to_predict, verbose=configuration["VERBOSE"])
                prediction = np.squeeze(prediction)
                prediction = np.argmax(prediction, axis=-1)

                if configuration["STDOUT"]:
                    print(f"[PREDICTION] Worker -> {worker_number} Row -> {R} Col -> {C} is {class_id_list[prediction]}")
                # save each pixel in the placeholder
                placeholder[R, C] = class_id_list[prediction]
        # save
        print(f"[SAVE] Saving")
        np.save(f"{save_folder}/slice_{worker_number}", placeholder)
        # end


    except Exception as ex:
        print(f"[EXCEPTION] Do the pixelwise classification throws exception {ex}")
        print(traceback.print_exc())


def create_slices_for_the_workers_v2(target_raster, number_of_worker):
    try:
        temp_counter = 0
        temp_N, temp_E = list(), list()

        maxy = target_raster['target_raster_max_y']
        miny = target_raster['target_raster_min_y']
        minx = target_raster['target_raster_min_x']
        maxx = target_raster['target_raster_max_x']

        distance = maxy - miny
        slices = maxy
        worker = distance / number_of_worker

        # temp_N.append([maxy, slices - worker])
        temp_E.append([minx, maxx])

        # add the other eleven parts
        start_N = maxy
        end_N = slices - worker

        while slices > miny:
            temp_N.append([start_N, end_N])
            temp_E.append([minx, maxx])
            slices -= worker
            start_N = end_N
            end_N = slices - worker

            # add counter we want all the list same height
            temp_counter += 1
            # break if this occours
            if temp_counter == number_of_worker:
                break

        return temp_N, temp_E
    except Exception as ex:
        print(f"create slices for the workers {ex}")


def create_slices_for_the_workers_v3(target_raster, number_of_worker):
    try:
        temp_counter = 0
        temp_N, temp_E = list(), list()

        maxy = target_raster['target_raster_max_y']
        miny = target_raster['target_raster_min_y']
        minx = target_raster['target_raster_min_x']
        maxx = target_raster['target_raster_max_x']

        y_wise_distance = maxy - miny
        slices = maxy
        y_wise_worker = y_wise_distance / number_of_worker

        # add the other eleven parts
        start_N = maxy
        end_N = slices - y_wise_worker

        while slices > miny:
            temp_N.append([start_N, end_N])
            slices -= y_wise_worker
            start_N = end_N
            end_N = slices - y_wise_worker

            # add counter we want all the list same height
            temp_counter += 1
            # break if this occours
            if temp_counter == number_of_worker:
                break

        # reset the counter for the x wise
        temp_counter = 0
        slices = minx
        x_wise_worker = (maxx - minx) / number_of_worker
        start_E = minx
        end_E = slices + x_wise_worker

        while slices < maxx:
            temp_E.append([start_E, end_E])
            slices += x_wise_worker
            start_E = end_E
            end_E = slices + x_wise_worker

            temp_counter += 1

            # break if this occours
            if temp_counter == number_of_worker:
                break

        return temp_N, temp_E
    except Exception as ex:
        print(f"create slices for the workers {ex}")

def yield_the_sliding_windows(tif, E, N, window_dimension, raster_size, current_channel):
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
        # create the window
        window = np.array(values_with_need).astype('uint8').reshape((window_dimension, window_dimension, -1))

        # remove the famous no value
        window[window == -9999.0] = -1
        window[window == -9999] = -1
        window[window == 32767] = -1
        window[window == 32767.0] = -1

        # normalize
        window = window / 255.0

        return window
    except Exception as ex:
        print(f"Yield the sliding windows throws exception {ex}")
        print(traceback.print_exc())
"""
"""
FROM HERE V2
"""





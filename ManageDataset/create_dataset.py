import sys
import os

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Configuration.configuration import *
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import gdal, osr
import rasterio


# set pd to show full cols
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


def generate_windows(dataset_path,
                     satellite_name,
                     path_to_the_image,
                     annotation,
                     channel,
                     windows_size,
                     filename,
                     no_data,
                     no_data_value,
                     override_channel,
                     datatype,
                     write_annotation_summary):
    try:
        # create path if necessary
        saving_numpy = os.path.join(dataset_path, f"{satellite_name}_{channel}", f"{filename}")
        saving_csv = os.path.join(dataset_path, "summary", f"{filename}.csv")

        if not os.path.exists(saving_numpy):
            os.makedirs(saving_numpy)

        # open ann
        shapefile = open(annotation, 'r')
        for ctn, line in enumerate(shapefile.readlines()):

            # jump header
            if ctn == 0:
                continue

            # preprocess the data
            line = line.strip()
            E = float(line.split(',')[0])
            N = float(line.split(',')[1])
            class_id = int(line.split(',')[2])

            print(f"E -> {E} - N -> {N}-> suotyypi -> {class_id}")

            # open the image with gdal
            tiff_data = gdal.Open(path_to_the_image, gdal.GA_ReadOnly)

            # get the tif profile
            geotransform = tiff_data.GetGeoTransform()
            raster_size = geotransform[1]

            print(f"Raster size {raster_size}")

            # I need an empty placeholder to temporary save the windows
            tiff_data = rasterio.open(path_to_the_image)
            # NOT OVERRIDE
            if not override_channel:
                if channel == "channel_1":
                    E = E - 20
                    N = N + 20
                elif channel == "channel_2":
                    E = E - 24
                    N = N + 24
                elif channel == "channel_3":
                    E = E - 32
                    N = N + 32
                elif channel == "channel_4":
                    E = E - 50
                    N = N + 50
                else:
                    E = E - 1
                    N = N + 1
            else:
                # OVERRIDE
                if channel == "channel_3":
                    E = E - 32
                    N = N + 32
                elif channel == "channel_5":
                    E = E - 1
                    N = N + 1
                else:
                    E = E - 20
                    N = N + 20

            coordinates = []
            for y in range(0, windows_size):
                N_point = N - y * raster_size
                for x in range(0, windows_size):
                    E_point = E + x * raster_size
                    coordinates.append((E_point, N_point))

            values_with_need = [tiff_val for tiff_val in tiff_data.sample(coordinates)]

            if write_annotation_summary:
                string_to_write = ""
                c = ""
                with open(saving_csv, 'a') as handler:
                    for values, coord in zip(values_with_need, coordinates):
                        c = f"{coord[0]},{coord[1]}"
                        for v in values:
                            # print(coord)
                            string_to_write = string_to_write + str(v) + ','
                        string_to_write = f"{string_to_write}{c},{class_id},{ctn}\n"
                        handler.writelines(string_to_write)
                        string_to_write = ""

            # create the window
            window = np.array(values_with_need).astype(datatype).reshape((windows_size, windows_size, -1))

            # assign no data value to no data
            window[window == no_data] = no_data_value

            print(f"Check the final shape -> {window.shape}")
            np.save(os.path.join(saving_numpy, f"w_{class_id}_{ctn}"), window)

        shapefile.close()
    except Exception as ex:
        print(f"[EXCEPTION] Generate the windows throws exception {ex}")


if __name__ == "__main__":
    try:

        # add arg parse is better than previous
        parser = argparse.ArgumentParser()

        parser.add_argument("-area", 
                            help="The area to make the classification", 
                            required=True)
                
        parser.add_argument("-path_list", 
                            help="the path to the file where to read the path of the window", 
                            required=True)
        
        parser.add_argument("-override_channel", 
                            help="ovveride channel size", 
                            required=False)

        # parse
        args = parser.parse_args()

        if args.override_channel == "YES":
            override = True
            config["channel_1"] = 5
            config["channel_2"] = 5
            config["channel_3"] = 5
            config["channel_4"] = 5
            config["channel_5"] = 1
        else:
            override = False

        # open the config file
        handler = open(f'{args.path_list}', 'r')
        for file_to_read in handler.readlines():
            folder_path = file_to_read.strip().split(',')[0]
            channel = file_to_read.strip().split(',')[1]
            no_data = file_to_read.strip().split(',')[2]
            no_data_value = file_to_read.strip().split(',')[3]

            # set up the running
            # ann path
            annotation_undrained = area_manager[args.area.upper()]["ANNOTATIONS"]["UNDRAINED"]
            annotation_drained = area_manager[args.area.upper()]["ANNOTATIONS"]["DRAINED"]

            # ds save path
            undrained_dataset_path  = area_manager[args.area.upper()]["DATASET_PATH"]["UNDRAINED"]
            drained_dataset_path =  area_manager[args.area.upper()]["DATASET_PATH"]["DRAINED"]

            # the size is the same
            windows_size = config[channel]

            # loop into all the images of that folder

            for images in os.listdir(folder_path):
                filename, extension = os.path.splitext(images)
                satellite_name = folder_path.split('/')[-1]
                PATH_OF_THE_IMAGE = os.path.join(folder_path, images)

                # set dt
                if filename.split('_')[0] == "derived":
                    datatype = 'float'
                else:
                    datatype = 'uint8'

                # exclude if the image is not a tif file
                if extension != '.tif' and extension != '.img':
                    print(f"[INFO] {filename} is not an image")
                    continue

                # generate undrained windows
                generate_windows(dataset_path=undrained_dataset_path,
                                 satellite_name=satellite_name,
                                 path_to_the_image=PATH_OF_THE_IMAGE,
                                 annotation=annotation_undrained,
                                 channel=channel,
                                 windows_size=windows_size,
                                 filename=filename.split('/')[-1],
                                 no_data=no_data,
                                 no_data_value=no_data_value,
                                 override_channel=override,
                                 write_annotation_summary=config["WRITE_ANNOTATION_SUMMARY"],
                                 datatype=datatype)
                
                # generate drained windows
                generate_windows(dataset_path=drained_dataset_path, 
                                 annotation=annotation_drained,
                                 satellite_name=satellite_name,
                                 path_to_the_image=PATH_OF_THE_IMAGE,
                                 channel=channel,
                                 windows_size=windows_size,
                                 filename=filename.split('/')[-1],
                                 no_data=no_data,
                                 no_data_value=no_data_value,
                                 override_channel=override,
                                 write_annotation_summary=config["WRITE_ANNOTATION_SUMMARY"],
                                 datatype=datatype)

        handler.close()

    except Exception as ex:
        print(f"[EXCEPTION] Main throws {ex}")




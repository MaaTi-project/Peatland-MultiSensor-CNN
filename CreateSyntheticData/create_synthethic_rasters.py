from osgeo import gdal, osr
import os
import numpy as np
import random

def saveRasterToTiffFile_with_multiples_bands(arr, filename, geotrans, nodata_value, driver_type=gdal.GDT_Byte):
    [channel, cols, rows] = arr.shape
    print(arr.shape)
    driver = gdal.GetDriverByName("GTiff")
    outdata = None
    outdata = driver.Create(filename, rows, cols, channel, driver_type)
    outdata.SetGeoTransform(geotrans)
    # sets same geotransform as input

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    ds = srs.ExportToWkt()

    outdata.SetProjection(ds)

    # sets same projection as input
    for i in range(channel):
        outdata.GetRasterBand(i + 1).WriteArray(arr[i, :, :])
        outdata.GetRasterBand(i + 1).SetNoDataValue(nodata_value)

    outdata.FlushCache()
    # saves to disk!!
    outdata = None
    band=None
    ds=None


def create_gaussian_array_data(dictionary_of_coordinates, mean, std_deviation):
    bands = random.randint(1, 10)
    raster_shape = (bands, dictionary_of_coordinates["height"], dictionary_of_coordinates["width"])
    array_data = np.random.normal(mean, std_deviation, raster_shape)
    return array_data


def create_random_uniform_array_data(dictionary_of_coordinates, lowerbound, upperbound):
    bands = random.randint(1, 10)
    raster_shape = (bands, dictionary_of_coordinates["height"], dictionary_of_coordinates["width"])
    array_data = np.random.uniform(lowerbound, upperbound, size=raster_shape)
    return array_data


def create_dataset_summary(path):
    handler = open(path, 'w')

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

    number_optical_inputs = 6
    number_forest_inputs = 4
    number_of_derived = 3

    dictionary_of_coordinates = {
        "minx": 393200.0,
        "maxy": 7324390.0,
        "pix_x": 10.0,
        "pix_y": -10.0,
        "height": 2974,
        "width": 2960
    }

    # shape of generated raster
    raster_shape = (1, dictionary_of_coordinates["height"], dictionary_of_coordinates["width"])

    # geotransform
    geotrans_info = (dictionary_of_coordinates['minx'], 10, 0.0, dictionary_of_coordinates['maxy'], 0.0, -10)

    # create random no data map
    random_no_data_map = np.random.choice([0, 1, 2], size=raster_shape)

    saveRasterToTiffFile_with_multiples_bands(arr=random_no_data_map,
                                            filename=os.path.join("..", "PixelWiseClassification", "no_data", "no_data.img"),
                                            geotrans=geotrans_info,
                                            nodata_value=255,
                                            driver_type=gdal.GDT_Byte)

    # generate other rasters
    # random rasters
    #channel_one_and_three_arrays = np.random.normal(0, 256, size=raster_shape, dtype='uint8')

    # number of input here
    for input_ctn in range(number_optical_inputs):

        # NB -> here is 10
        geotrans_info = (dictionary_of_coordinates['minx'], 10, 0.0, dictionary_of_coordinates['maxy'], 0.0, -10)

        path = os.path.join("..", "Datasets", "TestArea", "Optical")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


        array_data = create_gaussian_array_data(dictionary_of_coordinates=dictionary_of_coordinates,
                                                mean=0,
                                                std_deviation=1)

        saveRasterToTiffFile_with_multiples_bands(arr=array_data,
                                                filename=os.path.join(path, f"optical_{input_ctn}.tif"),
                                                geotrans=geotrans_info,
                                                nodata_value=255,
                                                driver_type=gdal.GDT_Byte)

    # number of input here
    for input_ctn in range(number_forest_inputs):

        # NB -> here is 16
        geotrans_info = (dictionary_of_coordinates['minx'], 10, 0.0, dictionary_of_coordinates['maxy'], 0.0, -10)

        path = os.path.join("..", "Datasets", "TestArea", "Forest")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        array_data = create_gaussian_array_data(dictionary_of_coordinates=dictionary_of_coordinates,
                                                mean=0,
                                                std_deviation=1)

        saveRasterToTiffFile_with_multiples_bands(arr=array_data,
                                                filename=os.path.join(path, f"forest_{input_ctn}.tif"),
                                                geotrans=geotrans_info,
                                                nodata_value=255,
                                                driver_type=gdal.GDT_Byte)

    # number of input here
    for input_ctn in range(number_forest_inputs):

        # NB -> here is 16
        geotrans_info = (dictionary_of_coordinates['minx'], 10, 0.0, dictionary_of_coordinates['maxy'], 0.0, -10)

        path = os.path.join("..", "Datasets", "TestArea", "Derived")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        
        array_data = create_random_uniform_array_data(dictionary_of_coordinates, 
                                                      lowerbound=0, 
                                                      upperbound=1)

        saveRasterToTiffFile_with_multiples_bands(arr=array_data,
                                                filename=os.path.join(path, f"derived_{input_ctn}.tif"),
                                                geotrans=geotrans_info,
                                                nodata_value=255,
                                                driver_type=gdal.GDT_Float32)
    
    
    if not os.path.exists(os.path.join("..", "Configuration", "Masterfiles")):
        os.makedirs(os.path.join("..", "Configuration", "Masterfiles"))
    
    # create a masterfile directly 
    create_dataset_summary(path=os.path.join("..", "Configuration", "Masterfiles", f"dataset_test_area.csv"))


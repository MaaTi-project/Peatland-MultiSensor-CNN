import json
import os
from osgeo import gdal, osr
import numpy as np
from Functions.geo_raster_functions import openTiffFileAsRaster, saveRasterToTiffFile


output_folder = os.path.join("final_arrays")

dictionary_of_coordinates = {
    "minx": 393200.0,
    "maxy": 7324390.0,
    "pix_x": 10.0,
    "pix_y": -10.0,
    "height": 2974,
    "width": 2960
}

if __name__ == '__main__':
    try:
        
        raster_nodata_value = 255
        raster_output_type = "classification"
        
        path_undrained = os.path.join("saved_arrays", f"raster_undrained.npy")
        path_drained = os.path.join("saved_arrays", f"raster_drained.npy")

        geotrans_info = (dictionary_of_coordinates['minx'], dictionary_of_coordinates['pix_x'], 0.0, dictionary_of_coordinates['maxy'], 0.0, dictionary_of_coordinates['pix_y'])

        print(f'[GEO-TRANSFORM] The geo transform is {geotrans_info}')

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        projection_referencdse_info_ds = srs.ExportToWkt()
        
        undrained = False

        for i in range(2):
            if undrained:
                undrained = False
                # load the two array
                undrained_stitched_prediction_raster = np.load(path_undrained)
                # get filename from the original array
                undrained_output_raster_filename = os.path.join(output_folder, f"final_raster_undrained.tif")

                # undrained
                saveRasterToTiffFile(undrained_stitched_prediction_raster.astype('uint8'), projection_referencdse_info_ds,
                                        undrained_output_raster_filename, geotrans_info, raster_output_type, raster_nodata_value)

            else:
                drained_stitched_prediction_raster = np.load(path_drained)
                drained_output_raster_filename = os.path.join(output_folder, f"final_raster_drained.tif")

                # drained
                saveRasterToTiffFile(drained_stitched_prediction_raster.astype('uint8'), projection_referencdse_info_ds,
                                        drained_output_raster_filename, geotrans_info, raster_output_type, raster_nodata_value)

    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception  {ex}")
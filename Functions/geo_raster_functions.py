from osgeo import gdal
import numpy as np
import sys
from Configuration.configuration import *
from Configuration.best_combination import *
from Functions.pixelwise_classification_utilities import *
import argparse

def openTiffFileAsRaster(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def saveRasterToTiffFile(arr, ds, filename, geotrans, raster_output_type, nodata_value, driver_type=gdal.GDT_Byte):
    [cols, rows] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = None
    if raster_output_type == "classification":
        outdata = driver.Create(filename, rows, cols, 1, driver_type)
    else: # Regression on default.
        outdata = driver.Create(filename, rows, cols, 1, gdal.GDT_Float32)
    # outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetGeoTransform(geotrans)
    # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())
    # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr)
    outdata.GetRasterBand(1).SetNoDataValue(nodata_value)
    # if you want these values transparent
    outdata.FlushCache()
    # saves to disk!!
    outdata = None
    band=None
    ds=None

def regenerate_the_raster_from_worker(row_worker, col_worker, type, path):
    try:
        col_counter = 0
        attach_col_wise = None
        for row in range(row_worker):
            attach_row_wise = None
            row_counter = 0

            for col in range(col_worker):
                if row_counter == 0:
                    attach_row_wise = np.load(f"{path}/worker_{row}/worker_{row}_{col}_{type}.npy")
                    row_counter += 1
                else:
                    attach_row_wise = np.concatenate((attach_row_wise,
                                                      np.load(f"{path}/worker_{row}/worker_{row}_{col}_{type}.npy")), axis=1)

            print(f'[ROW] Row complete {attach_row_wise.shape}')

            if col_counter == 0:
                attach_col_wise = attach_row_wise
                col_counter += 1
            else:
                attach_col_wise = np.concatenate((attach_col_wise, attach_row_wise), axis=0)

        print(f'[FINAL SHAPE] The raster final shape is {attach_col_wise.shape}')
        return attach_col_wise

    except Exception as ex:
        print(f"[EXCEPTION] Prepare the ratser from worker throws exception {ex}")
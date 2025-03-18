#!/usr/bin/env python

from Functions.functions import *
import argparse

if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--geo', required=False, help="geo reference input")
        parser.add_argument('--numpy', required=False, help="input matrix")
        parser.add_argument('--output', required=False, help="output image")

        args = parser.parse_args()

        if args.geo:
            some_input_geotiff_filename = args.geo
        else:
            some_input_geotiff_filename = "../../Csc/input/Sentinel2/S2A_MSIL2A_20180526T100031_N0208_R122_T34WFU_20180526T140955_Keminmaa.tif"

        if args.numpy:
            your_numpy_raster = np.load(args.numpy).astype('uint8')
        else:
            #your_numpy_raster = np.load(f'../Results/test_one_input/prediction_keminmaa_class_id__model_a_jukka_has_to_see.npy').astype('uint8')
            raise Exception(f"Input raster is wrong")
        if args.output:
            your_output_geotiff_filename = args.output
        else:
            your_output_geotiff_filename = "../Results/test_one_input/sanity_check.tif"

        print(f'We are in {os.getcwd()}')

        # open the tif file to get the profile
        with rasterio.open(some_input_geotiff_filename) as src:

            # get the metadata
            target_raster_min_E = src.profile['transform'][2]
            target_raster_max_N = src.profile['transform'][5]
            pixel_spatial_resolution = 10
            print(target_raster_min_E)
            print(target_raster_max_N)
            print(pixel_spatial_resolution )

        ###
        ### DO NOT CHANGE THE FOLLOWING CODE
        ###

        not_needed_dem_arr, projection_info_ds = openTiffFileAsRaster(some_input_geotiff_filename)  # Take projection info from some available source tiff you have. Here I have used Keminmaa_DEM
        geotrans_info = (target_raster_min_E, pixel_spatial_resolution, 0.0, target_raster_max_N, 0.0, -pixel_spatial_resolution)  # Make the geotrans_object --> combines the details you set above

        print(geotrans_info)

        # Save your Numpy raster to GeoTiff
        saveRasterToTiffFile(your_numpy_raster, projection_info_ds, your_output_geotiff_filename, geotrans_info)

    except Exception as ex:
        print(f'Main throws exception {ex}')
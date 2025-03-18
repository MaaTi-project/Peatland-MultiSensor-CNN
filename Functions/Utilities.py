import itertools
import math
import shutil
import re
import traceback
import os
import numpy as np
import geopandas as gpd
import rasterio
from sklearn.preprocessing import  OneHotEncoder
from matplotlib import pyplot as plt
from scipy.ndimage import *
from osgeo import gdal, osr, ogr


colors = {
    11: (255,0,0),
    12: (0,255,0),
    13: (188,143,143),
    14: (0,0,255),
    21: (255,255,0),
    22: (0,255,255),
    31: (255,0,255),
    32: (192,192,192),
    33: (128,128,128),
    41: (128,0,0),
    42: (128,128,0),
    43: (0,128,0),
    44: (128,0,128),
    45: (0,128,128),
    46: (0,0,128),
    47: (255,165,0),
    48: (176,196,222),
    51: (238,232,170),
    52: (154,205,50),
    53: (175,238,238),
    54: (65,105,225),
    55: (138,43,226),
    56: (221,160,221),
    61: (255,105,180),
    62: (255,250,205),
    63:	(240,255,255),
    64: (205,133,63),
    65: (178,34,34),
    71: (255,228,225),
    72: (245,255,250),
    81: (230,230,250),
    82: (240,248,255),
    83: (255,20,147),
    91: (105,105,105),
    101: (30,144,255),
    102: (220,220,220),
    103: (127,255,212),
    120: (34,139,34),
    130: (220,20,60)
}


def plot_accuracy(history, epochs, save_path, title):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), accuracy, label="train")
    plt.plot(range(epochs), val_accuracy, label="val train")
    plt.title(f'{title} model accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), loss, label="loss")
    plt.plot(range(epochs), val_loss, label="val los")
    plt.title(f'{title} model loss')
    plt.savefig(f'{save_path}')


def create_batch_points(annotations):
    try:
        # first step is to load the shapefile  and show it
        shapefile = gpd.read_file(annotations)

        # get suotyypiID and pointz
        suotyypi = shapefile["SuotyypId"]

        # get the suotyppi in array
        suotyypi = np.array(suotyypi)

        # loop through all points and transalte it in normal coordinates
        batch_points = []
        # extract each points
        geometry = [i for i in shapefile.geometry]
        for ctn, i in enumerate(range(len(geometry))):
            x, y = geometry[i].coords.xy
            batch_points.append([float(np.array(x)[0]), float(np.array(y)[0]), int(suotyypi[ctn])])
        return batch_points
    except Exception as ex:
        print(f"[EXCEPTION] create batch points throws exceptions {ex}")


def get_unique_labels_from_new_shapefile(shapefile, is_suotyypi=True):
    try:
        labels = []

        if is_suotyypi is True:
            ids = shapefile['SuotyypId']
        else:
            ids = shapefile["RavintLk"]

        for s in ids:
            if s not in labels:
                labels.append(s)
        return labels
    except Exception as ex:
        print("Get unique values throws exceptions " + str(ex))


def prepare_the_data_fast(path, rotation_grade=None):

    images = list()
    labels = list()
    try:
        for syotyypi in os.listdir(path):
            full_path = path + '/' + syotyypi
            for windows in os.listdir(full_path):
                if rotation_grade is not None:
                    win = rotate(input=np.load(full_path + '/' + windows), angle=float(rotation_grade), reshape=False)
                    images.append(win)
                else:
                    images.append(np.load(full_path + '/' + windows))
                labels.append(syotyypi)

        #  prepare the data
        images = np.array(images, dtype='float32')
        images = images / 255.0
        labels = np.array(labels, dtype='int')

        print(f"[LOAD COMPLETE] Load complete images size {images.shape} -> labels size {labels.shape}")
        return images, labels

    except Exception as ex:
        print(f"[EXCEPTION] Prepare the data fast throws exception -> {ex}")


def make_one_hot_encoding(labels_to_transform):
    try:
        # to categorical
        enc = OneHotEncoder()
        # train and valid set
        temp = np.reshape(labels_to_transform, (-1, 1))
        label_transformed = enc.fit_transform(temp).toarray()
        # check the one hot encoding
        print(f'[ONE HOT ENCODING]Labels are one-hot-encoded: {(label_transformed.sum(axis=1) - np.ones(label_transformed.shape[0])).sum() == 0}')
        return label_transformed
    except Exception as ex:
        print(f"Make one hot encoding throws exception  {ex}")


def get_tiff_data_at_coord_locations(input_tiff_path, coords):
    #print(input_tiff_path)
    tiff_data = gdal.Open(input_tiff_path, gdal.GA_ReadOnly) # luetaan tiff-data
    #tiff_proj = osr.SpatialReference(tiff_data.GetProjection()).GetAttrValue("PROJCS",0) # karttaprojektion nimi (jos tarvitsee tarkistaa)
    #tiff_epsg = int(osr.SpatialReference(tiff_data.GetProjection()).GetAttrValue("AUTHORITY",1)) # karttaprojektion epsg-koodi (jos tarvitaan esim. uudelleenprojisoint
    geotransform = tiff_data.GetGeoTransform()
    minx = geotransform[0] # x-koordinaatin minimi (lansi)
    maxy = geotransform[3] # y-koordinaatin maksimi (pohjoinen)
    pix_x = geotransform[1] # pikselikoko x-suunnassa; positiivinen (kasvaa lanteen)
    pix_y = geotransform[5] # pikselikoko y-suunnassa; negatiivinen (pienenee etelaan)
    x_ext = tiff_data.RasterXSize # rasterin koko (pikselia) x-suunnassa
    y_ext = tiff_data.RasterYSize # rasterin koko (pikselia) y-suunnassa
    maxx = minx + pix_x * x_ext # x-koordinaatin maksimi (ita)
    miny = maxy + pix_y * y_ext # y-koordinaatin minimi (etela)
    #tiff_data = None  # suljetaan rasteri
    tiff_data = rasterio.open(input_tiff_path)
    #for tiff_val in tiff_data.sample(coords):
    #    print("Reading ", input_tiff_path, tiff_val)
        #time.sleep(1)
    # tiff_arvot = [tiff_val[0] for tiff_val in tiff_data.sample(coords)]
    tiff_arvot = [tiff_val for tiff_val in tiff_data.sample(coords)]
    tiff_data = None # suljetaan rasteri
    tiff_info = {}
    tiff_info['x_min_boundary_coord'] = minx
    tiff_info['x_max_boundary_coord'] = maxx
    tiff_info['y_min_boundary_coord'] = miny
    tiff_info['y_max_boundary_coord'] = maxy
    tiff_info['number_of_x_pixels'] = x_ext
    tiff_info['number_of_y_pixels'] = y_ext
    tiff_info['spatial_pixel_size_meters_x'] = np.abs(pix_x)
    tiff_info['spatial_pixel_size_meters_y'] = np.abs(pix_x)
    tiff_info['pixel_values_at_coords'] = tiff_arvot
    return tiff_info


def get_the_geoinfo(input_tiff_path):
    tiff_data = gdal.Open(input_tiff_path, gdal.GA_ReadOnly) # luetaan tiff-data
    #tiff_proj = osr.SpatialReference(tiff_data.GetProjection()).GetAttrValue("PROJCS",0) # karttaprojektion nimi (jos tarvitsee tarkistaa)
    #tiff_epsg = int(osr.SpatialReference(tiff_data.GetProjection()).GetAttrValue("AUTHORITY",1)) # karttaprojektion epsg-koodi (jos tarvitaan esim. uudelleenprojisoint
    geotransform = tiff_data.GetGeoTransform()
    minx = geotransform[0] # x-koordinaatin minimi (lansi)
    maxy = geotransform[3] # y-koordinaatin maksimi (pohjoinen)
    pix_x = geotransform[1] # pikselikoko x-suunnassa; positiivinen (kasvaa lanteen)
    pix_y = geotransform[5] # pikselikoko y-suunnassa; negatiivinen (pienenee etelaan)
    x_ext = tiff_data.RasterXSize # rasterin koko (pikselia) x-suunnassa
    y_ext = tiff_data.RasterYSize # rasterin koko (pikselia) y-suunnassa
    maxx = minx + pix_x * x_ext # x-koordinaatin maksimi (ita)
    miny = maxy + pix_y * y_ext # y-koordinaatin minimi (etela)
    DEBUG = False
    if DEBUG:
        print(f'[DATA] Left border {minx}')
        print(f'[DATA] Upper border {maxy}')
        print(f'[DATA] Delta {pix_x}')
    return minx, maxy, pix_x


def get_better_fold(path=os.path.join("..", "Results", "test_one_input", "weights")):
    try:
        acc_dict = {}
        fold = None
        print(f"[SELECT] Selecting best model for prediction")
        for file in os.listdir(path):
            if file.split('_')[4] == 'model.h5':
                acc_dict[file.split('_')[1]] = file.split('_')[3]
        values = acc_dict.values()
        max_accuracy_val = max(list(values))
        for key, val in acc_dict.items():
            if val == max_accuracy_val:
                fold = key
        filename = f"{path}fold_{fold}_acc_{max_accuracy_val}_model.h5"
        print(f"[GOT IT] The better fold is {filename}")
        return filename
    except Exception as ex:
        print(f"Get better folds throws exception {ex}")


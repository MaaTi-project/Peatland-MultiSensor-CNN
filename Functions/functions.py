import itertools
import math
import shutil
import re
import traceback
import os
import numpy as np
import tensorflow as tf
import geopandas as gpd
import rasterio
import pandas as pd
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.utils import class_weight, compute_class_weight
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

def color_picker(labels_id):
    try:
        return colors[labels_id]
    except Exception as ex:
        print("[EXCEPTION] I do not find the right colors for the ID assigned now white " + str(ex) + " found label " + str(labels_id))
        return (255,255,255)

def prepare_the_data_fast_channel_wise(path, rotation_grade=None, channel=0):
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
                    win = np.load(full_path + '/' + windows)[:, :, channel]
                    win = np.reshape(win, (win.shape[0], win.shape[1], 1))
                    images.append(win)
                labels.append(syotyypi)

        #  prepare the data
        images = np.array(images, dtype='uint8')
        images = images / 255.0
        labels = np.array(labels, dtype='int')

        print(f"[LOAD COMPLETE] Load complete images size {images.shape} -> labels size {labels.shape}")
        return images, labels

    except Exception as ex:
        print(f"[EXCEPTION] Prepare the data fast throws exception -> {ex}")

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


def prepare_the_data(path, isSingle=None, exclude=None):
    images = list()
    labels = list()
    try:
        # now here go inside the main folder
        for image in os.listdir(path):
            if exclude is not None:
                for win_to_exclude in exclude:
                    if win_to_exclude == image:
                        print(f"[EXCLUSION PATH]excluding -> {win_to_exclude}")
                        continue
            # prepare the path
            image_to_walk = path + '/' + image
            # if it is single image
            if isSingle is not None:
                if isSingle == image:
                    #print(f"Walking in {image_to_walk}")
                    for syotyypi in os.listdir(image_to_walk):
                        full_path = image_to_walk + '/' + syotyypi
                        # foreach win get the win and syotyypi
                        for windows in os.listdir(full_path):
                            images.append(np.load(full_path + '/' + windows))
                            labels.append(syotyypi)
                    break
                else:
                    continue
            else:
                #print(f"Walking in {image_to_walk}")
                # all image in that folder
                for syotyypi in os.listdir(image_to_walk):
                    full_path = image_to_walk + '/' + syotyypi
                    # foreach win get the win and syotyypi
                    for windows in os.listdir(full_path):
                        images.append(np.load(full_path + '/' + windows))
                        labels.append(syotyypi)

        #  prepare the data
        images = np.array(images, dtype='uint8')
        images = images / 255.0
        labels = np.array(labels, dtype='int')

        print(f"[TASK] Load complete images size {images.shape} -> labels size {labels.shape}")

        return images, labels
    except Exception as ex:
        print(f"[EXCEPTION] Preapre the dat throws exception {ex}")


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


def all_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    f1Score = f1_score(y_true, y_pred, average='weighted')
    print("Accuracy  : {}".format(accuracy))
    #print("Precision : {}".format(precision))
    #print("f1Score : {}".format(f1Score))
    return accuracy, precision, f1Score


def calculate_sw_and_cw(train_labels):
    try:
        # samples weights
        sample_weights = class_weight.compute_sample_weight('balanced', np.array(train_labels))

        # class weights
        y_int = np.argmax(train_labels, axis=1)
        cw = compute_class_weight('balanced', np.unique(y_int), y_int)
        d_class_weights = dict(enumerate(cw))

        return d_class_weights, sample_weights
    except Exception as ex:
        print(f"Calculate the sw and cw throws exception {ex}")


def get_the_shape(path):
    try:
        # now here go inside the main folder
        for image in os.listdir(path):
            image_to_walk = path + '/' + image
            for syotyypi in os.listdir(image_to_walk):
                full_path = image_to_walk + '/' + syotyypi
                # foreach win get the win and syotyypi
                for windows in os.listdir(full_path):
                    dummy =  np.load(full_path + '/' + windows)
                    return dummy
    except Exception as ex:
        print(f"preapre the dat throws exception {ex}")


def make_stack_rounds(the_best, the_other, dummy_shapes=(2080, 10, 10, 0), rotation_grade=None, list_of_exclusion=None):
    main = np.empty(dummy_shapes)
    label = list()
    try:
        for ctn, file in enumerate(the_best):

            dir_to_walk = file
            temp = list()

            if list_of_exclusion is not None and file in list_of_exclusion:
                print(f'[EXCLUSION] [{ctn}] -> {file}')
                continue

            for syotyypi in os.listdir(dir_to_walk):
                full_path = dir_to_walk + '/' + syotyypi
                for window in os.listdir(full_path):
                    if rotation_grade is not None:
                        win = rotate(input=np.load(full_path + '/' + window), angle=float(rotation_grade), reshape=False)
                        temp.append(win)

                    else:
                        temp.append(np.load(full_path + '/' + window))
                    label.append(syotyypi)

            temp = np.array(temp, dtype='uint8')
            temp = temp / 255.0

            # concatenate all other rounds
            main = np.concatenate((main, temp), axis=3)

        if the_other is not None:
            # load the image to analyze
            normal_train, normal_label = prepare_the_data_fast(path=the_other, rotation_grade=rotation_grade)
            stacked = np.concatenate((main, normal_train), axis=3)
            print(f"[INFO] After stacking we got -> train_folder -> {stacked.shape} -> label -> {normal_label.shape}")
            print("--------")
            return stacked, normal_label
        else:
            return main, label

    except Exception as ex:
        print(f"Make the rounds throws exception {ex}")


def get_the_best_features(best_feat_dict):
    try:
        # get the key and the values
        key_list = list(best_feat_dict.keys())
        key_value = list(best_feat_dict.values())
        # get the max accuracy value
        max_value = max(key_value)
        position = key_value.index(max_value)
        # print out the best feat
        print(f"[BEST] I found that {key_list[position]} has best accuracy ({max_value})")
        # return the values
        return key_list[position], max_value
    except Exception as ex:
        print(f"Get the best feature throws exception {ex}")


def get_all_the_images(dictionary_of_input):
    res = list()
    main_folder = dictionary_of_input.keys()
    for ctn, main_path in enumerate(main_folder):
        for path in os.listdir(main_path):
            path_to_load = main_path + '/' + path
            res.append(path_to_load)
    return res, len(res)


def resume_the_feature_selection(path, counter=-1):
    try:
        res = list()
        file_handler = open(path, 'r')
        current = 0
        for line in file_handler.readlines():

            if line.strip() == "\n" or line.strip() == "+":

                continue
            elif line.find('get the best ->') != -1:
                line = line.replace('get the best ->', '')
                # get index of with
                get_with_index = line.index('with')
                line = line[: get_with_index].strip()
            else:
                print(f"[RESUMING] Loading {line.strip()}")
                res.append(line.strip())
                current += 1

            if counter != -1 and current == counter:
                print("[LOADING] arrived to destination")
                break
        return res, current
    except IOError as io:
        print(f"[IO-ERROR] Error loading the file {io}")
    except Exception as ex:
        print(f"[ERROR] resume the feature selection throws exception {ex}")


def prepare_the_stacked_list(dict_of_input, input_type):
    try:
        res = list()
        for k, v in dict_of_input.items():
            if v == input_type:
                for file in os.listdir(k):
                    res.append(k + '/' + file)
        return res
    except Exception as ex:
        print(f"Prepare the stacked list throw exception {ex}")


def preare_the_labels_for_analysis(shape_annotation, is_csc=False, is_suotyypi=True):
    try:
        # get the path where the files are dumped
        dump_path = os.path.dirname(os.path.realpath(".."))
        if is_csc is True:
            dump_path = dump_path + "/matti"
        # first step is to load the shapefile  and show it
        shapefile = gpd.read_file(shape_annotation)

        if is_suotyypi is True:
            # now get the count of syoutyypi ids
            suotyypi_group = shapefile.groupby('SuotyypId').Suotyyppi.count()
            #suotyypi_group.to_csv(dump_path + '/Annotation/Shapefiles manipulations/Results/shape_files_stat.csv')

            # get suotyypiID and pointz
            suotyypi = shapefile["SuotyypId"]
            point_z = shapefile["geometry"]

            # prepare the payload with point and suo zipped
            payload = shapefile[["SuotyypId", "geometry"]]
        else:
            # now get the count of syoutyypi ids
            suotyypi_group = shapefile.groupby('RavintLk').RavintLk.count()
            #suotyypi_group.to_csv(dump_path + '/Annotation/Shapefiles manipulations/Results/shapefiles_pistet_ravintLk_stat.csv')

            # get ids
            # get suotyypiID and pointz
            suotyypi = shapefile["RavintLk"]
            point_z = shapefile["geometry"]

            # prepare the payload with point and suo zipped
            payload = shapefile[["RavintLk", "geometry"]]

        # create labels array
        labels = np.array(suotyypi)

        # get unique labels from a shapefile
        labels_id = get_unique_labels_from_new_shapefile(shapefile=shapefile, is_suotyypi=is_suotyypi)

        return labels, labels_id

    except Exception as ex:
        print(f" Initialize the system throws exception {ex}")
        print(traceback.print_exc())
        return None


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

def move_train_accuracy_from_each_folders(dict_of_path):
    try:
        get_the_path = list(dict_of_path.keys())
        handler = open('../Results/tif_invis.txt', 'a')
        for key in get_the_path:
            handler.writelines(f"\n----- {key} -----\n")
            print(f"We are in {key}")
            for path in os.listdir(key):
                print(f"{key}/{path}")
                handler.writelines(f"{key}/{path}\n")
        handler.close()

        p = '../Results/'
        for round_number in range(122):
            if os.path.exists(f'{p}{round_number}'):
                path = f'{p}{round_number}'
                for tif in os.listdir(path):
                    image_folder = path + '/' + tif
                    print(f'[INFO] Walking in {image_folder}')
                    # if os.path.exists(f'{image_folder}/total_accuracy.csv'):
                    # total_accuracy = f'{image_folder}/total_accuracy.csv'
                    # print(total_accuracy)
                    # shutil.move(total_accuracy, f'/home/luca/Documents/maati/Paavo/Results/second stage_normal/total_acc_valid/{round_number}_{tif}_valid_acc.csv')
                    for sub_image in os.listdir(image_folder):
                        if sub_image == 'total_accuracy.csv':
                            print(sub_image)
                            continue
                        else:

                            single_path = f'{image_folder}/{sub_image}/history.csv'
                            print(f'[FILE] File -> {single_path}')
                            if os.path.exists(f'{single_path}'):
                                print(single_path)
                                shutil.move(single_path,
                                            f'/home/luca/Documents/maati/Paavo/Results/fold_history/{round_number}_{tif}_{sub_image}_folds_history.csv')
    except Exception as ex:
        print(f'Move the train accuracy throws exception {ex}')


def load_single_image_utility(path):
    try:
        with rasterio.open(path) as handler:
            tif_image = handler.read()
            # move the axes
            tif_image = np.moveaxis(tif_image, 0, -1)
        return tif_image
    except Exception as ex:
        print(f"Load the image utility throws exception {ex}")

def get_windows_indexed(windows_to_analyse, labels_id, path):
    try:
        temp = np.zeros((windows_to_analyse.shape[0], windows_to_analyse.shape[1]))
        for x in range(0, windows_to_analyse.shape[0]):
            for y in range(0, windows_to_analyse.shape[1]):
                temp[x, y] = labels_id[windows_to_analyse[x, y]]
        np.savetxt(f'{path}', temp, fmt="%d")
    except Exception as ex:
        print(f'Get the windows indexed throws exception {ex}')


def print_model_arch(input_model):
    try:
        ctn = 0
        for l in input_model.layers:
            print(f'[LAYER INPUT][{ctn}]: {l.input}')
            print(f'[LAYER OUTPUT][{ctn}]: {l.output}')
            print('')
            ctn += 1
    except Exception as ex:
        print(f'[EXCEPTION] print model arch throws exception {ex}')


def read_the_input_for_sliding_windows(input_list, path_dictionary, dummy_shape=(2974, 2960, 0), isNumpy=False, list_of_exclusion=None):
    try:
        dummy_shape = (dummy_shape[0], dummy_shape[1], 0)
        main = np.empty(dummy_shape)

        for ctn, image in enumerate(input_list):
            image_to_analyse = path_dictionary[image]
            print(f'[LOADING] [{ctn}] image -> {image_to_analyse}')

            # check for exclusion
            if list_of_exclusion is not None and image in list_of_exclusion:
                print(f'[EXCLUDED] [{ctn}] image -> {image_to_analyse}')
                continue

            # get the image and shape
            if isNumpy:
                tif_image = np.load(image_to_analyse).astype('float32')
                tif_image / 255.0
                height, width, channels = tif_image.shape
            else:
                tif_image = load_single_image_utility(path=image_to_analyse)
                height, width, channels = tif_image.shape

            # correct if necessary
            if tif_image.shape[0] != dummy_shape[0]:
                tif_image = tif_image[0:dummy_shape[0], :, :]

            if tif_image.shape[1] != dummy_shape[1]:
                tif_image = tif_image[:, 0:dummy_shape[1], :]

            # if the first image make an array


            # combine all array together
            main = np.concatenate((main, tif_image), axis=2)
            print(f'[LOAD COMPLETE] Load image complete height: {height} width: {width} channel: {channels}')
        print(f'[LOAD COMPLETE] The input is ready {main.shape}')

        return main
    except Exception as ex:
        print(f'[EXCEPTION] Read the input for sliding windows throws exception {ex}')

def create_confusion_matrices(true_labels, predicted_labels, labels_identification, title, path):
    try:
        # confusion matrix by scikit
        cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)

        plt.figure(figsize=(30, 30))
        plt.title(f"{title}")
        sn.heatmap(cm, fmt='g', cmap="YlGnBu",
                   annot=True, cbar=False,
                   xticklabels=labels_identification,
                   yticklabels=labels_identification)

        plt.xlabel("Y pred")
        plt.ylabel("Y true")

        plt.savefig(f"{path}")
        plt.show()

    except Exception as ex:
        print(f"Create confusion matrix trow exception {ex}")


def plot_new_cm(labels_id, y_true, y_pred, path, already_made_cm=None ,title='Confusion matrix', cmap=None, normalize=False, accuracy=None):
    try:
        # calc acc
        if already_made_cm is None:
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        else:
            cm = already_made_cm

        if accuracy is None:
            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(20, 20))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        if accuracy is None:
            plt.title(f'Predicted labels\n{title} - accuracy {accuracy} miss class {misclass}')
        else:
            plt.title(f'Predicted labels\n{title} - accuracy {accuracy}')

        if labels_id is not None:
            tick_marks = np.arange(len(labels_id))
            plt.xticks(tick_marks, labels_id, rotation=45)
            plt.yticks(tick_marks, labels_id)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path)
        #plt.show()
    except Exception as ex:
        print(f"Plot new cm throws exception {ex}")

def move_history_accuracy(path, folder_number):
    try:
        p = '../Results/'
        for round_number in range(122):
            if os.path.exists(f'{p}{round_number}'):
                path = f'{p}{round_number}'
                for tif in os.listdir(path):
                    image_folder = path + '/' + tif
                    print(f'[INFO] Walking in {image_folder}')
                    #if os.path.exists(f'{image_folder}/total_accuracy.csv'):
                        #total_accuracy = f'{image_folder}/total_accuracy.csv'
                        #print(total_accuracy)
                        #shutil.move(total_accuracy, f'/home/luca/Documents/maati/Paavo/Results/second stage_normal/total_acc_valid/{round_number}_{tif}_valid_acc.csv')
                    for sub_image in os.listdir(image_folder):
                        if sub_image == 'total_accuracy.csv':
                            print(sub_image)
                            continue
                        else:

                            single_path = f'{image_folder}/{sub_image}/history.csv'
                            print(f'[FILE] File -> {single_path}')
                            if os.path.exists(f'{single_path}'):
                                print(single_path)
                                shutil.move(single_path, f'/home/luca/Documents/maati/Paavo/Results/fold_history/{round_number}_{tif}_{sub_image}_folds_history.csv')
    except Exception as ex:
        print(f"Move history accuracy throws exception {ex}")


def plot_pixelwise_classification(true_window, predicted_window, true_indexed_window, predicted_indexed_window):
    try:
        labels = list(colors.keys())
        col = list()
        rgb = list(colors.values())

        for c in rgb:
            col.append(rgb_to_hex(c))

        cmap = LinearSegmentedColormap.from_list('whatever', col, N=len(labels))
        fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(20, 20), gridspec_kw={"width_ratios": [1, 1]})
        #ax2.set_xticks([])
        #ax2.set_yticks([])

        ax.set_title("True window")
        ax.axis('off')
        ax1.set_title("Predicted window")
        ax1.axis('off')

        img_1 = ax.imshow(true_window, vmin=0, vmax=len(labels), cmap=cmap)
        img_2 = ax1.imshow(predicted_window, vmin=0, vmax=len(labels), cmap=cmap)

        for i, j in itertools.product(range(true_indexed_window.shape[0]), range(true_indexed_window.shape[1])):
            ax.text(j, i, true_indexed_window[i, j],
                     horizontalalignment="center",
                     color="black")
            ax1.text(j, i, predicted_indexed_window[i, j],
                     horizontalalignment="center",
                     color="black")

        #cbar = fig.colorbar(img_1, cax=ax2, ticks=range(0, len(labels), 1))
        #cbar.set_ticklabels(ticklabels=labels)
        fig.savefig('../Results/pixelwise_classification/true_pred_windows.png')
        plt.show()
    except Exception as ex:
        print(f'Plot pixelwise classification throws exception {ex}')

def rgb_to_hex(rgb):
    return '#' + ('%02x%02x%02x' % rgb)


def create_csv_confusion_matrices(stack_true, stack_pred, labels_id, save_folder, accuracy, title="confusion_matrix" ,get_folder_number=0):
    try:
        np.savetxt(f'{save_folder}/y_stacked_true.txt', stack_true, fmt="%d")
        np.savetxt(f'{save_folder}/y_stacked_pred.txt', stack_pred, fmt="%d")
        np.savetxt(f'{save_folder}/labels_id.txt', labels_id, fmt="%d")

        cm = confusion_matrix(stack_true, stack_pred)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(cm, columns=[sorted(labels_id)], index=sorted(labels_id))
        df.to_csv(f"../Results/{save_folder}/{title}_acc_{accuracy}.csv")
    except Exception as ex:
        print(f"Print csv confution matrices throws exception {ex}")


def create_true_pred_labels_report(save_path, stack_true, stack_pred, labels_id):
    try:
        handler = open(f'{save_path}', 'w')
        handler.writelines('true_labels, predicted_labels\n')
        for i in range(len(stack_true)):
            val_true = stack_true[i]
            val_pred = stack_pred[i]
            print(f'{val_true} -- {val_pred}')
            handler.writelines(f'{labels_id[val_true]},{labels_id[val_pred]}\n')
        handler.close()
    except Exception as ex:
        print(f'Create true and pred labels throws exception {ex}')


def read_the_summary_files(main_folder):
    try:
        get_folder_number = int(re.search(r'\d+', main_folder).group())

        file_handler = open(f'../Results/summary/{main_folder}', 'r')
        for line in file_handler.readlines():
            if line.find('get the best ->') != -1:
                line = line.replace('get the best ->', '')
                # get index of with
                get_with_index = line.index('with')
                accuracy = line[get_with_index + 17:]
                line = line[: get_with_index].strip()
            else:
                continue
        return get_folder_number, accuracy, line
    except Exception as ex:
        print(f'Read the summary fles throws exception {ex}')


def get_true_hits(save_path, true_indexed_window, predicted_indexed_window):
    try:
        ctn = 0
        temp = np.empty((10, 10), dtype='bool')
        for x in range(0, true_indexed_window.shape[0]):
            for y in range(0, true_indexed_window.shape[1]):
                print(true_indexed_window[x, y] == predicted_indexed_window[x, y])
                temp[x, y] = true_indexed_window[x, y] == predicted_indexed_window[x, y]
                if true_indexed_window[x, y] == predicted_indexed_window[x, y]:
                    ctn += 1
        print(f'[HITS COUNTER] Got {ctn}')
        np.savetxt(f'{save_path}', temp, fmt="%s")
    except Exception as ex:
        print(f'Get the windows indexed throws exception {ex}')

def count_the_labels(window):
    try:
        # get unique class id from the windows
        unique, count = np.unique(window, return_counts=True)
        for u, c in zip(unique, count):
            print(f'[COUNT] class_id {u} count {c}')
    except Exception as ex:
        print(f'count the labels throws exceptions {ex}')



def create_full_image_labelled(image, output_path, dict_color):
    try:
        # qui labelled e no
        tif_image = load_single_image_utility(path=image)
        placeholder = tif_image[:,:, 0:1]

        labels = list(dict_color.keys())
        col = list()
        rgb = list(dict_color.values())

        for c in rgb:
            col.append('#' + ('%02x%02x%02x' % c))

        fig, ax = plt.subplots(figsize=(40,40))
        ax.set_title("True labels Sentinel2")
        ax.axis('off')
        cmap = LinearSegmentedColormap.from_list('whatever', col, N=len(labels))
        img_1 = ax.imshow(placeholder, vmin=0, vmax=len(labels), cmap=cmap, interpolation='bilinear')
        plt.show()
        fig.savefig(output_path)

    except Exception as ex:
        print(f'Create full windows labelled throws exception {ex}')

def mark_the_pixels(pathes_dict):
    try:
        keys = list(pathes_dict.keys())
        values = list(pathes_dict.values())

        for k, v in zip(keys, values):
            # stage 1 load the image and the csv file with the stuff
            print(f'[TIF PATH] Loading {k}')
            ori = load_single_image_utility(path=v)
            csv_path = '../Results/best_combination/data/' + k.split('/')[3] + '_points_translated.csv'
            print(f'[CSV PATH] {csv_path}')

            # standard scale the images
            scaler = StandardScaler()
            ori = scaler.fit_transform(ori.reshape(-1, ori.shape[-1])).reshape(ori.shape)

            handler = open(f'{csv_path}', 'r')
            ctn = 0
            for points in handler.readlines():
                if ctn == 0:
                    ctn += 1
                    continue

                stripped_points = points.strip()
                x = int(stripped_points.split(',')[0])
                y = int(stripped_points.split(',')[1])
                id = float(stripped_points.split(',')[2])
                print(f'[POINTS] X -> {x} Y -> {y} Id -> {id}')

                # foreach channnel mark the pixel
                for channel in range(ori.shape[2]):
                    ori[y, x, channel] = id
                    print(f'[POSITION {channel}] In position x -> {x} y -> {y} id -> {id} value {ori[y, x, channel]}')


            # save as np array
            filename = v.split('/')[-1]
            filename = filename.split('.')[0]
            np.save(f'../TifMarked/{filename}', ori)
            print("------")
    except Exception as ex:
        print(f'Mark the pixels throws exception {ex}')

def print_stacked_windows(total_windows, first_path='../Results/pixelwise_classification/predicted/true_window_0_normal.txt',
                          first_pred='../Results/pixelwise_classification/predicted/predicted_0_window.txt',
                          first_25='../Results/pixelwise_classification/predicted/true_window_0_raster.txt'):
    try:
        first = np.loadtxt(f'{first_path}').astype('uint8')
        first_pred = np.loadtxt(f'{first_pred}').astype('uint8')
        first_25 = np.loadtxt(f'{first_25}').astype('uint8')

        for i in range(total_windows):
            if i == 0:
                continue
            elif i == 1:

                second = np.loadtxt(f'../Results/pixelwise_classification/predicted/true_window_{i}_normal.txt').astype('uint8')
                temp = np.hstack((first, second))

                second_pred = np.loadtxt(f'../Results/pixelwise_classification/predicted/predicted_{i}_window.txt').astype('uint8')
                temp_pred = np.hstack((first_pred, second_pred))

                second_25 = np.loadtxt(f'../Results/pixelwise_classification/predicted/true_window_{i}_raster.txt').astype('uint8')
                temp_25 = np.hstack((first_25, second_25))

            else:
                other = np.loadtxt(f'../Results/pixelwise_classification/predicted/true_window_{i}_normal.txt').astype('uint8')
                temp = np.hstack((temp, other))

                other_pred = np.loadtxt(f'../Results/pixelwise_classification/predicted/predicted_{i}_window.txt').astype('uint8')
                temp_pred = np.hstack((temp_pred, other_pred))

                other_25 = np.loadtxt(f'../Results/pixelwise_classification/predicted/true_window_{i}_raster.txt').astype('uint8')
                temp_25 = np.hstack((temp_25, other_25))

        plt.figure(figsize=(150,20))
        plt.axis('off')
        plt.title('first 15 predicted windows')
        img = plt.imshow(temp_pred)

        for i, j in itertools.product(range(temp_pred.shape[0]), range(temp_pred.shape[1])):
            if int(temp_pred[i, j]) in list(colors.keys()):
                plt.text(j, i, temp_pred[i, j],
                         horizontalalignment="center",
                         color="black")
        plt.savefig('../Results/pixelwise_classification/predicted/first_15_pred.png')
        plt.show()

        plt.figure(figsize=(150,20))
        plt.axis('off')
        plt.title('first 15 true 25x25 windows')
        img = plt.imshow(temp_25)

        for i, j in itertools.product(range(temp_25.shape[0]), range(temp_25.shape[1])):
            if int(temp_25[i, j]) in list(colors.keys()):
                plt.text(j, i, temp_25[i, j],
                         horizontalalignment="center",
                         color="black")

        plt.savefig('../Results/pixelwise_classification/predicted/first_15_raster.png')
        plt.show()

        plt.figure(figsize=(150,20))
        plt.axis('off')
        plt.title('first 15 true _normal windows')
        img = plt.imshow(temp)

        for i, j in itertools.product(range(temp.shape[0]), range(temp.shape[1])):
            if int(temp[i, j]) in list(colors.keys()):
                plt.text(j, i, temp[i, j],
                         horizontalalignment="center",
                         color="black")

        plt.savefig('../Results/pixelwise_classification/predicted/first_15_normal.png')
        plt.show()
    except Exception as ex:
        print(f'Print the windows throws exception {ex}')

def create_final_prediction_from_array(source_file, output_file, dpi, title, is_print_label=False, cmap=False):
    try:
        # load the first one
        final_array = np.load(f'{source_file}').astype('uint8')
        print(f'[FINAL WINDOW SHAPE] {final_array.shape}')
        h, w = final_array.shape
        figsize = w / float(dpi), h / float(dpi)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(title)

        if cmap:
            col = list()
            rgb = list(colors.values())

            for c in rgb:
                col.append('#' + ('%02x%02x%02x' % c))

            cmap = LinearSegmentedColormap.from_list('whatever', col, N=len(list(colors.keys())))
            img = plt.imshow(final_array[:, :], vmin=0, vmax=len(list(colors.keys())), cmap=cmap)
        elif cmap == 'jet':
            img = plt.imshow(final_array[:, :], cmap='jet')
        else:
            img = plt.imshow(final_array[:, :], vmin=0, vmax=len(list(colors.keys())))

        if is_print_label:
            for i, j in itertools.product(range(final_array.shape[0]), range(final_array.shape[1])):
                if int(final_array[i, j]) in list(colors.keys()):
                    plt.text(j, i, final_array[i, j],
                             horizontalalignment="center",
                             color="black")

        plt.savefig(f'{output_file}')
        #plt.show()
    except Exception as ex:
        print(f"create_final_prediction_from_array throws exception {ex}")

def sliding_windows(input_normal, input_raster, steps, windows_size_normal=(11, 11), windows_size_raster=(25,25)):
    try:
        # get total height
        total_height = input_normal.shape[0]
        # get total width
        total_width = input_normal.shape[1]
        for h in range(0, total_height, steps):
            for w in range(0, total_width, steps):
                windows_normal = input_normal[h:h + windows_size_normal[1], w:w + windows_size_normal[0], :]
                windows_raster = input_raster[(2 + h * 5 - windows_size_raster[1]) // 2:(2 + h * 5 + windows_size_raster[1]) // 2, (2 + w * 5 - windows_size_raster[0]) // 2:(2 + w * 5 + windows_size_raster[0]) // 2 :]
                yield [h, w, windows_normal, windows_raster]
    except Exception as ex:
        print("CV sliding windows throws exception " + str(ex))

def build_two_input_model_pixels_prediction(input_shape_normal, input_shape_raster, output=39, last_activation='softmax'):
    try:
        normal = tf.keras.Input(shape=input_shape_normal, name="input_normal")
        raster = tf.keras.Input(shape=input_shape_raster, name="input_raster")

        # first NORM
        x = tf.keras.layers.Conv2D(8, 3, activation='relu', padding="same", name="conv2d_normal_1")(normal)
        x = tf.keras.layers.BatchNormalization(name="batch_norm_normal_1")(x)
        #x = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_normal_1", padding="same")(x)
        # sec
        x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same", name="conv2d_normal_2")(x)
        x = tf.keras.layers.BatchNormalization(name="batch_norm_normal_2")(x)
        #x = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_normal_2")(x)
        # flatten
        #x = tf.keras.layers.Flatten(name="flatten_normal")(x)
        x = tf.keras.Model(inputs=normal, outputs=x, name="model_with_normal_input")

        # first RASTER
        y = tf.keras.layers.Conv2D(8, 3, activation='relu', padding="same", name="conv2d_raster_1")(raster)
        y = tf.keras.layers.BatchNormalization(name="batch_norm_raster_1")(y)
        #y = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_raster_1")(y)
        # sec
        y = tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same", name="conv2d_raster_2")(y)
        y = tf.keras.layers.BatchNormalization(name="batch_norm_raster_2")(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=(16,16), strides=(1,1), padding="valid", name="get_same_size")(y)
        # flatten
        #y = tf.keras.layers.Flatten(name="flatten_raster")(y)
        y = tf.keras.Model(inputs=raster, outputs=y, name="model_with_raster_input")

        # combined
        combined = tf.keras.layers.Concatenate(axis=3)([x.output, y.output])
        testing = tf.keras.layers.Flatten(name="flatten_raster")(combined)
        #classifier = tf.keras.layers.Conv2D(39, 1, activation=last_activation, padding="same", name="list_for_the_report")(combined)
        classifier = tf.keras.layers.Dense(output, activation=last_activation, name="classifier")(testing)
        model = tf.keras.Model(inputs=[x.input, y.input], outputs=classifier, name="windows_prediction")

        return model

    except Exception as ex:
        print(f"Create two input network throws exception  {ex}")

def create_the_segmented_final_windows(windows_to_analyse, save_path):
    try:
        index_array = np.argmax(windows_to_analyse, axis=-1)
        result = np.zeros((windows_to_analyse.shape[0], windows_to_analyse.shape[1]), dtype='uint8')
        for y in range(index_array.shape[0]):
            for x in range(index_array.shape[1]):
                value_to_search = index_array[y, x]
                result[y, x] = windows_to_analyse[y, x, value_to_search]
        if save_path is not None:
            np.savetxt(f'{save_path}', result, fmt="%d")
        return result
    except Exception as ex:
        print(f'Cretae the final image throws exception{ex}')

def write_points_translated(points_translated, labels, file_path='../Results/csv/points_translated.csv'):
    try:
        file = open(file_path, 'w')
        file.writelines("SyotyypiID,X,Y\n")
        for i in range(len(labels)):
            file.writelines(f"{labels[i]},{points_translated[i][0, 0]},{points_translated[i][0, 1]}\n")
        file.close()
    except Exception as ex:
        print(f"Write points throws exceptions {ex}")


def sliding_window_yielding(input, steps, windows_size=(128, 128), list_to_render=None, is_single_channel=False, debug=False):
    try:
        # get total height
        total_height = input.shape[0]
        # get total width
        total_width = input.shape[1]
        for el in list_to_render:
            y = el[1]
            x = el[0]
            for h in range(0, total_height, steps):
                for w in range(0, total_width, steps):
                    if h <= y <= h + windows_size[0] and w <= x <= w + windows_size[1]:
                        if is_single_channel is False:
                            windows = input[h:h + windows_size[0], w:w + windows_size[1], :]
                        elif is_single_channel == 0:
                            windows = input[h:h + windows_size[0], w:w + windows_size[1]]
                        else:
                            windows = input[h:h + windows_size[0], w:w + windows_size[1]]
                        yield w, h, windows
                    else:
                        continue
    except Exception as ex:
        print(f"Sliding windows throws exception {ex}")


def write_raster_file(input, path, windows_size=(50,50), channel=8):
    with rasterio.open(path, 'w', driver='GTiff', width=windows_size[1], height=windows_size[0], dtype=rasterio.uint8, count=channel) as dist:
        dist.write(input.astype(rasterio.uint8))


def transform_coordinates(x, y, gt, ot):
    # tensore (h, w, c)
    # qui x div w altra immagine * w immagine corretta, y div h altra immage * h immagine corretta
    return x / ot.shape[1] * gt.shape[1], y / ot.shape[0] * gt.shape[0]


def return_points_from_labels(labels_id, file_path='../Results/csv/points_translated.csv'):
    try:
        file = open(file_path, 'r')
        list_to_render = list()
        jump_a_line = 0
        for f in file:
            if jump_a_line == 0:
                jump_a_line += 1
                continue
            # print(f.split(','))
            if int(f.split(',')[0]) == labels_id:
                x = f.split(',')[1]
                y = f.split(',')[2].strip('\n')
                #print(f"labels_id -> {labels_id} -> {x} <--> {y}" )
                list_to_render.append([int(x), int(y)])
        #print(list_to_render)
        file.close()
        return list_to_render
    except Exception as ex:
        print(f"Return points translated throws exceptions {ex}")


def return_points_same_place(labels_id, file_path="../Results/csv/points_translated.csv", isChm=None):
    try:
        file = open(file_path, 'r')
        list_to_render = list()
        jump_a_line = 0
        for f in file:
            if jump_a_line == 0:
                jump_a_line += 1
                continue
            # print(f.split(','))
            if int(f.split(',')[0]) == labels_id:
                x = int(f.split(',')[1])
                y = int(f.split(',')[2].strip('\n'))
                if isChm is not None:
                    x_t = int((5 + (x - 1) * 10))
                    y_t = int((5 + (y - 1) * 10))
                else:
                    x_t = int((3 + (x - 1) * 5))
                    y_t = int((3 + (y - 1) * 5))
                # print(f"labels_id -> {labels_id} -> {x} <--> {y}" )
                list_to_render.append([int(x_t), int(y_t)])
        # print(list_to_render)
        file.close()
        return list_to_render
    except Exception as ex:
        print(f"Return points same place throws exception {ex}")

def output_all_header_files(path_to_file, output_path):
    """
    transform format:
    https://www.perrygeo.com/python-affine-transforms.html
    [0] = width of a pixel
    [1] = row rotation (typically zero)
    [2] = x-coordinate of the upper-left corner of the upper-left pixel
    [3] = column rotation (typically zero)
    [4] = height of a pixel (typically negative)
    [5] = y-coordinate of the of the upper-left corner of the upper-left pixel
    """
    try:
        handler = open(output_path, 'w')
        handler.write(f"tif_image,driver,width,height,bands,transform_width_of_a_pixel,transform_row_rotation,transform_x_coordinates_of_upper_left_pixel,transform_col_rotation,transform_height_of_a_pixel,transform_y_coordinate_of_upper_left_pixel")
        handler.write('\n')

        for image in os.listdir(path_to_file):
            filename, extension = os.path.splitext(image)
            if extension != '.tif':
                print(f"{image} is not an image")
                continue
            with rasterio.open(path_to_file + '/' + image) as src:
                line = f"{image},{src.driver},{src.width},{src.height},{src.count},{src.profile['transform'][0]},{src.profile['transform'][1]},{src.profile['transform'][2]},{src.profile['transform'][3]},{src.profile['transform'][4]},{src.profile['transform'][5]}"
                handler.write(line)
                handler.write('\n')
        handler.close()
    except Exception as ex:
        print(f"Output all header file throws exception {ex}")

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

def create_sliding_windows_compact(tif_array, windows_size, delta=10, center_N=None, center_E=None):
    try:
        coordinates = []

        # Build the 25 meters x 25 meters window or 10 meters x 10 meters
        for y in range(windows_size):
            N = center_N + math.floor(windows_size / 2) * delta - y * delta  # Were going up-down row by row
            for x in range(windows_size):
                E = center_E - math.floor(windows_size / 2) * delta + x * delta  # We go from left to right
                coordinates.append((E, N))

        values_with_need = [tiff_val for tiff_val in tif_array.sample(coordinates)]

        # create the window
        window = np.array(values_with_need).astype('uint8').reshape((windows_size, windows_size, -1))
        return window

    except Exception as ex:
        print(f"Create sliding windows compact throws exception {ex}")

def create_sliding_windows_with_gdal(tif_image, windows_size, delta=10, center_N=None, center_E=None):
    try:
        # sliding wins
        #main_window_tensor = np.empty((windows_size, windows_size, 0))
        #for path in tit_list:
            #print(f'[WALKING] Walking in {path}')

        # start from a point that we decide
        if center_N is not None and center_E is not None:
            Na = center_N
            Ea = center_E
        else:
            Ea, Na, delta = get_the_geoinfo(input_tiff_path=tif_image)

        window_as_a_1d_coordinate_list = []

        # Build the 25 meters x 25 meters window or 10 meters x 10 meters
        for y in range(windows_size):
            N = Na  + math.floor(windows_size / 2) * delta - y * delta  # Were going up-down row by row
            for x in range(windows_size):
                E = Ea - math.floor(windows_size / 2) * delta + x * delta  # We go from left to right
                window_as_a_1d_coordinate_list.append((E, N))

            # get the correct points
            tiff_info = get_tiff_data_at_coord_locations(tif_image, window_as_a_1d_coordinate_list)

            # get the pixels list
            pixel_values_as_a_big_long_list = tiff_info['pixel_values_at_coords']

            # create the window
            window = np.array(pixel_values_as_a_big_long_list).astype('uint8').reshape((windows_size, windows_size, -1))
            #print(f'[TEMP ARRAY ] temp window shape -> {temp_array.shape}')
            # concatenate with the other
            #main_window_tensor = np.concatenate((main_window_tensor, temp_array), axis=2)
            #print('-------')
        #print(f'[FINAL] Final {window.shape}')
        return window
    except Exception as ex:
        print(f'[EXCEPTION] Create sliding windows with gdal throws exception {ex}')


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

def openTiffFileAsRaster(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def saveRasterToTiffFile(arr, ds, filename, geotrans):
    [cols, rows] = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    # outdata = driver.Create(filename, rows, cols, 1, gdal.GDT_Float32)
    outdata = driver.Create(filename, rows, cols, 1)
    # outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetGeoTransform(geotrans)  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr)
    # outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None

def make_numpy_array_readable_for_matlab(output, path=None, input=None):
    try:
        if path is not None:
            array = np.load(f'{path}')
        else:
            array = input
        scipy.io.savemat(f'{output}', mdict={"window": array})
    except Exception as ex:
        print(f'Make numpy array readable for matlab throws exception {ex}')

def get_better_fold(path=f"../Results/test_one_input/weights/"):
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

def read_master_file(path):
    try:

        list_normal = []
        list_nvi = []
        list_chm = []
        list_dem = []
        windows_dimension = []

        handler = open(path, 'r')
        for file in handler.readlines():
            splitted = file.strip().split(',')
            if splitted[0] == 'sar':
                list_normal.append(splitted[1])
            elif splitted[0] == 'nvi':
                list_nvi.append(splitted[1])
            elif splitted[0] == 'chm':
                list_chm.append(splitted[1])
            else:
                list_dem.append(splitted[1])

            if len(splitted) > 1:
                windows_dimension.append(int(splitted[2]))

        return list_normal, list_nvi, list_chm, list_dem, windows_dimension
    except Exception as ex:
        print(f"Read master file throws exception {ex}")


    try:
        handler = open(f'{file_input}')

        shapefile = gpd.read_file(annotation)
        group = shapefile.groupby('SuotyypId').Suotyyppi.count()

        id = list()
        dist = list()

        jump_one = 0
        for line in handler.readlines():
            if jump_one == 0:
                jump_one += 1
                continue
            id.append(int(line.strip().split(',')[0]))
            dist.append(int(line.strip().split(',')[1]))

        sns.set(rc={'figure.figsize': (20, 20)})
        ax = sns.barplot(group.index, group.values)
        ax.set(xlabel=None)
        # ax.set_ylabel('')
        ax.set_xlabel('')
        plt.savefig(f'{file_output}')
        plt.show()
    except Exception as ex:
        print(f"{ex}")

def read_master_file_input_selection(path):
    try:
        handler = open(path)
        splitter = {}
        windows_dimension = []
        for file in handler.readlines():
            splitted = file.strip().split(',')
            if int(splitted[2]) == 5:
                splitter[splitted[1]] = 5
            else:
                splitter[splitted[1]] = 25
            windows_dimension.append(int(splitted[2]))
        handler.close()

        return splitter, windows_dimension
    except Exception as ex:
        print(f'read_master_file_input_selection throws exception {ex}')

def make_average(list_to_analyse):
    tot = 0
    for i in list_to_analyse:
        tot += float(i)
    return tot / float(len(list_to_analyse))

def prepare_the_input_injection(dictionary_of_input, the_bests, input_selected):
    try:
        res = list()
        for ctn, folder in enumerate(list(dictionary_of_input.keys())):
            if folder in the_bests:
                print(f"[EXCLUSION] of {folder}")
            #elif dictionary_of_input[folder] == input_selected:
                #continue
            else:
                res.append(folder)
        return res
    except Exception as ex:
        print(f"[EXCEPTION] prepare_the_input_injection {ex}")

def filter_the_input(dictionary_of_input, the_bests, input_type):
    try:
        res = list()
        main_folder = dictionary_of_input.keys()
        for ctn, main_path in enumerate(main_folder):
            #print(f"We are in main folder {main_path}")
            # get the path and eclude the path that we do not need
            for path in os.listdir(main_path):
                path_to_load = main_path + '/' + path
                if path_to_load in the_bests:
                    print(f"[EXCLUSION] of {path_to_load}")
                    continue
                if dictionary_of_input["../Dataset/" + path_to_load.split('/')[2]] == input_type:
                    continue
                #print(f"{[ctn]} We are looking at the {path_to_load}")
                res.append(path_to_load)
        return res
    except Exception as ex:
        print(f"filter_the_input throws exception {ex}")
import argparse
import traceback
import numpy as np 
import tensorflow as tf 

import os 
import sys 

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Configuration.configuration import config, area_manager, cnn_fine_tuning
from sklearn.model_selection import StratifiedKFold
from Functions.InputSelection_utilities import (calculate_features_number, 
                                                load_data_and_configuration, 
                                                make_one_hot_encoding, 
                                                make_the_folder_work,
                                                output_list_converter,
                                                do_the_fitting_multiprocessing,
                                                do_round_accuracy_for_single,
                                                make_the_mean,
                                                get_the_best)
from joblib import Parallel, delayed
import time


# we do not need to stack here 
config["STACKING"] = False
#config["channel_list"] = ["channel_1", "channel_2", "channel_3"]


if __name__ == "__main__":
    try:

        np.set_printoptions(threshold=None)

        parser = argparse.ArgumentParser()

        parser.add_argument("-area", 
                            help="the area you want to analyse", 
                            type=str, 
                            required=True)

        parser.add_argument("-types", 
                            help="undrained or drained", 
                            type=str, 
                            required=True)
        
        args = parser.parse_args()

        #current_masterfile = area_manager[args.area][f"MASTERFILE_{args.types.upper()}"]

        config["DATASET_PATH"] = os.path.join("..", "Windows", "TestArea", args.types.lower())

        config["FEATURE_SAVE_FOLDER"] = area_manager[args.area]["INPUT_SELECTION_1_STAGE"] + f"_{args.types.upper()}"
        config["FEATURES_SUMMARY"] = area_manager[args.area]["INPUT_SELECTION_1_STAGE"] + f"_{args.types.upper()}"

        tf.get_logger().setLevel('ERROR')

        if config["NOGPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            for i in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[i], True)

        best_dict = {}

        #all_possible_combination = calculate_features_number(path=current_masterfile)
        input_data, total_number_of_samples, windows_ctn = load_data_and_configuration(current_masterfile=config['DATASET_PATH'])

        if config["STDOUT"]:
            print(f'[DATA CONFIG] {input_data}')
            print(f'[DATA CONFIG] Channel list {input_data.keys()}')
            print(f"[DATA CONFIG] total number of samples {total_number_of_samples}")
            print(f"[DATA CONFIG] total number of windows {windows_ctn}")
            print("----")

        #for channel in input_data.keys():
        #    print(channel, all_possible_combination)
 
        for channel in input_data.keys():
            
            # labels set are the same everywhere
            label_set = input_data[channel][0].label_set
        
            # create a besty dict to hold name and accuracy of the best
            besty = {}

            # prepare cnn config
            # rest cnn each channel change
            cnn_configuration = {
                "input_channel_1": None,
                "input_channel_2": None,
                "input_channel_3": None,
                "input_channel_4": None,
                "input_channel_5": None,
                "neuron_numbers": cnn_fine_tuning["neurons_numbers"],
                "kernel_channel_1": None,
                "kernel_channel_2": None,
                "kernel_channel_3": None,
                "kernel_channel_4": None,
                "kernel_channel_5": None,
                "inputs": 1,
                "output": len(np.unique(label_set)),
                "stride": cnn_fine_tuning["stride"],
                "regularization": cnn_fine_tuning["regularization"],
                "data_augmentation": False,
                'rotation_factor': cnn_fine_tuning["rotation_factor"],
                "last_activation": 'softmax',
                "dropout_rate": cnn_fine_tuning["dropout_rate"]
            }

            # prepareing the encoding label
            label_encoded = make_one_hot_encoding(labels=label_set)

            # instance of kfold
            kfold = StratifiedKFold(n_splits=config["CV"], shuffle=True)

            for member in input_data[channel]:

                # this argmax is temp just to get the folds
                kfold.get_n_splits(member.data, label_set)
                folds = list(kfold.split(member.data, label_set))


                # configure the cnn params
                cnn_configuration[f"input_{channel}"] = (member.shape[1], member.shape[2], member.shape[3])
                cnn_configuration[f"kernel_{channel}"] = (member.shape[3], member.shape[3])
                print(cnn_configuration)
                # start timer
                round_timer = time.time()

            
                # create empty variables that hold sat name img name and best
                satellite_folder = member.satellite_name
                image_name = member.image_name
                list_of_current_features = list()
                
                best_set = {}

                # create the folder structore like before
                saving_path, predictions_path, satellite_path, weights_folder = make_the_folder_work(
                    feature_save_folder=os.path.join(config["FEATURE_SAVE_FOLDER"], image_name))

                # show CNN auto config
                if config["STDOUT"]:
                    print(f"[CONFIG] Configuration complete")
                    print(f"[CONFIG] CNN is configured now")
                    print(cnn_configuration)

                fold_counter = 0
                list_of_scored_accuracy = list()
                list_of_train_accuracy = list()
               
                # start timer
                cv_timer = time.time()
                if config["NOGPU"]:
                    out = Parallel(n_jobs=config['CV'], verbose=100)(
                        delayed(do_the_fitting_multiprocessing)(configuration=config,
                                                                current_cnn_config=cnn_configuration,
                                                                encoded_labels=label_encoded,
                                                                channel_list=[channel],
                                                                input_channel_1=member.data if "channel_1" == channel else None,
                                                                input_channel_2=member.data if "channel_2" == channel else None,
                                                                input_channel_3=member.data if "channel_3" == channel else None,
                                                                input_channel_4=member.data if "channel_4" == channel else None,
                                                                input_channel_5=member.data if "channel_5" == channel else None,
                                                                train_index=train_idx,
                                                                best_data=best_set,
                                                                validation_index=validation_idx) for
                        train_idx, validation_idx in kfold.split(member.data, label_set)
                    )
                    # end timer
                    print(f"[TIMER] Elapsed timer {time.time() - cv_timer} s")

                    print(f"[SAVE THE CV] Saving CV from mult processing")

                    list_of_scored_accuracy, list_of_train_accuracy = output_list_converter(output_list=out,
                                                                                            saving_path=saving_path,
                                                                                            predictions_path=predictions_path,
                                                                                            satellite_path=satellite_path,
                                                                                            real_label=label_set,
                                                                                            configuration=config)

                else:
                    out = list()
                    for train_idx, validation_idx in kfold.split(np.zeros(len(member.data)), label_set):
                        print(f"[GPU] I am with the GPUs")
                        o = do_the_fitting_multiprocessing(configuration=config,
                                                            channel_list=[channel],
                                                            current_cnn_config=cnn_configuration,
                                                            encoded_labels=label_encoded,
                                                            input_channel_1=member.data if "channel_1" == channel else None,
                                                            input_channel_2=member.data if "channel_2" == channel else None,
                                                            input_channel_3=member.data if "channel_3" == channel else None,
                                                            input_channel_4=member.data if "channel_4" == channel else None,
                                                            input_channel_5=member.data if "channel_5" == channel else None,
                                                            train_index=train_idx,
                                                            validation_index=validation_idx,
                                                            best_data=best_set)
                        out.append(o)
                    # end timer
                    print(f"[TIMER] Elapsed timer {time.time() - cv_timer} s")

                    print(f"[SAVE THE CV] Saving CV from mult processing")
                    list_of_scored_accuracy, list_of_train_accuracy = output_list_converter(output_list=out,
                                                                                            saving_path=saving_path,
                                                                                            predictions_path=predictions_path,
                                                                                            satellite_path=satellite_path,
                                                                                            real_label=label_set,
                                                                                            configuration=config)
                # generate confusion matrices
                if config["CONFUSION"]:
                    os.system(f'{os.path.join("..", "Utilities", "confusion_matrix_creation.py")} --folder {satellite_path} --type all')

                            
                do_round_accuracy_for_single(path=os.path.join(config["FEATURE_SAVE_FOLDER"], f"summary_{len(input_data.keys())}.csv"),
                                             channel=channel,
                                             current_input=member.full_path,
                                             mean_of_accuracy=make_the_mean(list_of_scored_accuracy))

            # HERE I HAVE TO ADD THE BEST
            best_string = get_the_best(path_of_summary=os.path.join(config["FEATURE_SAVE_FOLDER"], f"summary_{len(input_data.keys())}.csv"))

            best_dict[best_string[0]] = os.path.join(best_string[1])

        if config["STDOUT"]:
            print(f"[BEST] The best is {best_dict} with a score of  {best_string[2]}")
            
            print(f"[TIMER] Round timer is {time.time() - round_timer}")
        
    except Exception as ex:
        print(f"Main throws exception {ex}")
        print(traceback.print_exc())

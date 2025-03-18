import argparse
import os
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import sys 

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)



from Functions.InputSelection_utilities import (load_the_best_combination_dictionary, 
                                                load_data_and_configuration,
                                                make_one_hot_encoding, 
                                                prepare_the_sets_for_analysis,
                                                make_the_folder_work,
                                                do_the_fitting_multiprocessing_v2,
                                                output_list_converter,
                                                do_the_final_report_v2,
                                                write_summary_v2)

from Configuration.configuration import config, area_manager, cnn_fine_tuning
from Configuration.input_selection_placeholder import AnalyseTheInputs
from joblib import Parallel, delayed
import time
import pickle

# overload all the variables
config["INPUTS"] = 2
config["DUMP"] = True
config["EPOCHS"] = 1

channel_permutationms  = [['channel_1', 'channel_3'], ['channel_1', 'channel_2'], ['channel_2', 'channel_3']]



if __name__ == "__main__":
        
        parser = argparse.ArgumentParser()

        parser.add_argument("-area",
                            help="the area you want to analyse",
                            type=str,
                            required=True)

        parser.add_argument("-types",
                            help="undrained or drained",
                            type=str,
                            required=True)

        parser.add_argument("-resume",
                            help="resume or not",
                            type=str,
                            required=False)

        parser.add_argument("-ocv",
                            help="override cv",
                            type=bool,
                            required=False)

        # parse
        args = parser.parse_args()

        config["FEATURE_SAVE_FOLDER"] = area_manager[args.area]["INPUT_SELECTION_2_STAGE"] + f"_{args.types.upper()}"


        if config["NOGPU"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
            for i in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[i], True)

        #current_masterfile = area_manager[args.area][f"MASTERFILE_BEST_ONLY_{args.types.upper()}"]
        config["DATASET_PATH"] = area_manager[args.area]["DATASET_PATH"][args.types.upper()]
        main_is_path = area_manager[args.area]["INPUT_SELECTION_2_STAGE"] + f"_{args.types.upper()}"    

        if args.area == "TESTAREA":
            area = "TestArea"
        else:
            area = args.area

        # load the data 1 time only
        input_data, total_number_of_samples, total_number_of_windows = load_data_and_configuration(current_masterfile=config['DATASET_PATH'])

        best_dictionary = load_the_best_combination_dictionary(current_results_path=os.path.join("..", area, "Results", f"input_selection_first_stage_{args.types.upper()}", f"summary_{len(input_data.keys())}.csv"),
                                                               channel_list=input_data.keys())

        
        if config["STDOUT"]:
            for channel in input_data:
                for element in input_data[channel]:
                    print(f'[WALKING IN] channel {channel} -- {element.stringify()}')
        
        for round_number in range(len(channel_permutationms)):
            print(f"[ROUND] We are on round {round_number}/{len(channel_permutationms)}")

            accuracy_of_the_round = -1 
            holder_for_temporary_best = {}

            # sivous 
            for channel in input_data.keys():
                if channel not in best_dictionary.keys(): continue
                
                # get the channel 
                for gt_element in input_data[channel][:]:
                    for best_element in best_dictionary[channel]:
                        checker = AnalyseTheInputs(element_a=gt_element, element_b=best_element)
                        if(checker.similarity_check()):
                            gt_element.is_best = True
                            input_data[channel].remove(gt_element)
                


            for index in range(len(channel_permutationms)):
                # get the currents channel to anaylise
                channel_list = channel_permutationms[index]

                # set the saving path, so I do not bother anymore to change it
                base_path = os.path.join(config["FEATURE_SAVE_FOLDER"], f"Round_{round_number}")
                round_path = os.path.join(config["FEATURE_SAVE_FOLDER"], f"Round_{round_number}", f"possible_combination_{'_'.join(channel_list)}")
                
                # we need labels 
                label_encoded = make_one_hot_encoding(labels=input_data[channel_list[0]][0].label_set)

                # run all possible inputs
                if len(input_data[channel_list[0]]) == 0 and len(input_data[channel_list[1]]): 
                    print(f"[IS] NO eelement in the iterators {len(input_data[channel_list[0]])} or {len(input_data[channel_list[1]])}")

                first_channel = channel_list[0]
                second_channel = channel_list[1]


                # instance of kfold
                kfold = StratifiedKFold(n_splits=config["CV"], shuffle=True)

                if config["STDOUT"]:
                    print(f"[LOAD DATA COMPLETE]")
                    print(f"[CV] Get number of split -> {kfold.get_n_splits(input_data[first_channel][0].data, input_data[first_channel][0].label_set)}")

                for first_channel_iterator in range(len(input_data[channel_list[0]])):
                    
                    # reset co9nfig and train each round 
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
                        "inputs": config["INPUTS"],
                        "output": len(np.unique(input_data[channel_list[0]][0].label_set)),
                        "stride": cnn_fine_tuning["stride"],
                        "regularization": cnn_fine_tuning["regularization"],
                        "data_augmentation": True,
                        'rotation_factor': cnn_fine_tuning["rotation_factor"],
                        "last_activation": 'softmax',
                        "dropout_rate": cnn_fine_tuning["dropout_rate"]
                    }
                                                
                    train_dataset = {}
                    
                    
                    train_dataset, cnn_configuration = prepare_the_sets_for_analysis(train_dataset=train_dataset, 
                                                                    cnn_configuration=cnn_configuration, 
                                                                    current_sample_number=first_channel_iterator,
                                                                    dataset=input_data, 
                                                                    current_channel=first_channel, 
                                                                    best_dictionary=best_dictionary)

                    
                    for second_channel_iterator in range(len(input_data[channel_list[1]])):

                        train_dataset, cnn_configuration = prepare_the_sets_for_analysis(train_dataset=train_dataset, 
                                                                                        cnn_configuration=cnn_configuration, 
                                                                                        current_sample_number=second_channel_iterator,
                                                                                        dataset=input_data, 
                                                                                        current_channel=second_channel, 
                                                                                        best_dictionary=best_dictionary)
                        
                        image_name = f"{input_data[first_channel][first_channel_iterator].relative_path.split('/')[-1]}_{input_data[second_channel][second_channel_iterator].relative_path.split('/')[-1]}"

                        if input_data[first_channel][first_channel_iterator].relative_path == input_data[second_channel][second_channel_iterator].relative_path:
                            print(f"[CNN] No necessary to make  a run with same inputs")
                        
                        if config["STDOUT"]:
                            print(f"[CNN CONFIGURATION] CNN configuration loaded {cnn_configuration}")


                        # create the folder structure like before
                        saving_path, predictions_path, satellite_path, weights_folder = make_the_folder_work(feature_save_folder=os.path.join(round_path, image_name))
                        #print(saving_path, predictions_path, satellite_path, weights_folder)
                        
                        cross_timer = time.time()
                        
                        if config["NOGPU"]:

                            out = Parallel(n_jobs=int(config["CV"]))(
                                delayed(do_the_fitting_multiprocessing_v2)(configuration=config,
                                                                        current_cnn_config=cnn_configuration,
                                                                        current_channel_list=channel_list,
                                                                        encoded_labels=label_encoded,
                                                                        training_data=train_dataset,
                                                                        training_idx=train_idx,
                                                                        validation_idx=validation_idx,
                                                                        ) for train_idx, validation_idx in kfold.split(input_data[first_channel][0].data, input_data[first_channel][0].label_set)
                            )
                        else:

                            out = list()
                            for train_idx, validation_idx in kfold.split(input_data[first_channel][0].data, input_data[first_channel][0].label_set):
                                print(f"[GPU] I am with the GPUs")
                                o = do_the_fitting_multiprocessing_v2(configuration=config,
                                                                    current_cnn_config=cnn_configuration,
                                                                    current_channel_list=channel_list,
                                                                    encoded_labels=label_encoded,
                                                                    training_data=train_dataset,
                                                                    training_idx=train_idx,
                                                                    validation_idx=validation_idx)
                                out.append(o)

                        print(f"[SAVE THE CV] Saving CV from mult processing")
                        list_of_scored_accuracy, list_of_train_accuracy = output_list_converter(output_list=out,
                                                                                                saving_path=saving_path,
                                                                                                predictions_path=predictions_path,
                                                                                                satellite_path=satellite_path,
                                                                                                real_label=input_data[first_channel][0].label_set,
                                                                                                configuration=config)

                        print(f"[TIME] End of cross validation {time.time() - cross_timer} s.")

                        do_the_final_report_v2(saving_path=os.path.join(main_is_path, "Accuracy_scored_list.csv"),
                                               channel_list=channel_list,
                                               path_iterator_i=input_data[f"{first_channel}"][first_channel_iterator].relative_path,
                                               path_iterator_ii=input_data[f"{second_channel}"][second_channel_iterator].relative_path,
                                               mean_of_scored_accuracy=np.mean(list_of_scored_accuracy),
                                               verbose=config["VERBOSE"])                       

                        if accuracy_of_the_round  < np.mean(list_of_scored_accuracy):
                            holder_for_temporary_best[np.mean(list_of_scored_accuracy)] = [input_data[first_channel][first_channel_iterator], input_data[second_channel][second_channel_iterator]] 
                            accuracy_of_the_round = np.mean(list_of_scored_accuracy)

                        if config["STDOUT"]:
                            print(f'[BEST ROUND COMBINATION] {holder_for_temporary_best}') 


            #  here I get the best 
            index_of_the_best_combi = max(list(holder_for_temporary_best.keys()))
            round_best_combination = holder_for_temporary_best[index_of_the_best_combi]
            
            print(f"[BEST] Best combination processing")
            for c in channel_list:
                list_to_analyse = list() 
                # extract the channel from round list 
                for i in round_best_combination:
                    if i.channel == c:
                        list_to_analyse.append(i) 

                #current_channel_wise_list = best_dictionary[c]
                current_channel_wise_list = [obj.relative_path for obj in best_dictionary[c]]
                
                for element in list_to_analyse:
                    if element.relative_path not in current_channel_wise_list:
                        print(f"[ADDING] Adding new element to the list")
                        best_dictionary[c].append(element)
                     

                print(f"[BEST] Cleaning data")
                for gt_element in input_data[c]:
                    for best_element in best_dictionary[c]:

                        checker = AnalyseTheInputs(element_a=gt_element, 
                                                   element_b=best_element)
                        
                        if(checker.similarity_check()):
                            gt_element.is_best = True
                            input_data[c].remove(gt_element)


            # here summary
            write_summary_v2(path=os.path.join(main_is_path, "summary_best.txt"),
                             best_dictionary=best_dictionary,
                             mean_of_accuracy=index_of_the_best_combi)
             
    
            print(f"[END ROUND] New best disctionary is {best_dictionary}")


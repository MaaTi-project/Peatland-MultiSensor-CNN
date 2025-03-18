import time
import numpy as np 
import argparse

import os 
import sys 

# Get the absolute path of the project root (go up one level from the current script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import tensorflow as tf 
from Configuration.configuration import config, cnn_fine_tuning
from NNFactory.neural_network_factory import create_multi_input_network 
from Functions.InputSelection_utilities import (stack_inputs,
                                                load_best_dictionary_from_input_selection_results,
                                                make_one_hot_encoding)


training_timer = time.time()

print(f"[TRAIN THE BEST MODEL]")
# rest cnn each channel change
config["INPUTS"] = 3
input_data = {}


if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser()
        
        parser.add_argument("-area", 
                            help="area to analyse", 
                            type=str, 
                            required=True)

        parser.add_argument("-types", 
                            help="drain or undrained", 
                            type=str, 
                            required=True)
                
        parser.add_argument("-ocv", 
                            help="ocv", 
                            type=int, 
                            required=False)
        
        parser.add_argument("-override_channel", 
                            help="ovveride channel size", 
                            required=False)

        # parse
        args = parser.parse_args()

        if args.area == "TESTAREA":
            area = "TestArea"
        else:
            area = args.area

        BEST_DICTIONARY_PATH = os.path.join("..", area, "Results", f"input_selection_second_stage_{args.types.upper()}", "summary_best.txt") 

        if args.ocv:
            config["CV"] = args.ocv
        else:
            config["CV"] = 0

        if args.override_channel == "SI":
            override = True
            config["channel_1"] = 5
            config["channel_2"] = 5
            config["channel_3"] = 5
            config["channel_4"] = 5
            config["channel_5"] = 1
        else:
            override = False

        best_combination = load_best_dictionary_from_input_selection_results(input_selection_results_path=BEST_DICTIONARY_PATH)
        best_combination, label_set = stack_inputs(best_combination=best_combination)

        config["INPUTS"] = len(best_combination.keys())
        
        satellite_path = os.path.join("..", area,  "Results", "BestCombination", args.types.lower(), "confusion_matrix", "predictions")
        save_weight = os.path.join("..", area, "Results", "BestCombination", args.types.lower(), "weights")

        if not os.path.exists(f'{satellite_path}'):
            os.makedirs(f'{satellite_path}')

        if not os.path.exists(f'{save_weight}'):
            os.makedirs(f'{save_weight}')

        # rest cnn each channel change
        cnn_configuration = {
            "input_channel_1": None,
            "input_channel_3": None,
            "input_channel_2": None,
            'input_channel_4': None,
            'input_channel_5': None,
            "neuron_numbers": cnn_fine_tuning["neurons_numbers"],
            "kernel_channel_1": None,
            "kernel_channel_3": None,
            "kernel_channel_2": None,
            "kernel_channel_4": None,
            "kernel_channel_5": None,
            "inputs": config["INPUTS"],
            "output": None,
            "stride": cnn_fine_tuning["stride"],
            "regularization": cnn_fine_tuning["regularization"],
            "data_augmentation": True,
            'rotation_factor': cnn_fine_tuning["rotation_factor"],
            "last_activation": 'softmax',
            "dropout_rate": cnn_fine_tuning["dropout_rate"]
        }

        for channel in list(best_combination.keys()):

            input_data[f"input_{channel}"] = best_combination[channel]
            dummy_data_to_prepare_cv_split = input_data[f"input_{channel}"][0]
            # configure the cnn params
            cnn_configuration[f"input_{channel}"] = (input_data[f"input_{channel}"].shape[1], input_data[f"input_{channel}"].shape[2], input_data[f"input_{channel}"].shape[3])
            cnn_configuration[f"kernel_{channel}"] = (input_data[f"input_{channel}"].shape[3], input_data[f"input_{channel}"].shape[3])

            if config["STDOUT"]:
                print(f'[INPUT FOR CHANNEL] {channel} {input_data[f"input_{channel}"].shape}')
                print(f"[CNN CONFIG] {cnn_configuration}")

        # prepareing the encoding label
        cnn_configuration['output'] = len(np.unique(label_set))
        encoded_labels = make_one_hot_encoding(labels=label_set)

        print(f"[CNN] The CNN is now configured {cnn_configuration} for best combination")
        compiled_model = create_multi_input_network(**cnn_configuration)

        # compile the model
        compiled_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"[CNN] The CNN is now configured {cnn_configuration}")

        history = compiled_model.fit(
            input_data,
            encoded_labels,
            epochs=config["EPOCHS"],
            verbose=config["VERBOSE"],
            batch_size=config["BATCH_SIZE"])

        # save the model
        compiled_model.save(os.path.join(save_weight, f"best_model_{args.types.lower()}.keras"))
            
        print(f"[TIMER] Best model timer {time.time() - training_timer}")
    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception {ex}")

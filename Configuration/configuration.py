import tensorflow as tf
import os

area_manager = {
    "TESTAREA": {
        "INPUT_SELECTION_1_STAGE": os.path.join("..", "TestArea", "Results", "input_selection_first_stage"),
        "INPUT_SELECTION_2_STAGE": os.path.join("..", "TestArea", "Results", "input_selection_second_stage"),
        "DATASET_PATH": {
            "DRAINED": os.path.join("..", "Windows", "TestArea", "drained"),
            "UNDRAINED": os.path.join("..", "Windows", "TestArea", "undrained"),
        },
        "FEATURES_PATH": os.path.join("..", "TestArea", "Results"),

        "ANNOTATIONS": {
                "DRAINED": os.path.join("..", "Annotations", "TestArea", "Dataset", "annotation_drained.csv"),
                "UNDRAINED": os.path.join("..", "Annotations", "TestArea", "Dataset", "annotation_undrained.csv"),
        },

        "PATH_TO_BEST_MODEL": os.path.join("..", "TestArea", "BestModel"),

    },
}


config = {
    "BATCH_SIZE": 128,
    "EPOCHS": 1,
    "CV": 5,
    "VERBOSE": 0,
    "STDOUT": True,
    "CONFUSION": False,
    "PLOT": False,
    "STACKING": True,
    "NOGPU": True,
    "BEST_MODEL": False,
    "WRITE_ANNOTATION_SUMMARY": False,
    "ONLY": None,
    "INPUTS": 2,
    "FEATURE_SAVE_FOLDER": "",
    "DATASET_PATH": "",
    "FEATURES_PATH": "",
    "FEATURES_SUMMARY": "",
    "SAVING_PATH": "",
    "START_ROUND": 0,
    "TOTAL_ROUNDS": 24,
    "channel_1": 5,
    "channel_2": 25,
    "channel_3": 5,
    "channel_4": 50,
    "channel_5": 1,
    "OFFSET": 0.0,
    "no_data_value": 255,
    "channel_list": ["channel_1", "channel_2", "channel_3", "channel_4", "channel_5"],

}

input_selection_combination = {
    "1_input": [['channel_2'], ['channel_3'], ['channel_1'], ["channel_4"], ["channel_5"]],
    "2_inputs": [['channel_1', 'channel_2'], ['channel_2', 'channel_3'], ['channel_1', 'channel_3'],
                 ['channel_1', 'channel_4'], ['channel_1', 'channel_5'], ['channel_2', 'channel_4'],
                 ['channel_2', 'channel_5'], ['channel_3', 'channel_4'], ['channel_3', 'channel_5']],
    "multiples": ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5']
}

cnn_fine_tuning = {
    "neurons_numbers": 60,
    "stride": (1, 1),
    "regularization": tf.keras.regularizers.L2(0.06),
    "rotation_factor": (-0.2, 0.5),
    "dropout_rate": 0.1
}

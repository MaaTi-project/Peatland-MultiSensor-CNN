# 🚀 Configuration

we write here an intro 

## 📝 Area manager

Here you can find pathes to inportant file used in the analysis. They are all in one place in order to make the execution of the algorithm easier. 

```python

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


```

## 📝 Configuration

Others sets of parameters used across all the algorithm. 

```python

config = {
    "BATCH_SIZE": 128,
    "EPOCHS": 500,
    "CV": 5,
    "VERBOSE": 1,
    "STDOUT": True,
    "CONFUSION": False,
    "PLOT": False,
    "STACKING": True,
    "NOGPU": True,
    "BEST_MODEL": False,
    "WRITE_ANNOTATION_SUMMARY": False,
    "ONLY": None,
    "INPUTS": 3,
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

```

<table align="center">
  <tr>
    <th>Parameter</th>
    <th>Values</th>
  </tr>
  <tr>
    <td>BATCH_SIZE</td>
    <td>Batch size used in each iteration of the model</td>
  </tr>
  <tr>
    <td>EPOCHS</td>
    <td>Number of epochs during training phase of the model</td>
  </tr>
  <tr>
    <td>CV</td>
    <td>Number of fold to use in cross validation</td>
  </tr>
   <tr>
    <td>VERBOSE</td>
    <td>Add verbosity during training and prediction phases</td>
  </tr>
    <tr>
    <td>STDOUT</td>
    <td>Add more verbosity to the algorithm</td>
  </tr>
   <tr>
    <td>CONFUSION</td>
    <td>generate confusion matrices on place.</td>
  </tr>
   <tr>
    <td>PLOT</td>
    <td>Plot history of training</td>
  </tr>
   <tr>
    <td>STACKING</td>
    <td>Used in input selection to stack band-wise input</td>
  </tr>
     <tr>
    <td>NOGPU</td>
    <td>Use GPU or multi-processing during the execution of the algorithm</td>
  </tr>
   <tr>
    <td>BEST_MODEL</td>
    <td>During each training fold create a temporary best model for further analysis</td>
  </tr>
   <tr>
    <td>WRITE_ANNOTATION_SUMMARY</td>
    <td>For each windows write its metadata</td>
  </tr>
   <tr>
    <td>ONLY</td>
    <td>Deprecated parameters</td>
  </tr>
     <tr>
    <td>INPUTS</td>
    <td>Force number of CNN inputs intead of dynamically allocate it</td>
  </tr>
       <tr>
    <td>FEATURE_SAVE_FOLDER</td>
    <td>Placeholder to that assign features path during the algorithm execution.</td>
  </tr>
  <tr>
    <td>DATASET_PATH</td>
    <td>Placeholder to that assign dataset path during the algorithm execution.</td>
  </tr>
   <tr>
    <td>FEATURES_PATH</td>
    <td>Use this parameter to assign a path outside the root folder of this code</td>
  </tr>
   <tr>
    <td>FEATURES_SUMMARY</td>
    <td>Placeholder to the path of feature summary</td>
  </tr>
   <tr>
    <td>SAVING_PATH</td>
    <td>Placeholder used as main saving folder</td>
  </tr>
    <tr>
    <td>START_ROUND</td>
    <td>If you want to resume the input selection form different round than the staring one</td>
  </tr>
   <tr>
    <td>TOTAL_ROUNDS</td>
    <td>Override total number of round during input selection (deprecated)</td>
  </tr>
     <tr>
    <td>channel_*</td>
    <td>Windows dimension of each channel isa not overrided</td>
  </tr>
       <tr>
    <td>OFFSET</td>
    <td>Set offset during windows creation</td>
  </tr>
   <tr>
    <td>no_data_value</td>
    <td>No data value assigned to windows and rasters</td>
  </tr>
     <tr>
    <td>channel_list</td>
    <td>Name of the channels to use</td>
  </tr>


</table>



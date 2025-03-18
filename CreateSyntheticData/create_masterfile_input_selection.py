import os
import numpy as np

def do_input_selections_masterfile(path, windows_size, type_of_annotations):
    windows_size = 5
    types_annotations = "undrained"
    handler = open(path, 'w')

    for types in os.listdir(os.path.join("..", "Windows", "TestArea", type_of_annotations)):
        #for raster in os.listdir(os.path.join("..", "Windows", "TestArea", types_annotations, types)):
        fullpath = os.path.join("..", "Windows", "TestArea", type_of_annotations, types)

        if types == "Derived":
            row = f"{fullpath},{windows_size},channel_3"
        elif types == "Optical":
            row = f"{fullpath},{windows_size},channel_1"
        else:
            row = f"{fullpath},{windows_size},channel_2"

        handler.write(f"{row}\n")

    handler.close()



if __name__ == '__main__':
    types_annotations="undrained"
    do_input_selections_masterfile(path=os.path.join("..", "Configuration", "Masterfiles", f"Masterfile_test_area_{types_annotations}.csv"),
                                   windows_size=5,
                                   type_of_annotations=types_annotations)
    
    types_annotations="drained"
    do_input_selections_masterfile(path=os.path.join("..", "Configuration", "Masterfiles", f"Masterfile_test_area_{types_annotations}.csv"),
                                windows_size=5,
                                type_of_annotations=types_annotations)




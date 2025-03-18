import os
import numpy as np
import argparse


ROW = 20
COL = 20

if __name__ == '__main__':
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("-folder", 
                            help="Insert the output folder", 
                            type=str, 
                            required=True)
        
        args = parser.parse_args()

        PATH_TO_OUTPUT = os.path.join(args.folder)

        row_undrained = None
        row_drained = None

        if not os.path.exists("save_arrays"):
            os.makedirs("save_arrays")

        for row_worker in range(ROW):
            worker_folder = os.path.join(PATH_TO_OUTPUT, f"worker_{row_worker}")
            print(f"[WALK] Walking into {worker_folder}")
            # set the col None
            column_undrained = None
            column_drained = None

            for col_worker in range(COL):
                # do undrained
                if column_undrained is None:
                    column_undrained = np.load(os.path.join(f'{worker_folder}', f"worker_{row_worker}_{col_worker}_undrained.npy"))
                else:
                    column_undrained = np.concatenate((column_undrained, np.load(os.path.join(f'{worker_folder}', f"worker_{row_worker}_{col_worker}_undrained.npy"))), axis=1)

                # do drained
                if column_drained is None:
                    column_drained = np.load(os.path.join(f'{worker_folder}', f"worker_{row_worker}_{col_worker}_drained.npy"))
                else:
                    column_drained = np.concatenate((column_drained, np.load(os.path.join(f'{worker_folder}', f"worker_{row_worker}_{col_worker}_drained.npy"))), axis=1)

            # rebuilt the rows undrained
            if row_undrained is None:
                row_undrained = column_undrained
            else:
                row_undrained = np.concatenate((row_undrained, column_undrained))

            # rebuilt the rows undrained
            if row_drained is None:
                row_drained = column_drained
            else:
                row_drained = np.concatenate((row_drained, column_drained))

        print(f"[BUILD COMPLETE] The drained shape is {row_drained.shape} and the undrained is {row_undrained.shape}")
        np.save(os.path.join(f"saved_arrays", f'raster_drained.npy'), row_drained)
        np.save(os.path.join(f"saved_arrays", f'raster_undrained.npy'), row_undrained)

    except Exception as ex:
        print(f"[EXCEPTION] Main throws exception {ex}")



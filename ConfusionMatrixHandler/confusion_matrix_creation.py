#!/usr/bin/env python

from Functions.functions import *
from sklearn.metrics import accuracy_score
import argparse
import pandas as pd

if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--folder', required=True, help="path of the folder to analyse")
        parser.add_argument('--type', required=False, help="csv for csv file or png for png all for both")
        args = parser.parse_args()

        stacked_true, stacked_predicted = None, None
        cv_folds = 5

        if args.folder is not None:
            folder = args.folder
        else:
            folder = os.path.join("..", "Results", "test_one_input")

        if args.type == "all":
            PNG = True
            CSV = True
        elif args.type == "csv":
            CSV = True
            PNG = False
        elif args.type == "png":
            CSV = False
            PNG = True
        else:
            CSV = False
            PNG = False

        print(f'[FOLDER] Walking in {folder}')

        # read the id from the folder and all the folds arrays
        labels_id = np.load(os.path.join(folder, "predictions", "labels_id.npy"))

        for i in range(cv_folds):

            if i == 0:
                stacked_true = np.load(os.path.join(folder, "predictions", f"y_true_{i}.npy"))
                stacked_predicted = np.load(os.path.join(folder, "predictions", f"y_pred_{i}.npy"))
            else:
                stacked_true = np.concatenate(stacked_true, np.load(os.path.join(folder, "predictions", f"y_true_{i}.npy")))
                stacked_predicted = np.concatenate(stacked_predicted, np.load(os.path.join(folder, "predictions", f"y_pred_{i}.npy")))


        if CSV:
            cm = confusion_matrix(stacked_true, stacked_predicted)
            df = pd.DataFrame(cm, columns=[sorted(labels_id)], index=sorted(labels_id))
            df.to_csv(f"{folder}/cm_{accuracy_score(stacked_true, stacked_predicted)}.csv")

        if PNG:
            cm = confusion_matrix(stacked_true, stacked_predicted)
            plot_new_cm(labels_id=sorted(labels_id),
                        y_true=stacked_true,
                        y_pred=stacked_predicted,
                        path=f"{folder}/confusion_matrix.png",
                        already_made_cm=cm,
                        title=f" ",
                        normalize=False,
                        accuracy=accuracy_score(stacked_true, stacked_predicted))
    except Exception as ex:
        print(f'[EXCEPTION] Cm creator throws exception {ex}')
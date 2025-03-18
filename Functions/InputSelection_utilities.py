from itertools import cycle
from NNFactory.neural_network_factory import *
from Functions.Utilities import *
from Configuration.input_selection_placeholder import *


def load_data_and_configuration(current_masterfile):
    try:
        
        input_data = {}            
        total_number_of_samples = 0

        for path in os.listdir(current_masterfile):
            
            current_satellite_name = path.split('_')[0]
            current_channel = f"{path.split('_')[1]}_{path.split('_')[-1]}"

            if not current_channel in input_data.keys():
                input_data[current_channel] = list() 
            
            full_path = os.path.join(current_masterfile, path)
            print(f"[FULL PATH {current_channel}] full path is  {full_path}")


            for image_path in os.listdir(full_path):
                full_image_path = os.path.join(full_path, image_path)

                temp_list = list()
                label_set = list()
                windows_ctn = 0
                
                for windows in os.listdir(f'{full_path}/{image_path}'):
                
                    label_set.append(int(windows.split('_')[1]))
                    temp = np.load(f'{full_path}/{image_path}/{windows}')
                    #print(temp.shape)
                    temp_list.append(temp)
                    

                concatenated = np.array(temp_list) 
                windows_ctn = concatenated.shape[0]
                input_data[current_channel].append(InputSelectionPlaceHolder(channel=current_channel,
                                                                            id=total_number_of_samples,
                                                                            full_path=full_image_path,
                                                                            data=concatenated,
                                                                            satellite_name=current_satellite_name,
                                                                            image_name=image_path,
                                                                            label_set=label_set,
                                                                            shape=concatenated.shape))
                total_number_of_samples += 1
        return input_data, total_number_of_samples, windows_ctn
    except Exception as ex:
        print(f"[EXCEPTION] Read the masterfile throws exception {ex}")
        print(traceback.print_exc()) 


def read_the_master_file(path):
    try:
        collection_of_images = {}
        input_data = {}
        
        for line in os.listdir(path):
            print(line.strip().split('_'))
            current_image_name = line.strip().split('_')[0]
            current_channel = f"{line.strip().split('_')[1]}_{line.strip().split('_')[-1]}"

            if current_channel is not None and current_channel not in collection_of_images.keys():
                collection_of_images[current_channel] = list() 
            
            if current_channel is not None and current_channel not in input_data.keys():
                input_data[current_channel] = list() 

            for tif_file in os.listdir(os.path.join(path, line)):
                full_tif_path = os.path.join(os.path.join(path, line, tif_file))

                print(f"[CHANNEL {current_channel}] Loaded {full_tif_path}")
                collection_of_images[current_channel].append(full_tif_path)


        return collection_of_images
    except Exception as ex:
        print(f"[EXCEPTION] read the master file throws exception {ex}")
        return None


def prepare_data(path, image_size):
    try:
        train_set, label_set, real_label = list(), list(), list()

        for window in os.listdir(path=path):
            label_set.append(int(window.split('_')[1]))
            real_label.append(int(window.split('_')[1]))

            # load the data
            train_set.append(np.load(f'{path}/{window}'))

        # transform the list to array
        train_set = np.array(train_set).astype('float')
        label_set = np.array(label_set).astype('int')

        train_set = train_set.reshape((len(train_set), image_size, image_size, -1))
        train_set = train_set / 255.0

        return train_set, label_set, np.unique(real_label)

    except Exception as ex:
        print(f"Prepare data [EXCEPTION] {ex}")


def make_one_hot_encoding(labels):
    try:
        # to categorical
        enc = OneHotEncoder(handle_unknown='ignore')
        # train and valid set
        temp = np.reshape(labels, (-1, 1))
        label_encoded = enc.fit_transform(temp).toarray()
        print(f'[ONE HOT ENCODING] Labels are one-hot-encoded: {(label_encoded.sum(axis=1) - np.ones(label_encoded.shape[0])).sum() == 0}')
        return label_encoded
    except Exception as ex:
        print(f"[EXCEPTION] Make one hot encoding throws exception {ex}")


def calculate_features_number(path):
    try:
        # the first step is to check how many channel the network has
        handler = open(path)
        holder = list()
        temp = list()
        for line in handler.readlines():
            if line.strip().split(',')[2] not in holder:
                temp.append([line.strip().split(',')[2]])
                # add an holder just to keep only one channel into the list
                holder.append(line.strip().split(',')[2])

        return temp

    except Exception as ex:
        print(f"[EXCEPTION] Calculate the features number throws exception {ex}")


def load_the_best_list(best_list, image_shape, config, data_configuration, current, verbose=False):
    try:

        if best_list is None:
            return np.zeros((0, 0, 0, 0)), 0

        # filter the channels
        list_of_operation = list()

        for b in best_list:
            list_of_operation.append(b)

        if len(list_of_operation) > 0:
            # start concatenation
            concatenated = np.zeros((image_shape[0], image_shape[1], image_shape[2], 0))
            
            for best in list_of_operation:
                ts, ls = list(), list()
                if verbose:
                    print(f'[BEST] -> {config["DATASET_PATH"]}/{best}')
                for ctn, win in enumerate(os.listdir(f'{config["DATASET_PATH"]}/{best}')):
                    ts.append(np.load(f'{config["DATASET_PATH"]}/{best}/{win}'))
                    ls.append(int(win.split('_')[1]))

                ts = np.array(ts).astype('float')
                ls = np.array(ts).astype('int')
                ts = ts.reshape((len(ts), image_shape[1], image_shape[2], -1))
                ts = ts / 255.0

                concatenated = np.concatenate((concatenated, ts), axis=3)
            # print(f'[STACKED] -> {concatenated.shape}')
            return concatenated, ls
        else:
            if verbose:
                print(f'[NO NEED TO STACK]')
            return np.zeros((0, 0, 0, 0)), 0

    except Exception as ex:
        print(f"load best list throws exception {ex}")


def make_the_folder_work(feature_save_folder, verbose=False):
    try:
        saving_path = os.path.join(feature_save_folder, "weights")
        predictions_path = os.path.join(feature_save_folder, "predictions")
        satellite_path = os.path.join(feature_save_folder)
        weights_folder = os.path.join(feature_save_folder, "weights")

        # create necessary folders to save our stuff
        if not os.path.exists(f'{feature_save_folder}'):
            os.makedirs(f'{feature_save_folder}')

        if not os.path.exists(satellite_path):
            os.makedirs(satellite_path)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)

        if verbose:
            # remove old weights so we do not get confused
            print(f"[REMOVE] Removing old weights from the folder")
        for fil in os.listdir(weights_folder):
            if os.path.exists(f"{weights_folder}{fil}"):
                os.remove(f"{weights_folder}{fil}")

        if verbose:
            print(f"[REMOVE] Removing old prediction from the folder")
        for fil in os.listdir(predictions_path):
            if os.path.exists(f"{predictions_path}/{fil}"):
                os.remove(f"{predictions_path}/{fil}")

        if verbose:
            print(f"[REMOVE] Removing all the old images")
        os.system(f'rm -rf {satellite_path}/*.png')
        os.system(f'rm -rf {satellite_path}/*.csv')

        return saving_path, predictions_path, satellite_path, weights_folder

    except Exception as ex:
        print(f"[EXCEPTION] Make the folder work throws exception {ex}")


def do_the_fitting_multiprocessing(configuration, current_cnn_config, encoded_labels, channel_list,
                                   input_channel_1, input_channel_2, input_channel_3, best_data,
                                   train_index, validation_index, input_channel_4=None, input_channel_5=None):
    try:
        
        compiled_model = create_multi_input_network(**current_cnn_config)
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        # compile the model
        compiled_model.compile(optimizer=opt, loss=tf.keras.metrics.categorical_crossentropy, metrics=['accuracy'])
        
        training_data = {}
        validation_data = {}
        for channel_number, channel in enumerate(channel_list):
            
            if channel == "channel_1":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:

                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_1[train_index]), best_data[channel][train_index]), axis=3)
                    validation_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_1[validation_index]), best_data[channel][validation_index]), axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_1[train_index])
                    validation_data[f"input_{channel}"] = np.array(input_channel_1[validation_index])

            if channel == "channel_2":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:

                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_2[train_index]), best_data[channel][train_index]), axis=3)
                    validation_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_2[validation_index]), best_data[channel][validation_index]), axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_2[train_index])
                    validation_data[f"input_{channel}"] = np.array(input_channel_2[validation_index])

            if channel == "channel_3":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:

                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_3[train_index]), best_data[channel][train_index]), axis=3)
                    validation_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_3[validation_index]), best_data[channel][validation_index]), axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_3[train_index])
                    validation_data[f"input_{channel}"] = np.array(input_channel_3[validation_index])

            if channel == "channel_4":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:

                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_4[train_index]), best_data[channel][train_index]), axis=3)
                    validation_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_4[validation_index]), best_data[channel][validation_index]), axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_4[train_index])
                    validation_data[f"input_{channel}"] = np.array(input_channel_4[validation_index])

            if channel == "channel_5":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:

                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_5[train_index]), best_data[channel][train_index]), axis=3)
                    validation_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_5[validation_index]), best_data[channel][validation_index]), axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_5[train_index])
                    validation_data[f"input_{channel}"] = np.array(input_channel_5[validation_index])

            if configuration["STDOUT"]:
                print(f'[TRAINING INPUT] -> input_{channel} -> {training_data[f"input_{channel}"].shape}')
                print(f'[VALIDATION INPUT] -> input_{channel} -> {validation_data[f"input_{channel}"].shape}')
                print(f"[LABELS] {encoded_labels.shape}")
                print("---")
        # fit
        history = compiled_model.fit(
            training_data,
            encoded_labels[train_index],
            epochs=configuration["EPOCHS"],
            validation_data=(validation_data, encoded_labels[validation_index]),
            verbose=configuration["VERBOSE"],
            batch_size=configuration["BATCH_SIZE"])

        # score
        scores = compiled_model.evaluate(
            validation_data,
            encoded_labels[validation_index],
            verbose=configuration["VERBOSE"]
        )

        print("[SCORE] Accuracy %s: %.2f%%" % (compiled_model.metrics_names[1], scores[1] * 100))

        # predict
        predictions = compiled_model.predict(validation_data, verbose=configuration["VERBOSE"])
        predictions = np.argmax(predictions, axis=-1)

        # return scored, trained, model, pred, real label and history
        return [scores[1] * 100, np.mean(history.history['accuracy']), compiled_model, predictions,
                np.argmax(encoded_labels[validation_index], axis=-1), history]

    except Exception as ex:
        print(f"[EXCEPTION] Do the fitting throws exception {ex}")
        print(traceback.print_exc())


def output_list_converter(configuration, output_list, saving_path, predictions_path, satellite_path, real_label):
    try:
        list_of_scored_accuracy, list_of_train_accuracy = list(), list()
        for fold_counter, el in enumerate(output_list):
            # add scored acc of each fold
            list_of_scored_accuracy.append(el[0])
            # add train acc each fold
            list_of_train_accuracy.append(el[1])
            # model save
            el[2].save(os.path.join(saving_path, f"fold_{fold_counter}_acc_{el[0]}_model.keras"))
            # save real label
            np.save(os.path.join(predictions_path, f"y_true_{fold_counter}.npy"), el[4])
            # save pred labels
            np.save(os.path.join(predictions_path, f"y_pred_{fold_counter}.npy"), el[3])

            if configuration["PLOT"]:
                # plot the CNN
                tf.keras.utils.plot_model(el[2],
                                          to_file=os.path.join(satellite_path, "multiples_input_model_plot.png"),
                                          show_shapes=True,
                                          show_layer_names=True)

                plot_accuracy(history=el[5],
                              epochs=configuration["EPOCHS"],
                              save_path=os.path.join(satellite_path, f"multiples_input_fold_{fold_counter}_plot.png"),
                              title=f"fold_{fold_counter} ")
        # save ids
        np.save(os.path.join(predictions_path, "labels_id.npy"), np.array(real_label))
        return list_of_scored_accuracy, list_of_train_accuracy

    except Exception as ex:
        print(f"[EXCEPTION] Output list converter throws exception {ex}")
        print(traceback.print_exc())


def do_best_model(configuration, weights_folder, saving_path, channel_list, input_channel_1, input_channel_2,
                  input_channel_3, best_data, encoded_labels):
    try:
        # get the best model
        model_path_to_load = get_better_fold(path=weights_folder)
        # after I created the mini raster I load the model:
        model = tf.keras.models.load_model(model_path_to_load)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        training_data = {}
        # validation_data = {}
        for channel_number, channel in enumerate(channel_list):

            # for each channel prepare  the dict of inputs
            if channel == "channel_1":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:
                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_1), best_data[channel]),
                                                                       axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_1)

            if channel == "channel_2":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:
                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_2), best_data[channel]),
                                                                       axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_2)

            if channel == "channel_3":
                if configuration["STACKING"] and best_data[channel].shape[3] != 0:
                    training_data[f"input_{channel}"] = np.concatenate((np.array(input_channel_3), best_data[channel]),
                                                                       axis=3)
                else:
                    training_data[f"input_{channel}"] = np.array(input_channel_3)

            if configuration["STDOUT"]:
                print(f'[TRAINING INPUT] -> {channel} -> {training_data[f"input_{channel}"].shape}')
                print("---")

        # fit the model
        history = model.fit(
            training_data,
            encoded_labels,
            epochs=configuration["EPOCHS"],
            verbose=configuration["VERBOSE"],
            batch_size=configuration["BATCH_SIZE"]
        )

        # save the model
        model.save(os.path.join(saving_path, "best_model.h5"))

        return history
    except Exception as ex:
        print(f"[EXCEPTION] Do best model throws exception {ex}")


def do_final_report(path, list_of_scored_accuracy, list_of_train_accuracy, history_full_training, combination_name,
                    type_of_score="scored", verbose=False):
    try:
        string_train_acc = ""
        string_scored_acc = ""
        sum_train_acc = 0
        sum_scored_acc = 0
        for a, b in zip(list_of_scored_accuracy, list_of_train_accuracy):
            string_scored_acc += f"{a},"
            string_train_acc += f"{b},"
            sum_train_acc += b
            sum_scored_acc += a

        if type_of_score == "scored":
            string_scored_acc += f"{sum_scored_acc / len(list_of_scored_accuracy)}"
            with open(path, 'a') as hander:
                hander.write(f"{combination_name},{string_scored_acc}")
                hander.write('\n')

            if verbose:
                print(f"List of scored accuracy during CV {list_of_scored_accuracy}")
                print(f"List of train accuracy during CV {list_of_train_accuracy}")
                print(f"Mean of scored accuracy during CV {sum_scored_acc / len(list_of_scored_accuracy)}")

        else:
            if history_full_training is not None:
                string_train_acc += f"{sum_train_acc / len(list_of_train_accuracy)},{np.mean(history_full_training.history['accuracy'])}"
            with open(path, 'a') as hander:
                hander.write(f"{combination_name},{string_train_acc}")
                hander.write('\n')

            if verbose:
                print(f"Mean of train accuracy during CV {sum_train_acc / len(list_of_train_accuracy)}")
                print(f"Mean of train accuracy with all data {np.mean(history_full_training.history['accuracy'])}")

    except Exception as ex:
        print(f"[EXCEPTION] Do final report throws exception {ex}")


def make_the_mean(input_list):
    try:
        tot = 0
        for i in range(len(input_list)):
            tot += input_list[i]
        return float(tot) / float(len(input_list))
    except Exception as ex:
        print(f"[EXCEPTION] Make the mean throws exception {ex}")


def get_the_best(path_of_summary):
    try:
        handler = open(path_of_summary)
        images, numbers = list(), list()
        for element in handler.readlines():
            temp = element.strip().split(',')
            numbers.append(float(temp[-1]))
            images.append(element)

        pos = numbers.index(max(numbers))
        return images[pos].split(',')
    except Exception as ex:
        print(f"[EXCEPTION] Get the best best throws exception {ex}")


def do_round_accuracy_for_single(path, channel, current_input, mean_of_accuracy):
        handler = open(path, 'a')
        handler.write(f'{channel},{current_input},{mean_of_accuracy}\n')
        handler.close()

def do_round_summary(path, configuration, current_dict, mean_of_accuracy):
    try:
        handler = open(path, 'a')
        string = ""

        # write the best only if it is stacking
        for key in current_dict:
            string += f"{key},{current_dict[key][0]},{current_dict[key][1]},"

        #print(string)
        handler.write(f'{string}{mean_of_accuracy}\n')
        handler.close()

    except Exception as ex:
        print(f"[EXCEPTION] Write summary throws exception {ex}")


def make_simple_iteration_list(channel_list, input_data):
    try:
        temp = list()
        if len(channel_list) != 1:
            for channel in channel_list:
                temp.append(len(input_data[channel]))
        else:
            for channel in channel_list:
                temp.append(len(input_data[channel]))
                temp.append(1)
        return temp
    except Exception as ex:
        print(f"[EXCEPTION] Make simple iteration list throws exception{ex}")


def get_the_best_from_the_list(collection_of_accuracy, configuration):
    try:
        temp = list()
        for el in collection_of_accuracy:
            temp.append(el[1])
        m = max(temp)
        index = temp.index(m)

        if configuration["STDOUT"]:
            print("----------------------------------------")
            print(f"THE BEST IS {collection_of_accuracy[index]}")

        return collection_of_accuracy[index]
    except Exception as ex:
        print(f"[EXCEPTION] Get the best from the list throws exception{ex}")


def preload_and_stack_best_list(best_dict, dataset_path):
    try:
        list_of_pathes = list()
        besty = {}
        for key, val in best_dict.items():
            concatenated = list()
            for ctn, elements in enumerate(val):
                list_of_pathes.append(f'{dataset_path}/{elements}')
                if ctn == 0:
                    for w in os.listdir(f'{dataset_path}/{elements}'):
                        concatenated.append(np.load(f"{dataset_path}/{elements}/{w}"))
                    concatenated = np.array(concatenated)
                else:
                    temp = list()
                    for w in os.listdir(f'{dataset_path}/{elements}'):
                        temp.append(np.load(f"{dataset_path}/{elements}/{w}"))
                    temp = np.array(temp)
                    concatenated = np.concatenate((concatenated, temp), axis=3)
            #besty[key] = concatenated / 255.
            besty[key] = BestSelectionPlaceHolder(id=0,
                                                  channel=key,
                                                  list_of_full_pathes=list_of_pathes,
                                                  data=concatenated / 255.)
        return besty
    except Exception as ex:
        print(f"[EXCEPTION] Preload_and_stack_best_list throws exception {ex}")


def read_masterfile_v2(current_masterfile):
    try:
        total_number_of_samples = 0
        input_data = list()

        handler = open(current_masterfile)
        for line in handler.readlines():
            templine = line.strip().split(',')
            print(templine)
            for path in os.listdir(templine[0]):
                full_path = path = os.path.join(templine[0], path)

                temp_list = list()
                label_set = list()
                windows_ctn = 0

                for windows in os.listdir(full_path):
                    label_set.append(int(windows.split('_')[1]))
                    temp = np.load(f'{full_path}/{windows}')
                    #print(temp.shape)
                    temp_list.append(temp)
                    

                concatenated = np.array(temp_list) 
                windows_ctn = concatenated.shape[0]
                input_data.append(InputSelectionPlaceHolder(channel=templine[3],
                                                            id=total_number_of_samples,
                                                            full_path=full_path,
                                                            data=concatenated,
                                                            satellite_name=templine[0],
                                                            image_name=templine[1],
                                                            label_set=label_set,
                                                            shape=concatenated.shape))
                total_number_of_samples += 1
        return input_data, total_number_of_samples, windows_ctn
    except Exception as ex:
        print(f"[EXCEPTION] Read the masterfile throws exception {ex}")
        print(traceback.print_exc())


def load_the_data_into_dictionary(path, concatenate):
    try:
        list_of_windows = list()
        for w in os.listdir(f"{path}"):
            list_of_windows.append(np.load(f'{path}/{w}'))

        list_of_windows = np.array(list_of_windows)
        list_of_windows = list_of_windows / 255.

        if concatenate is not None:
            list_of_windows = np.concatenate((list_of_windows, concatenate), axis=3)

        return list_of_windows
    except Exception as ex:
        print(f"[EXCEPTION] Load the data into dictionary throws exception {ex}")


def do_the_fitting_multiprocessing_v2(configuration, current_cnn_config, current_channel_list, encoded_labels,
                                   training_data, training_idx, validation_idx):
    try:
        # compile the model
        compiled_model = create_multi_input_network(**current_cnn_config)
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)

        # compile the model
        compiled_model.compile(optimizer=opt, loss=tf.keras.metrics.categorical_crossentropy, metrics=['accuracy'])
        # get local variables or it won t work
        X_train = {}
        X_test = {}

        for channel in current_channel_list:
            X_train[f"input_{channel}"] = training_data[f"input_{channel}"][training_idx]
            X_test[f"input_{channel}"] = training_data[f"input_{channel}"][validation_idx]

            if configuration["STDOUT"]:
                print(f'[TRAINING INPUT] -> {channel} -> {X_train[f"input_{channel}"].shape}')
                print(f'[VALIDATION INPUT] -> {channel} -> {X_test[f"input_{channel}"].shape}')
                print(f"[LABELS] {encoded_labels.shape}")
                print("---")

        # fit
        history = compiled_model.fit(
            X_train,
            encoded_labels[training_idx],
            epochs=configuration["EPOCHS"],
            validation_data=(X_test, encoded_labels[validation_idx]),
            verbose=0,
            batch_size=configuration["BATCH_SIZE"],
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

        # score
        scores = compiled_model.evaluate(
            X_test,
            encoded_labels[validation_idx],
            verbose=0
        )

        print("[SCORE] Accuracy %s: %.2f%%" % (compiled_model.metrics_names[1], scores[1] * 100))

        # predict
        predictions = compiled_model.predict(X_test, verbose=0)
        predictions = np.argmax(predictions, axis=-1)
        # return scored, trained, model, pred, real label and history
        return [scores[1] * 100, np.mean(history.history['accuracy']), compiled_model, predictions,
                np.argmax(encoded_labels[validation_idx], axis=-1), history]

    except Exception as ex:
        print(f"[EXCEPTION] Do the fitting multiprocessing throws exception {ex}")


def do_the_final_report_v2(saving_path, channel_list, path_iterator_i, path_iterator_ii, mean_of_scored_accuracy, verbose):
    try:
        with open(saving_path, 'a') as hander:
            hander.write(f"{channel_list[0]},{path_iterator_i},{channel_list[1]},{path_iterator_ii},{mean_of_scored_accuracy}")
            hander.write('\n')

        if verbose:
            print(f"[CURRENT COMBINATION] {channel_list[0]},{path_iterator_i},{channel_list[1]},{path_iterator_ii},{mean_of_scored_accuracy}")
    except Exception as ex:
        print(f"[EXCEPTION] Do the final report v2 throws exception {ex}")


def write_summary_v2(path, best_dictionary, mean_of_accuracy):
    try:
        h = open(f'{path}', 'a')
        for key in best_dictionary.keys():
            for element in best_dictionary[key]:
                h.write(f"{key},{element.full_path}\n")
                h.write("+\n")

        h.write("=\n")
        h.write(f"Mean of cv accuracy {mean_of_accuracy}\n")

        h.close()
    except Exception as ex:
        print(f"[EXCEPTION] Write the summary v2 throws exception {ex}")


def preload_and_stack_best_list_v2(configuration, best_dict):
    try:
        best_dictionary = {}
        ctn = 0
        for key, val in best_dict.items():
            print(key)
            # check the number of samples
            samples_number = 0
            for i in os.listdir(os.path.join(f"{best_dict[key][0]}")):
                samples_number += 1

            concatenated = None

            print(f'[LOG] {key} -> {best_dict[key]}')

            temp_list = best_dict[key]

            # loop the full list of path
            for el in temp_list:
                windows_list = list()
                dir = f"{el}"
                # inside the list load all the windows
                for w in os.listdir(dir):
                    temp = np.load(f"{dir}/{w}")
                    temp = temp / 255.
                    windows_list.append(temp)

                if concatenated is None:
                    concatenated = np.array(windows_list)
                else:
                    concatenated = np.concatenate((concatenated, np.array(windows_list)), axis=3)

            print(f"[RESUMING IS DONE] I am resuming the channel {key} with shape {concatenated.shape}")

            best_dictionary[key] = BestSelectionPlaceHolder(
                id=ctn,
                channel=key,
                list_of_full_pathes=temp_list,
                data=concatenated

            )

            ctn += 1

        print(f'[RESUMING] Input selection {best_dictionary}')
        return best_dictionary

    except Exception as ex:
        print(f"[EXCEPTION] Resume the input selection throws exception {ex}")
        print(traceback.print_exc())

"""
def load_the_best_combination_dictionary(current_results_path, channel_list):
    best_combination = {}

    for real_channel in channel_list:

        best_combination[real_channel] = []
        actual_accuracy = -1

        opener = open(current_results_path, 'r')

        for row in opener.readlines():
            channel = row.strip().split(',')[0]
            path = row.strip().split(',')[1]
            raster_name = row.strip().split(',')[2]
            accuracy = row.strip().split(',')[3]

            if real_channel == channel and actual_accuracy <= float(accuracy):

                if len(best_combination[real_channel]) == 0:
                    best_combination[real_channel].append(os.path.join(path, raster_name))
                else:
                    best_combination[real_channel][0] = os.path.join(path, raster_name)
        opener.close()

    return best_combination
"""

def prepare_the_sets_for_analysis(train_dataset, cnn_configuration, current_sample_number ,dataset, current_channel, best_dictionary):
    # allocate the sets 
    concatenated = None
    if current_channel in best_dictionary:
        for element in best_dictionary[current_channel]:
            
            if concatenated is None:
                concatenated = element.data
            else:
                concatenated = np.concatenate((concatenated, element.data), axis=3)
            
            # conmcat input plus best 
            train_dataset[f"input_{current_channel}"] = np.concatenate((dataset[current_channel][current_sample_number].data, concatenated), axis=3)
    else:
        # if there is nio besty to concat 
        train_dataset[f"input_{current_channel}"] = dataset[current_channel][current_sample_number].data

    cnn_configuration[f"input_{current_channel}"] = (train_dataset[f"input_{current_channel}"].shape[1], train_dataset[f"input_{current_channel}"].shape[2], train_dataset[f"input_{current_channel}"].shape[3])
    cnn_configuration[f"kernel_{current_channel}"] = (train_dataset[f"input_{current_channel}"].shape[3], train_dataset[f"input_{current_channel}"].shape[3])

    return train_dataset, cnn_configuration


def load_the_dictionary_with_best_combination(path):
    best_combination = {}
    opener = open(path, 'r')
    
    for row in opener.readlines():
        splitted = row.strip().split(',')
 
        if splitted[1] not in best_combination.keys():
            best_combination[splitted[1]] = list() 
        
        best_combination[splitted[1]].append(splitted[0])

    return best_combination

def load_the_data(list_of_pathes):

    concatenated = None 
    for element in list_of_pathes:
        temp_windows = list() 
        label_set = list() 
        for windows in os.listdir(element):
            label_set.append(int(windows.split('_')[1]))
            full_path = os.path.join(element, windows)
            temp_windows.append(np.load(full_path))
        if concatenated is None:
            concatenated = np.array(temp_windows)
            
        else:
            concatenated = np.concatenate((concatenated, np.array(temp_windows)), axis=3)
    print(concatenated.shape)
    concatenated = concatenated / 255. 
    label_set = np.array(label_set)
    return concatenated, label_set 


def load_the_best_combination_dictionary(current_results_path, channel_list):
    best_combination = {}

    for real_channel in channel_list:

        best_combination[real_channel] = []
        actual_accuracy = -1

        opener = open(current_results_path, 'r')

        for row_number, row in enumerate(opener.readlines()):
            channel = row.strip().split(',')[0]
            path = row.strip().split(',')[1]
            raster_name = row.strip().split(',')[1].split('/')[-1]
            accuracy = row.strip().split(',')[2]

            list_of_windows = list() 
            labels_set = list()  
            for windows in os.listdir(os.path.join(path)):
                temp = np.load(os.path.join(path, windows))
                labels_set.append(int(windows.split('_')[1]))
                list_of_windows.append(temp)
        
            combined = np.array(list_of_windows)
            combined = combined / 255. 

            temp = InputSelectionPlaceHolder(channel=channel,
                                             id=row_number,
                                             full_path=path,
                                             satellite_name=raster_name,
                                             image_name=raster_name,
                                             label_set=labels_set,
                                             data=combined,
                                             shape=combined.shape,
                                             accuracy=float(accuracy),
                                             is_best=True)

            if real_channel == channel and actual_accuracy <= float(accuracy):

                if len(best_combination[real_channel]) == 0:
                 best_combination[real_channel].append(temp)
                else:
                    best_combination[real_channel][0] = temp

        opener.close()

    return best_combination


def read_masterfile(current_masterfile):
    try:
        total_number_of_samples = 0
        input_data = {}

        handler = open(current_masterfile)
        for line in handler.readlines():
            templine = line.strip().split(',')
            
            if not templine[3] in input_data.keys():
                input_data[templine[3]] = list() 

            for path in os.listdir(templine[0]):
                full_path = path = os.path.join(templine[0], path)
                print(full_path)
                temp_list = list()
                label_set = list()
                windows_ctn = 0

                for windows in os.listdir(full_path):
                    label_set.append(int(windows.split('_')[1]))
                    temp = np.load(f'{full_path}/{windows}')
                    #print(temp.shape)
                    temp_list.append(temp)
                    

                concatenated = np.array(temp_list) 
                windows_ctn = concatenated.shape[0]
                input_data[templine[3]].append(InputSelectionPlaceHolder(channel=templine[3],
                                                                        id=total_number_of_samples,
                                                                        full_path=full_path,
                                                                        data=concatenated,
                                                                        satellite_name=templine[0],
                                                                        image_name=templine[1],
                                                                        label_set=label_set,
                                                                        shape=concatenated.shape))
                total_number_of_samples += 1
        return input_data, total_number_of_samples, windows_ctn
    except Exception as ex:
        print(f"[EXCEPTION] Read the masterfile throws exception {ex}")
        print(traceback.print_exc())


def load_best_dictionary_from_input_selection_results(input_selection_results_path):
    list_of_stuff = list()
    current_handler = {}
    best_combination = {}

    handler = open(input_selection_results_path, 'r')

    for line in handler.readlines():
        current_row = line.strip().split(' ')
        
        if current_row[0] != 'Mean':

            if current_row[0] == '+' or current_row[0] == '=': 
                continue

            list_of_stuff.append(line.strip().split(','))
        else:
            accuracy = float(current_row[-1])
            current_handler[accuracy] = list_of_stuff
            list_of_stuff = list() 
    
    handler.close() 

    best_accuracy = max(current_handler.keys()) 
    temp = current_handler[best_accuracy]
    for element_ctn, element in enumerate(temp):
        channel = element[0]
        path = element[1]
        
        if channel not in best_combination.keys():
            best_combination[channel] = list()
            
        temp_win = list() 
        temp_label = list () 

        for windows in os.listdir(path):
            temp_label.append(int(windows.split('_')[1]))
            
            temp_win.append(
                np.load((os.path.join(path, windows)))
            )
                
        
        best_combination[channel].append(InputSelectionPlaceHolder(
            channel=channel,
            id=element_ctn,
            satellite_name=path.split('/')[-2],
            image_name=path.split('/')[-1],
            full_path=path,
            label_set=np.array(temp_label, dtype='int'),
            data=np.array(temp_win, dtype='float'),
            shape=np.array(temp_win, dtype='float').shape
        ))
    
    print(f"[LOAD] Best combination is loaded {best_combination}")
    return best_combination


def stack_inputs(best_combination):
    output = {}
    for channel in best_combination.keys():
        combined = None

        for element in best_combination[channel]:
            labels = element.label_set
            if combined is None:
                combined = element.data
            else:
                combined = np.concatenate((combined, element.data), axis=3)
        
        output[channel] = combined  

    return output, labels 
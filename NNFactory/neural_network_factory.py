from Functions.functions import *


def create_multi_input_network(input_channel_1,
                               input_channel_2,
                               input_channel_3,
                               input_channel_4,
                               input_channel_5,
                               neuron_numbers,
                               kernel_channel_1,
                               kernel_channel_2,
                               kernel_channel_3,
                               kernel_channel_4,
                               kernel_channel_5,
                               regularization,
                               data_augmentation,
                               inputs=1,
                               add_extra_layers=0,
                               pool_size=2,
                               stride=1,
                               dropout_rate=None,
                               rotation_factor=(-0.2, 0.5),
                               output=39,
                               last_activation='softmax',
                               is_flatten=False,
                               layer_name=""):
    try:
        if inputs == 1:
            print("[NN FACTORY] 1 input selected")
            if input_channel_1 is not None:
                input_layer = tf.keras.Input(shape=input_channel_1, name="input_channel_1")
                kernel = kernel_channel_1

            elif input_channel_2 is not None:
                input_layer = tf.keras.Input(shape=input_channel_2, name="input_channel_2")
                kernel = kernel_channel_2

            elif input_channel_3 is not None:
                input_layer = tf.keras.Input(shape=input_channel_3, name="input_channel_3")
                kernel = kernel_channel_3

            elif input_channel_4 is not None:
                input_layer = tf.keras.Input(shape=input_channel_4, name="input_channel_4")
                kernel = kernel_channel_4

            else:
                input_layer = tf.keras.Input(shape=input_channel_5, name="input_channel_5")
                kernel = kernel_channel_5


            # body sar
            body = internal_body_of_the_network(model_input=input_layer,
                                                kernel_size=kernel,
                                                neuron_numbers=neuron_numbers,
                                                regularization=regularization,
                                                add_extra_layers=add_extra_layers,
                                                pool_size=pool_size,
                                                stride=stride,
                                                dropout_rate=dropout_rate,
                                                rotation_factor=rotation_factor,
                                                is_flatten=is_flatten,
                                                layer_name=f"body_{layer_name}",
                                                data_augmentation=data_augmentation)
            # create the classy
            classifier = tf.keras.layers.Dense(output, activation=last_activation, name="classifier")(body)
            model = tf.keras.Model(inputs=[input_layer], outputs=[classifier], name="model_with_1_input")

            return model
        else:
            print("[NN FACTORY] Multiples input")
            model_input, model_output =list(), list()
            if input_channel_1 is not None:
                sar = tf.keras.Input(shape=input_channel_1, name="input_channel_1")

                #body sar
                body_sar = internal_body_of_the_network(model_input=sar,
                                                        kernel_size=kernel_channel_1,
                                                        neuron_numbers=neuron_numbers,
                                                        regularization=regularization,
                                                        add_extra_layers=add_extra_layers,
                                                        pool_size=pool_size,
                                                        stride=stride,
                                                        dropout_rate=dropout_rate,
                                                        rotation_factor=rotation_factor,
                                                        is_flatten=is_flatten,
                                                        layer_name="body_channel_1",
                                                        data_augmentation=data_augmentation)

                model_sar = tf.keras.Model(inputs=sar, outputs=body_sar, name="model_with_channel_1_input")
                # add the model input and output the lists for the combined modules
                model_input.append(model_sar.input)
                model_output.append(model_sar.output)

            if input_channel_2 is not None:
                dem = tf.keras.Input(shape=input_channel_2, name="input_channel_2")
                body_dem = internal_body_of_the_network(model_input=dem,
                                                        kernel_size=kernel_channel_2,
                                                        neuron_numbers=neuron_numbers,
                                                        regularization=regularization,
                                                        add_extra_layers=add_extra_layers,
                                                        pool_size=pool_size,
                                                        stride=stride,
                                                        dropout_rate=dropout_rate,
                                                        rotation_factor=rotation_factor,
                                                        is_flatten=is_flatten,
                                                        layer_name="body_channel_2",
                                                        data_augmentation=data_augmentation)
                # create the dem model
                model_dem = tf.keras.Model(inputs=dem, outputs=body_dem, name="model_with_channel_2_input")
                model_input.append(model_dem.input)
                model_output.append(model_dem.output)

            if input_channel_3 is not None:
                nfi = tf.keras.Input(shape=input_channel_3, name="input_channel_3")
                # body nfi
                body_nvi = internal_body_of_the_network(model_input=nfi,
                                                        kernel_size=kernel_channel_3,
                                                        neuron_numbers=neuron_numbers,
                                                        regularization=regularization,
                                                        add_extra_layers=add_extra_layers,
                                                        pool_size=pool_size,
                                                        stride=stride,
                                                        dropout_rate=dropout_rate,
                                                        rotation_factor=rotation_factor,
                                                        is_flatten=is_flatten,
                                                        layer_name="body_channel_3",
                                                        data_augmentation=data_augmentation)
                # create the model
                model_nfi = tf.keras.Model(inputs=nfi, outputs=body_nvi, name="model_with_channel_3_input")
                model_input.append(model_nfi.input)
                model_output.append(model_nfi.output)

            if input_channel_4 is not None:
                chm = tf.keras.Input(shape=input_channel_4, name="input_channel_4")
                body_chm = internal_body_of_the_network(model_input=chm,
                                                        kernel_size=kernel_channel_4,
                                                        neuron_numbers=neuron_numbers,
                                                        regularization=regularization,
                                                        add_extra_layers=add_extra_layers,
                                                        pool_size=pool_size,
                                                        stride=stride,
                                                        dropout_rate=dropout_rate,
                                                        rotation_factor=rotation_factor,
                                                        is_flatten=is_flatten,
                                                        layer_name="body_channel_4",
                                                        data_augmentation=data_augmentation)

                model_chm = tf.keras.Model(inputs=chm, outputs=body_chm, name="model_with_channel_4_input")
                model_input.append(model_chm.input)
                model_output.append(model_chm.output)

            if input_channel_5 is not None:
                aero = tf.keras.Input(shape=input_channel_5, name="input_channel_5")
                body_aero = internal_body_of_the_network(model_input=aero,
                                                         kernel_size=kernel_channel_5,
                                                         neuron_numbers=neuron_numbers,
                                                         regularization=regularization,
                                                         add_extra_layers=add_extra_layers,
                                                         pool_size=pool_size,
                                                         stride=stride,
                                                         dropout_rate=dropout_rate,
                                                         rotation_factor=rotation_factor,
                                                         is_flatten=is_flatten,
                                                         layer_name="body_channel_5",
                                                         data_augmentation=data_augmentation)

                model_aero = tf.keras.Model(inputs=aero, outputs=body_aero, name="model_with_channel_5_input")
                model_input.append(model_aero.input)
                model_output.append(model_aero.output)



            # combined
            combined = tf.keras.layers.Concatenate(axis=-1)(model_output)
            classifier = tf.keras.layers.Dense(output,
                                               activation=last_activation,
                                               kernel_regularizer=None,
                                               bias_regularizer=None,
                                               name=f"classifier_{layer_name}")(combined)

            model = tf.keras.Model(inputs=model_input, outputs=classifier, name=f"maati_{layer_name}")

        return model

    except Exception as ex:
        print(f"Create multi input network throws exception {ex}")

"""
CHM dataset which is originally 1x1m resolution, should be converted to 2x2m resolution to match the resolution 
of DEM datasets.

[10:29 AM] Fahimeh Farahnakian
In CNN, there is only one "input line" corresponding to 2x2m resolution dataSatellite datasets 5x5 window 
sizes OK NFI datasets (16x16m) resolution 5x5 window ok CHM + DEM window size 25x25

If input is 25x25 window, we convolve it by 5x5 kernel, and this results to 21x21 

21x21 is maxpooled with 3x3 with stride 3 which results to 7x7  
"""

def internal_body_of_the_network(model_input,
                                 kernel_size,
                                 neuron_numbers,
                                 regularization,
                                 data_augmentation,
                                 layer_name="name",
                                 add_extra_layers=0,
                                 pool_size=2,
                                 stride=1,
                                 dropout_rate=None,
                                 rotation_factor=(-0.2, 0.5),
                                 is_flatten=False):
    try:

        # data aug
        if rotation_factor is not None:
            x = tf.keras.layers.RandomRotation(rotation_factor, name=f"random_rotation_{layer_name}")(model_input)

            # added data aug
            if data_augmentation:
                x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)

            # first layer
            x = tf.keras.layers.Conv2D(neuron_numbers, kernel_size=kernel_size, activation='relu', padding="same",
                                       kernel_regularizer=regularization,
                                       bias_regularizer=regularization,
                                       activity_regularizer=None,
                                       name=f"conv2d_{layer_name}_1")(x)
        else:
            x = tf.keras.layers.Conv2D(neuron_numbers, kernel_size=kernel_size, activation='relu', padding="same",
                                       kernel_regularizer=regularization,
                                       bias_regularizer=regularization,
                                       activity_regularizer=None,
                                       name=f"conv2d_{layer_name}_1")(model_input)


        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_name}_1")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size, strides=stride, name=f"max_pool_{layer_name}_1", padding="same")(x)
        # put prima pooling
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        # sec
        x = tf.keras.layers.Conv2D(neuron_numbers, kernel_size=kernel_size, activation='relu', padding="same",
                                   kernel_regularizer=regularization,
                                   bias_regularizer=regularization,
                                   activity_regularizer=None,
                                   name=f"conv2d_{layer_name}_2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_name}_2")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size, strides=stride, name=f"max_pool_{layer_name}_2", padding="same")(x)

        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if add_extra_layers > 0:
            for i in range(add_extra_layers):
                x = tf.keras.layers.Conv2D(neuron_numbers + i * 2, kernel_size=kernel_size, activation='relu',
                                           padding="same",
                                           kernel_regularizer=regularization,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           name=f"extra_{layer_name}_{i}")(x)
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_extra_{layer_name}_{i}")(x)
                x = tf.keras.layers.MaxPooling2D(pool_size, strides=stride, name=f"max_pool_extra_{layer_name}_{i}", padding="same")(
                    x)

                if dropout_rate is not None:
                    x = tf.keras.layers.Dropout(dropout_rate)(x)

        # flatten
        if is_flatten:
            x = tf.keras.layers.Flatten(name=f"flatten_{layer_name}")(x)
        else:
            x = tf.keras.layers.Reshape((-1,), name=f"reshape_{layer_name}")(x)

        return x
    except Exception as ex:
        print(f"internal body of the network throws exception {ex}")


def create_four_input_model_pix_classy(input_shape_normal, input_shape_dem, input_shape_chm, input_shape_nvi,output=39, last_activation='softmax'):
    try:
        normal = tf.keras.Input(shape=input_shape_normal, name="input_normal")
        dem = tf.keras.Input(shape=input_shape_dem, name="input_dem")
        chm = tf.keras.Input(shape=input_shape_chm, name="input_chm")
        nvi = tf.keras.Input(shape=input_shape_nvi, name="input_nvi")

        # first NORM number of channel the kernel size
        x = tf.keras.layers.Conv2D(input_shape_normal[2], input_shape_normal[2], activation='relu', padding="same", name="conv2d_normal_1")(normal)
        x = tf.keras.layers.BatchNormalization(name="batch_norm_normal_1")(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_normal_1", padding="same")(x)
        x= tf.keras.layers.Dropout(0.2)(x)
        # sec
        x = tf.keras.layers.Conv2D(input_shape_normal[2] * 2, input_shape_normal[2], activation='relu', padding="same", name="conv2d_normal_2")(x)
        x = tf.keras.layers.BatchNormalization(name="batch_norm_normal_2")(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_normal_2")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # flatten
        x = tf.keras.layers.Flatten(name="flatten_normal")(x)
        x = tf.keras.Model(inputs=normal, outputs=x, name="model_with_normal_input")

        # first RASTER
        y = tf.keras.layers.Conv2D(input_shape_dem[2], input_shape_dem[2], activation='relu', padding="same", name="conv2d_dem_1")(dem)
        y = tf.keras.layers.BatchNormalization(name="batch_norm_dem_1")(y)
        y = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_dem_1")(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        # sec
        y = tf.keras.layers.Conv2D(input_shape_dem[2] * 2, input_shape_dem[2], activation='relu', padding="same", name="conv2d_dem_2")(y)
        y = tf.keras.layers.BatchNormalization(name="batch_norm_dem_2")(y)
        y = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_dem_2")(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        # flatten
        y = tf.keras.layers.Flatten(name="flatten_dem")(y)
        y = tf.keras.Model(inputs=dem, outputs=y, name="model_with_dem_input")

        # first CHM
        z = tf.keras.layers.Conv2D(input_shape_chm[2], input_shape_chm[2], activation='relu', padding="same", name="conv2d_chm_1")(chm)
        z = tf.keras.layers.BatchNormalization(name="batch_norm_chm_1")(z)
        z = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_chm_1")(z)
        z = tf.keras.layers.Dropout(0.2)(z)
        # sec
        z = tf.keras.layers.Conv2D(input_shape_chm[2] * 2, input_shape_chm[2], activation='relu', padding="same", name="conv2d_chm_2")(z)
        z = tf.keras.layers.BatchNormalization(name="batch_norm_chm_2")(z)
        z = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_chm_2")(z)
        z = tf.keras.layers.Dropout(0.2)(z)
        # flatten
        z = tf.keras.layers.Flatten(name="flatten_chm")(z)
        z = tf.keras.Model(inputs=chm, outputs=z, name="model_with_chm_input")

        # first nvi
        a = tf.keras.layers.Conv2D(input_shape_nvi[2] , input_shape_nvi[2], activation='relu', padding="same", name="conv2d_nvi_1")(nvi)
        a = tf.keras.layers.BatchNormalization(name="batch_norm_nvi_1")(a)
        a = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_nvi_1")(a)
        a = tf.keras.layers.Dropout(0.2)(a)
        # sec
        a = tf.keras.layers.Conv2D(input_shape_nvi[2] * 2, input_shape_nvi[2], activation='relu', padding="same", name="conv2d_nvi_2")(a)
        a = tf.keras.layers.BatchNormalization(name="batch_norm_nvi_2")(a)
        a = tf.keras.layers.MaxPooling2D(2, strides=2, name="max_pool_nvi_2")(a)
        a = tf.keras.layers.Dropout(0.2)(a)
        # flatten
        a = tf.keras.layers.Flatten(name="flatten_nvi")(a)
        a = tf.keras.Model(inputs=nvi, outputs=a, name="model_with_nvi_input")

        # combined
        combined = tf.keras.layers.Concatenate(axis=-1)([x.output, y.output, z.output, a.output])
        classifier = tf.keras.layers.Dense(output, activation=last_activation, name="classifier")(combined)
        model = tf.keras.Model(inputs=[x.input, y.input, z.input, a.input], outputs=classifier, name="maati")

        return model

    except Exception as ex:
        print(f"Create two input network throws exception  {ex}")

def create_transfer_learning_for_2_features(path_to_a_weights, path_to_b_weights, output_classes, regularization, data_augmentation):
    try:

        best_model_a_weights = tf.keras.models.load_model(f"{path_to_a_weights}")
        best_model_b_weights = tf.keras.models.load_model(f"{path_to_b_weights}")

        # two lists that holds input and pooutput
        inputs_list, outputs_list = list(), list()

        # change the input name
        best_model_a_weights.layers[0]._name = "input_a"
        best_model_b_weights.layers[0]._name = "input_b"

        input_5 = x = best_model_a_weights.input
        input_25 = y = best_model_b_weights.input

        if data_augmentation:
            # insert random flip here
            x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", name="rnd_flip_a")(x)
            y = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", name="rnd_flip_b")(y)

        for layer_a, layer_b in zip(best_model_a_weights.layers[1:-1], best_model_b_weights.layers[1:-1]):
            # rename the model layer
            layer_a._name = layer_a.name + str("_a")
            layer_b._name = layer_b.name + str("_b")

            # freeze the weights because we need to keep the previous
            layer_a.trainable = False
            layer_b.trainable = False
            # transfer the freeze layer to the model
            x = layer_a(x)
            y = layer_b(y)

        # hold the inputs here
        inputs_list.append(input_5)
        inputs_list.append(input_25)
        outputs_list.append(x)
        outputs_list.append(y)

        combined = tf.keras.layers.Concatenate(axis=-1)(outputs_list)
        classifier = tf.keras.layers.Dense(output_classes,
                                           activation='softmax',
                                           kernel_regularizer=regularization,
                                           bias_regularizer=regularization,
                                           name="classifier_from_tf")(combined)
        # create the model
        model = tf.keras.Model(inputs=inputs_list, outputs=classifier, name="maati_transfer_learning")

        return model

    except Exception as ex:
        print(f"create transfer learning for 2 features {ex}")


def create_transfer_learning_for_3_features(path_to_a_weights, path_to_b_weights, path_to_c_weights, output_classes, regularization, data_augmentation):
    try:

        best_model_a_weights = tf.keras.models.load_model(f"{path_to_a_weights}")
        best_model_b_weights = tf.keras.models.load_model(f"{path_to_b_weights}")
        best_model_c_weights = tf.keras.models.load_model(f"{path_to_c_weights}")

        # two lists that holds input and pooutput
        inputs_list, outputs_list = list(), list()

        # change the input name
        best_model_a_weights.layers[0]._name = "input_a"
        best_model_b_weights.layers[0]._name = "input_b"
        best_model_c_weights.layers[0]._name = "input_c"

        input_sar = x = best_model_a_weights.input
        input_raster = y = best_model_b_weights.input
        input_nfi = z = best_model_c_weights.input

        if data_augmentation:
            # insert random flip here
            x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", name="rnd_flip_a")(x)
            y = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", name="rnd_flip_b")(y)
            z = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", name="rnd_flip_c")(z)

        for layer_a, layer_b, layer_c in zip(best_model_a_weights.layers[1:-1], best_model_b_weights.layers[1:-1], best_model_c_weights.layers[1:-1]):
            # rename the model layer
            layer_a._name = layer_a.name + str("_a")
            layer_b._name = layer_b.name + str("_b")
            layer_c._name = layer_c.name + str("_c")

            # freeze the weights because we need to keep the previous
            layer_a.trainable = False
            layer_b.trainable = False
            layer_c.trainable = False
            # transfer the freeze layer to the model
            x = layer_a(x)
            y = layer_b(y)
            z = layer_c(z)

        # hold the inputs here
        inputs_list.append(input_sar)
        inputs_list.append(input_raster)
        inputs_list.append(input_nfi)

        outputs_list.append(x)
        outputs_list.append(y)
        outputs_list.append(z)

        combined = tf.keras.layers.Concatenate(axis=-1)(outputs_list)
        classifier = tf.keras.layers.Dense(output_classes,
                                           activation='softmax',
                                           kernel_regularizer=regularization,
                                           bias_regularizer=regularization,
                                           name="classifier_from_tf")(combined)
        # create the model
        model = tf.keras.Model(inputs=inputs_list, outputs=classifier, name="maati_transfer_learning")

        return model

    except Exception as ex:
        print(f"create transfer learning for 3 features {ex}")


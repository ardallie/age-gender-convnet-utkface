"""
Convolutional Neural Network (CNN) that uses
UTKFace dataset to estimate age and classify gender.
"""

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot

# Disable Pylint warnings about Keras imports
# pylint: disable=E0611
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

STAGES = ["train", "val", "test"]
EPOCHS = 128
BATCH_SIZE = 160
ALL_IMAGES = 23708
MODEL_NAME = "age_gender_model"

dir_root = os.getcwd()
model_folder = f"{dir_root}/output/models"
model_path = f"{model_folder}/{MODEL_NAME}.h5"
plot_path = f"{dir_root}/output/plots/{MODEL_NAME}.png"
history_path = f"{dir_root}/output/history/{MODEL_NAME}.csv"
# folder containing all images from the dataset
dir_all_images = f"{dir_root}/all_images"
# caution. this temporary folder can be deleted programmatically.
# do not change it until you're absolutely sure.
dir_split = f"{dir_root}/tmp_split_images"

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def make_folders():
    """
    Creates new folders for train, val and test data.
    It deletes and re-creates the /split_images folder
    with three subfolders contianing the split images.
    """
    if os.path.exists(dir_split):
        shutil.rmtree(dir_split)
        print(f"\nDeleted the folder {dir_split}")
    if not os.path.exists(dir_split):
        os.mkdir(dir_split)
        print(f"Created the folder {dir_split}")
    for f in STAGES:
        f_path = os.path.join(dir_split, f)
        if not os.path.exists(f_path):
            # make a new folder for data type (train, val, test)
            os.mkdir(f_path)
            print(f"Created the folder {f_path}")


def prepare_item_split(df, split=0.8):
    """
    The function allocates images from a single folder to three subgroups: train, val, test.
    Uses numpy permutations to split the data without shuffling it.
    It prevents from putting only one age group into a subgroup.

    Args:
        df: dataframe with metadata
        split: the split ratio

    Returns:
        a Python dict with 3 keys (train, val, test).
        Each key has a corresponding list of files belonging to that group
    """
    p = np.random.permutation(len(df))
    train_up_to = int(len(df) * split)
    train_set = p[:train_up_to]
    test_set = p[train_up_to:]
    train_up_to = int(train_up_to * split)
    train_set, val_set = train_set[:train_up_to], train_set[train_up_to:]
    groups = {}
    for i in train_set:
        groups[i] = STAGES[0]
    for j in val_set:
        groups[j] = STAGES[1]
    for k in test_set:
        groups[k] = STAGES[2]
    return groups


def split_files(df):
    """
    Copies files from one folder to the respective train, val and test folders.

    Args:
        df: dataframe with metadata
    """
    item_split = prepare_item_split(df, 0.90)
    assert len(item_split) == ALL_IMAGES
    # add folders train, test, split
    make_folders()
    # copy files to the newly created folders
    print(f"Copying images to split folders {dir_split}")
    for i, row in df.iterrows():
        src = os.path.join(dir_root, dir_all_images, row["filename"])
        dst = os.path.join(dir_split, item_split.get(i), row["filename"])
        shutil.copyfile(src, dst)
    print("Copying complete.")


def get_label(fname):
    """
    Reads the provided filename string and extracts the age and gender labels

    Args:
        fname: filename containing labels delimitged with _

    Returns:
        a tuple containing age, gender and the input filename
    """
    if fname is None or fname == "":
        return ""
    metadata = fname.split("_")
    age = int(metadata[0])
    gender = int(metadata[1])
    display_name = "Female" if gender == 1 else "Male"
    return age, gender, display_name, fname


def get_all_labels(dir_name):
    """
    Read all files from a directory then retrieve the metadata (labels) from file names
    Age should NOT be normalised to 0-1 range.

    Args:
        dir_name: directory name (train, val, test)

    Returns:
        Pandas dataframe containing age, gender and filename
    """
    all_image_files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    labels = []
    for file_name in all_image_files:
        metadata = get_label(file_name)
        labels.append([*metadata])
    df = pd.DataFrame(labels)
    df.columns = ["age", "gender", "display_name", "filename"]
    # DO NOT convert values to string
    # df['age'] = df['age'].apply(lambda x: str(x))
    return df


def get_image_title(df_row):
    """
    Get the image title (e.g. Female, 34 years)

    Args:
        df_row: the single row containing image metadata

    Returns:
        an image title
    """
    return f'{df_row["display_name"]}, {df_row["age"]} year{"s" if df_row["age"] > 1 else ""}'


def display_images(dir_name, df):
    """
    Displays the series of images from the selected folder

    Args:
        dir_name: directory name (train, val, test)
        df: dataframe containing image metadata. It must match the files in the dir_name.
    """
    pyplot.figure(figsize=(10, 10), facecolor="#323232")
    for i in range(20):
        pyplot.subplot(5, 5, i + 1)
        pyplot.axis("off")
        rnd = np.random.randint(0, df.shape[0])
        row = df.iloc[rnd]
        filename = dir_name + "/" + row["filename"]
        if not os.path.exists(filename):
            print("No such file:" + filename)
        pyplot.imshow(pyplot.imread(filename))
        pyplot.setp(pyplot.title(get_image_title(row)), color="#eaeaea", fontsize=10.0)
    pyplot.show()


def prepare_conv_model(inp):
    """
    Prepares a convolutional layer structure using Keras Functional API
    This structure is reused by age and gender models.

    Args:
        inp: input layer with a size of (200, 200, 3)

    Returns: the model compliant with Keras Functional API
    """
    m = (Conv2D(64, (3, 3), activation="relu", input_shape=(200, 200, 3)))(inp)
    m = (MaxPooling2D((2, 2)))(m)
    m = (Dropout(0.2))(m)
    m = (Conv2D(64, (3, 3), activation="relu"))(m)
    m = (MaxPooling2D((2, 2)))(m)
    m = (Dropout(0.2))(m)
    m = (Conv2D(64, (3, 3), activation="relu"))(m)
    m = (MaxPooling2D((2, 2)))(m)
    m = (Dropout(0.2))(m)
    m = (Conv2D(64, (3, 3), activation="relu"))(m)
    m = (MaxPooling2D((2, 2)))(m)
    m = (Dropout(0.2))(m)
    return m


def prepare_gender_model(inp):
    """
    Prepares a layer structure for gender model using Keras Functional API
    This structure uses the same convolutional layer base.

    Args:
        inp: input layer with a size of (200, 200, 3)

    Returns: the model compliant with Keras Functional API
    """
    m = (Flatten(name="gender_branch"))(inp)
    m = (Dropout(0.4))(m)
    m = (Dense(128, activation="relu"))(m)
    m = (BatchNormalization())(m)
    m = (Dropout(0.4))(m)
    m = (Dense(16, activation="relu"))(m)
    m = (Dropout(0.4))(m)
    m = (Dense(1, activation="sigmoid", name="gender"))(m)
    return m


def prepare_age_model(inp):
    """
    Prepares a layer structure for age model using Keras Functional API
    This structure uses the same convolutional layer base.

    Args:
        inp: input layer with a size of (200, 200, 3)

    Returns: the model compliant with Keras Functional API
    """
    m = (Flatten(name="age_branch"))(inp)
    m = (Dropout(0.3))(m)
    m = (Dense(256, activation="relu"))(m)
    m = (BatchNormalization())(m)
    m = (Dropout(0.3))(m)
    m = (Dense(128, activation="relu"))(m)
    m = (Dropout(0.3))(m)
    m = (Dense(1, activation="linear", name="age"))(m)
    return m


def assemble_model():
    """
    Assemble the model from two separate structures: age and gender

    Returns: an amalgamated model
    """
    inp = Input(shape=(200, 200, 3), name="input_layer")
    m = prepare_conv_model(inp)
    model_gender = prepare_gender_model(m)
    model_age = prepare_age_model(m)
    model_built = Model(inputs=inp, outputs=[model_gender, model_age])
    model_built.summary()
    # Graphvis issues with plotting
    # https://stackoverflow.com/a/62611005/6666457
    # In terminal with admin rights:
    # dot -c
    plot_model(
        model_built,
        to_file=plot_path,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=150,
    )
    return model_built


# Disable Pylint warnings about the number of arguments
# pylint: disable=R0913
def image_generator(df, dir_name, y_col, class_mode="binary", aug=False, batch_size=1):
    """
    Image Generator from the Data Frame

    Additional reading:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    https://github.com/keras-team/keras-preprocessing/blob/362fe9f8daf556151328eb5d02bd5ae638c653b8/keras_preprocessing/image.py#L1015
    https://vijayabhaskar96.medium.com/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1

    Args:
        df: dataframe with metadata
        dir_name: directory name (train, val, test)
        y_col: data label or list of data labels
        class_mode: class mode (binary, categorical, multi-output)
        aug: True if data augmentation is requred. False will return original images
        batch_size: number of images returned in each batch

    Returns:
        a batch of images

    """
    # 1. generate augmented images for training
    aug_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        # horizontal_flip=True,
        # rotation_range=15,
        # zoom_range=0.2
        # channel_shift_range=32,
    )

    # 2. generate unchanged images for validation and test
    img_gen = ImageDataGenerator(
        rescale=1.0 / 255,
    )
    # aug parameter to decide which generator to use
    data_gen = aug_gen if aug is True else img_gen
    # return the image flow
    return data_gen.flow_from_dataframe(
        df,
        target_size=(200, 200),
        directory=dir_name,
        x_col="filename",
        y_col=y_col,
        batch_size=batch_size,
        class_mode=class_mode,
    )


def compile_model(m):
    """
    Compile the Keras model.

    Args:
        m: Keras model

    Returns:
        compiled Keras model
    """
    opt = Adam(learning_rate=0.0002)
    loss_weights = {"gender": 2, "age": 0.25}
    loss_functions = {
        "gender": "binary_crossentropy",
        "age": "mse",
    }
    metrics = {
        "gender": "accuracy",
        "age": "mae",
    }
    m.compile(loss=loss_functions, metrics=metrics, optimizer=opt, loss_weights=loss_weights)
    return m


def train(m):
    """
    Train the convolutional neural network.

    Args:
        m: model

    Returns:
        a tuple containing a model and a history object
    """
    dir_train = f"{dir_split}/{STAGES[0]}"
    lab_train = get_all_labels(dir_train)
    gen_train = image_generator(
        lab_train, dir_train, y_col=["gender", "age"], class_mode="multi_output", aug=True, batch_size=BATCH_SIZE
    )
    dir_val = f"{dir_split}/{STAGES[1]}"
    lab_val = get_all_labels(dir_val)
    gen_val = image_generator(
        lab_val, dir_val, y_col=["gender", "age"], class_mode="multi_output", aug=False, batch_size=BATCH_SIZE
    )
    m = compile_model(m)

    # Early Stopping callback
    es_cb = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=12, restore_best_weights=True)

    # Save the history to CSV
    # https://keras.io/api/callbacks/csv_logger/
    csv_cb = CSVLogger(history_path, separator=",", append=False)

    h = m.fit(
        gen_train,
        steps_per_epoch=lab_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=gen_val,
        validation_steps=lab_val.shape[0] // BATCH_SIZE,
        callbacks=[es_cb, csv_cb],
    )
    return m, h


def save_model(m):
    """
    Saves a model in H5 format, given the extension is provided.
    If an extension .h5 is not provided in the file name,
    the folder structure will be created

    Args:
        m: Keras model
    """
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    m.save(model_path)


def plot_line_chart(h, series):
    """
    Plot the line chart

    Args:
        h: history object
        series: a dict containinga measure as key and display name as value. E.g.
        {
            "gender_accuracy": "Gender Accuracy",
            "val_gender_accuracy": "Validation Gender Accuracy"
        }

    """
    pyplot.figure(figsize=(16, 8))
    for s in series:
        pyplot.plot(h[s], label=series[s])
    pyplot.legend()
    pyplot.xlabel("Epoch")
    pyplot.grid(True)
    pyplot.show()


def show_results(h):
    """
    Plots learning results using the line charts

    Args:
        h: history object
    """
    series1 = {"gender_accuracy": "Gender Accuracy", "val_gender_accuracy": "Val. Gender Accuracy"}
    plot_line_chart(h, series1)

    series2 = {"gender_loss": "Gender Loss", "val_gender_loss": "Val. Gender Loss"}
    plot_line_chart(h, series2)

    series3 = {"loss": "Loss", "val_loss": "Val. Loss"}
    plot_line_chart(h, series3)

    series4 = {"age_mae": "Age MAE", "val_age_mae": "Val. Age MAE"}
    plot_line_chart(h, series4)


def get_gender_label(x):
    """
    Get a gender label (Man, Woman)

    Args:
        x: an integer denoting gender

    Returns:
        gender label

    """
    if x == 0:
        return "Man"
    elif x == 1:
        return "Woman"
    return None


def load_test_images():
    """
    Load the test images using Keras image generator.
    """
    dir_test = f"{dir_split}/{STAGES[2]}"
    lab_test = get_all_labels(dir_test)
    return image_generator(
        lab_test, dir_test, y_col=["gender", "age"], class_mode="multi_output", aug=False, batch_size=1
    )


def test_model(model):
    """
    Test the pre-trained model.

    Args:
        model: pre-trained model
    """
    gen_test = load_test_images()
    model.evaluate(gen_test, verbose=1)

    c = 0
    for batch in gen_test:
        img = batch[0][0]
        (_act_gender, _act_age) = batch[1]
        (_pred_gender, _pred_age) = model.predict(np.reshape(img, (1, 200, 200, 3)))
        pred_gender = round(_pred_gender[0][0])
        pred_age = round(_pred_age[0][0])
        actual_gender = _act_gender[0]
        actual_age = _act_age[0]

        pred_gender_label = get_gender_label(pred_gender)
        actual_gender_label = get_gender_label(actual_gender)

        print(f"Predicted: {pred_gender_label}, {pred_age} | ", end="")
        print(f"Actual: {actual_gender_label}, {actual_age}")

        pyplot.title(f"PRE: {pred_gender_label}, {str(pred_age)} | ACT: {actual_gender_label}, {str(actual_age)}")
        pyplot.imshow(img)
        pyplot.show()
        c += 1
        if c >= 6:
            break


def run_split_images():
    """
    Delete then re-create a temporary folder /tmp_split_images.
    Images from /all_images folder are copied into three respective subfolders:
    train, test and val
    """
    all_labels = get_all_labels(dir_all_images)
    assert len(all_labels) == ALL_IMAGES
    split_files(all_labels)


def run_training():
    """
    Runs the model training.
    """
    if not os.path.exists(dir_split):
        print("\nPlease split the images first.\n")
        return
    all_labels = get_all_labels(dir_all_images)
    assert len(all_labels) == ALL_IMAGES
    display_images(dir_all_images, all_labels)
    model = assemble_model()
    (trained_model, history) = train(model)
    save_model(trained_model)
    show_results(history.history)


def run_testing():
    """
    Runs the model testing.
    """
    if not os.path.isfile(model_path):
        print("\nPlease train the model first.\n")
        return
    model = load_model(model_path)
    test_model(model)


def run_gpu_info():
    """
    Get the GPU info from TensorFlow.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    is_cuda = tf.test.is_built_with_cuda()
    print(f"\nIs built with CUDA? {is_cuda}")
    if len(gpus) > 0:
        print(f"Detected GPU(s) \n {gpus}\n")


def main():
    """
    Entry point.
    """
    run_gpu_info()


if __name__ == "__main__":
    main()

# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

torch.manual_seed(42)

from ResNet_Module import ResidualBlock, Resnet

# Changes to be made - Add path to Train and Val files used in ResNet
# Model Files to be renamed

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = (
    "/Users/abirabh/Downloads/file"
)
OUTPUT_CSV_ABSOLUTE_PATH = "/Users/abirabh/Documents/DL_Project_Task1/output_RN2.csv"
# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

frameSize = 2048
hopSize = 512

transform_img = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = 4
else:
    num_workers = 1

# If use not using a GPU which supports CUDA, please ignore the warning regarding NNPACK

# print(device)
model = Resnet(ResidualBlock, [3, 4, 6, 3]).to(device)
model.load_state_dict(
    torch.load(
        f"{os.getcwd()}/checkpoint_resnet_final.pth",
        map_location=torch.device(device),
    )
)

print("Model Loaded!")

pred_to_label = {
    0: "Fart",
    1: "Guitar",
    2: "Gunshot_and_gunfire",
    3: "Hi-hat",
    4: "Knock",
    5: "Laughter",
    6: "Shatter",
    7: "Snare_drum",
    8: "Splash_and_splatter",
    9: "car_horn",
    10: "dog_barking",
    11: "drilling",
    12: "siren",
}

final_labelling = {
    "Fart": 4,
    "Guitar": 5,
    "Gunshot_and_gunfire": 6,
    "Hi-hat": 7,
    "Knock": 8,
    "Laughter": 9,
    "Shatter": 10,
    "Snare_drum": 12,
    "Splash_and_splatter": 13,
    "car_horn": 1,
    "dog_barking": 2,
    "drilling": 3,
    "siren": 11,
}

currdir = TEST_DATA_DIRECTORY_ABSOLUTE_PATH[
    : TEST_DATA_DIRECTORY_ABSOLUTE_PATH.rfind("/") + 1
]
print(currdir)
if currdir == "":
    currdir = "/"


# Referred from
# https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb
def create_spectrogram(audio_file, image_file, folder):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # print(audio_file)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frameSize)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(folder + image_file[: image_file.rfind(".")] + ".png")
    plt.close(fig)
    return folder + image_file[: image_file.rfind(".")] + ".png"


def convertAudio(audiofile, target_dir, curr_dir):
    extension = audiofile.split(".")[-1]
    # print(audiofile)
    if "new" not in audiofile:
        newfile = (
            curr_dir
            + "/"
            + audiofile[: audiofile.rfind("/") + 1]
            + "new_"
            + audiofile[audiofile.rfind("/") + 1 :]
        )
    else:
        newfile = audiofile
    audiofile2 = curr_dir + "/" + audiofile
    try:
        audio = AudioSegment.from_file(
            file=curr_dir + "/" + audiofile, format=extension
        )
    except:
        y, fs = sf.read(curr_dir + "/" + audiofile)
        sf.write(newfile, y, fs)
        audio = AudioSegment.from_file(file=newfile, format=extension)
        audiofile2 = newfile
    padded_audio = AudioSegment.silent(duration=4000)
    padded_audio = padded_audio.overlay(
        AudioSegment.from_file(audiofile2).set_frame_rate(22050).set_channels(1)
    )[0:4000]
    aug_file = (
        f"{target_dir}/pro_"
        + audiofile[audiofile.rfind("/") + 1 : audiofile.rfind(".")]
        + ".wav"
    )
    padded_audio.export(out_f=aug_file, format="wav")
    return aug_file


def evaluate(file_path):
    file_path = file_path[file_path.rfind("/") + 1 :]
    target_dir = currdir + "processed_Audio/"
    audio_dir = TEST_DATA_DIRECTORY_ABSOLUTE_PATH
    # Write your code to predict class for a single audio file instance here
    aug_file = convertAudio(file_path, target_dir, audio_dir)
    image = "spectro_" + file_path
    # print(image)
    spectro_loc = create_spectrogram(aug_file, image, target_dir)
    # print(spectro_loc)
    img = Image.open(spectro_loc)
    img = img.convert("RGB")
    # print(img.getcolors())
    img = transform_img(img)

    # img = img.to(device)
    model.eval()
    output = model(img.unsqueeze(0))
    # print(output)
    predicted_class = final_labelling[pred_to_label[torch.argmax(output).item()]]
    # print(test_file)
    return predicted_class


def evaluate_batch(file_path_batch, batch_size=32):
    # Write your code to predict class for a batch of audio file instances here
    predicted_class_batch = 0
    return predicted_class_batch


def test():
    try:
        os.mkdir(currdir + "processed_Audio")
    except:
        shutil.rmtree(currdir + "processed_Audio")
        os.mkdir(currdir + "processed_Audio")

    filenames = []
    predictions = []

    # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
    for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
        # prediction = evaluate(file_path)
        try:
            absolute_file_name = os.path.join(
                TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name
            )
            prediction = evaluate(absolute_file_name)

            filenames.append(absolute_file_name)
            predictions.append(prediction)
        except:
            print(
                "Skipping File, as it is either not an audio file, or has wrong formatting! ",
                file_name,
            )
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(
        OUTPUT_CSV_ABSOLUTE_PATH, index=False
    )


def test_batch(batch_size=32):
    filenames = []
    predictions = []

    # paths = os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    paths = [os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, i) for i in paths]

    # Iterate over the batches
    # For each batch, execute evaluate_batch function & append the filenames for that batch in the filenames list and the corresponding predictions in the predictions list.

    # The list "paths" contains the absolute file path for all the audio files in the test directory. Now you may iterate over this list in batches as per your choice, and populate the filenames and predictions lists as we have demonstrated in the test() function. Please note that if you use the test_batch function, the end filenames and predictions list obtained must match the output that would be obtained using test() function, as we will evaluating in that way only.

    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(
        OUTPUT_CSV_ABSOLUTE_PATH, index=False
    )


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
test()
# test_batch()

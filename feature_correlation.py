# Beat tracking example
import librosa
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns


def main():
    # all feature data from audio files
    feature_data = extract_features()
    # all trial data from audio files
    trial_data = extract_trial_data()

    # generates csv file with feature and trial data
    create_csv_file(feature_data, trial_data)

    # generates cross correlation matrix from data in csv file
    generate_matrix()


def extract_features():
    # list of dicts containing file feature data
    feature_data = []

    # directory of audio files in binary
    directory = os.fsencode(os.getcwd() + "/audioFiles")

    # iterates through files gets feature data and appends to feature_data
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp3"):
            feature_data.append(get_features(
                os.getcwd() + "/audioFiles/" + filename))
        else:
            continue

    return feature_data


def extract_trial_data():
    sd_file_name = os.getcwd() + "/audioData/trial_data.csv"

    data = pd.read_csv(sd_file_name)

    return data.to_dict(orient="records")


def get_features(filename):
    # loads the audio as a waveform `y` and stores the sampling rate as `sr`
    y, sr = librosa.load(filename)
    # spereates waveform 'y' into its percusive and harmonic elements
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # computes a spectral flux onset strength envelope.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # computes spectrogram magnitude from waveform 'y'
    S = np.abs(librosa.stft(y))

    # extracts estimated tempo(bpm)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    # extracts tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    for frame in tempogram:
        frame = np.mean(frame)
    tempogram = np.mean(tempogram)

    # extraxts Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for frame in mfcc:
        frame = np.mean(frame)
    mfcc = np.mean(mfcc)

    # extracts spectral centroid
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    # extracts spectral banwidth
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
    # extracts spectral contrast
    spec_cont = np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)[0])
    # extracts spectral flatness
    spec_flat = np.mean(librosa.feature.spectral_flatness(y=y)[0])
    # extracts spectral rolloff
    spec_roll = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)[0])

    # extracts root mean square value for each frame
    rms = np.mean(librosa.feature.rms(y=y)[0])

    # extracts tonnetz form harmonic component of a song
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    # Seperating tonnetz dimensions
    fifth_x_axis = np.mean(tonnetz[0])
    fifth_y_axis = np.mean(tonnetz[1])
    minor_x_axis = np.mean(tonnetz[2])
    minor_y_axis = np.mean(tonnetz[3])
    major_x_axis = np.mean(tonnetz[4])
    major_y_axis = np.mean(tonnetz[5])

    return {'file_name': os.path.basename(filename), "tempo": tempo, "tempogram": tempogram,
            "spec_cent": spec_cent, "spec_bw": spec_bw, "spec_cont": spec_cont, "spec_flat": spec_flat,
            "spec_roll": spec_roll, "mcff": mfcc, "rms": rms,
            "tonnetz_fifth_x_axis": fifth_x_axis, "tonnetz_fifth_y_axis": fifth_y_axis,
            "tonnetz_minor_x_axis": minor_x_axis, "tonnetz_minor_y_axis": minor_y_axis,
            "tonnetz_major_x_axis": major_x_axis, "tonnetz_major_y_axis": major_y_axis}


def create_csv_file(feature_data, trial_data):
    # path where data will be stored
    filename = os.getcwd() + "/audioData/study_data.csv"

    # combine data from study and feature extraction
    data = []
    for i in feature_data:
        for x in trial_data:
            if x["file_name"] == i["file_name"]:
                i.update(x)
                data.append(i)

    with open(filename, mode='w') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


def generate_matrix():
    df_file_name = os.getcwd() + "/audioData/study_data.csv"
    matrix_file_name = os.getcwd() + "/audioData/matrix.csv"

    df = pd.read_csv(df_file_name).drop(["file_name"], axis=1)

    matrix = df.corr()
    matrix.to_csv(matrix_file_name)


if __name__ == "__main__":
    main()

# Beat tracking example
import librosa
import os
import csv
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import pearsonr


def main():
    feature_data = extract_features()
    create_csv_file(feature_data)
    generate_matrix()


def extract_features():

    # Stores a list of dicts containing file feature data
    feature_data = []

    directory = os.fsencode(os.getcwd() + "/audioFiles")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp3"):
            feature_data.append(get_features(
                os.getcwd() + "/audioFiles/" + filename))
        else:
            continue

    return feature_data


def get_features(filename):
    # loads the audio as a waveform `y` and stores the sampling rate as `sr`
    y, sr = librosa.load(filename)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # computes a spectral flux onset strength envelope.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    S = np.abs(librosa.stft(y))

    # extracts estimated tempo(bpm)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    # extracts tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    # extracts spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    # extracts spectral banwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # extracts spectral contrast
    spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr)

    # extracts spectral flatness
    spec_flat = librosa.feature.spectral_flatness(y=y)

    # extracts spectral rolloff
    spec_roll = librosa.feature.spectral_rolloff(S=S, sr=sr)

    # extracts tonnetz form harmonic component of a song
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # mcff/rms

    return {'file_name': os.path.basename(filename), "tempo": tempo[0], "tempogram": tempogram,
            "spec_cent": spec_cent, "spec_bw": spec_bw, "spec_cont": spec_cont, "spec_flat": spec_flat,
            "tonnetz": tonnetz}


def create_csv_file(feature_data):
    filename = os.getcwd() + "/audioData/feature_data.csv"
    with open(filename, mode='w') as csvfile:
        fieldnames = feature_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in feature_data:
            writer.writerow(entry)


def corr_np(data1, data2):
    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()
    corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr


def generate_matrix():
    fd_file_name = os.getcwd() + "/audioData/feature_data.csv"
    dr_file_name = os.getcwd() + "/audioData/danger_ratings.csv"

    df = pd.read_csv(fd_file_name)
    dr = pd.read_csv(dr_file_name)

    tempo_data = df.loc[:, "tempo"]
    danger_ratings = dr.loc[:, "danger_rating"]
    print(np.corrcoef(tempo_data, danger_ratings))


if __name__ == "__main__":
    main()

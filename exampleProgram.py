# Beat tracking example
import librosa
import os
import csv


def main():
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

    create_csv_file(feature_data)


def get_features(filename):
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # Estimates the tempo in bmp
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    print("Estimated Tempo:     %f bpm" % tempo[0])

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    print("Spectral Flatness:   ", spectral_flatness)

    return {'file_name': os.path.basename(filename), "tempo": tempo[0], "spectral_flatness": spectral_flatness}


def create_csv_file(feature_data):
    filename = "featureData.csv"
    with open(filename, mode='w') as csvfile:
        fieldnames = feature_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in feature_data:
            writer.writerow(entry)


main()

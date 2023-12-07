import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
class Utility:
    def __init__(self):
        self.BASE_DIR = os.getcwd()

    def remove_all_file(self, name, folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    if name in the_file:
                        os.unlink(file_path)
                elif os.path.isdir(file_path):
                    self.remove_all_file(name, file_path)
            except Exception as e:
                print(e)

    def create_spectogram(self, data):
        plt.subplot(2, 2, 1)
        print("Data shape: " + str(data.shape))
        spectrum, freq, t, im = plt.specgram(data[:, 0], Fs=44100, NFFT=1024)
        plt.title("Data shape: " + str(data.shape))
        plt.subplot(2, 2, 2)
        plt.title("Spectrum Spectral density shape: " + str(spectrum.shape[0]))
        plt.plot(spectrum[:, 0])
        plt.subplot(2, 2, 3)
        plt.title("Window shape: " + str(spectrum.shape[1]))
        plt.plot(spectrum[0, :])
        plt.subplot(2, 2, 4)
        plt.title("Frequency: " + str(freq.shape))
        plt.plot(spectrum[:, 0], freq)
        plt.show()
    def execute_recursively(self, folder, name, callback_function):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path) and file == name:
                callback_function(file_path)
                return
            elif os.path.isdir(file_path):
                self.execute_recursively(file_path, name, callback_function)

    def create_wavefile(self, data, url, sampling_rate):
        wav.write(url, sampling_rate, data)

    def convert_to_wave(self, folder, data_filename, sr):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path) and data_filename in file:
                    data = np.load(file_path)
                    wav.write(os.path.join(folder, data_filename + ".wav"), sr, data)
            elif os.path.isdir(file_path):
                print("Converting files of directory: " + file_path)
                self.convert_to_wave(file_path, data_filename, sr)
    def rename(self, dir, old_name, new_name):
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path) and old_name in file:
                os.rename(file_path, os.path.join(dir, new_name))
            elif os.path.isdir(file_path):
                print("Renaming files in " + file_path)
                self.rename(file_path, old_name, new_name)
    def label_training_data(self, raw_data_dir, label_data_base_dir, np_data_name, inp_map):
        for file in os.listdir(raw_data_dir):
            file_path = os.path.join(raw_data_dir, file)
            if os.path.isfile(file_path) and np_data_name in file_path:
                print("Playing audio sample: " + file_path)
                data = np.load(file_path)
                sd.play(data)
                print()
                index = int(input("Choose directory: Base dir: " + label_data_base_dir))
                if 0 <= index < len(inp_map):
                    dir_name = inp_map[index]
                    old_dir_path = raw_data_dir
                    new_dir_path = os.path.join(label_data_base_dir, dir_name)
                    print("Transferring data from " + old_dir_path + " to " + new_dir_path)
                    shutil.move(old_dir_path, new_dir_path)
                elif index == -1:
                    shutil.rmtree(raw_data_dir)
            elif os.path.isdir(file_path):
                self.label_training_data(file_path, label_data_base_dir, np_data_name, inp_map)

        # sound\noise
        # music
        # talk
        # sound\voice


if __name__ == "__main__":
    util = Utility()
    base_dir = os.path.join(os.getcwd(), "dataset", "predicted", "other")
    input_map = [os.path.join("sound", "noise"), "music", "talk", os.path.join("sound", "voice")]
    #data = np.load(base_dir+'data.npy', allow_pickle=True)
    #util.create_spectogram(data)
    #util.convert_to_wave(base_dir, "data", 44100)
    #util.rename(base_dir, "sepctogram.npy", "spectogram.npy")
    util.label_training_data(base_dir, os.path.join(os.getcwd(), "dataset", "teset"), "data.npy", input_map)

import json, os, random
import numpy as np
class Utility:
    @staticmethod
    def load_numpy_data(file_path):
        return np.load(file_path, allow_pickle=True)
    @staticmethod
    def get_information(json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
        return data
class DatasetCollector:
    def __init__(self, data_name):
        self.data_folder = data_name
        self.BASE_DIR = 'D:\\thesis\\ConvNet\\MyNet\\temp\\'
        self.dataset_base_dir = self.BASE_DIR + 'dataset\\'
        self.network_base_dir = self.BASE_DIR + 'networks\\'
        self.train_dataset_dir = self.dataset_base_dir + "train\\"
        self.test_dataset_dir = self.dataset_base_dir + "test\\"
        self.classifier_information_path = self.BASE_DIR + "classifier_information.json"
        self.audio_information_path = self.BASE_DIR + "audio_information.json"
        self.raw_data_file = "data.npy"
        self.fourier_data_file = "spectogram.npy"
        self.all_data_files = ["data.npy", "spectogram.npy"]

        self.audio_information = Utility.get_information(self.audio_information_path)
        self.classifier_information = Utility.get_information(self.classifier_information_path)
        self.sampling_rate = self.audio_information["sampling_rate"]
        self.blocksize = self.audio_information["blocksize"]
        self.record_duration = self.audio_information["record_duration"]
        self.frequency_data = np.fft.fftfreq(self.sampling_rate*self.record_duration, d=1/self.sampling_rate)

        self.right_data_urls = None
        self.wrong_data_urls = None
    def log(self, text):
        #print(text)
        ...

    def get_numpy_data(self, folder_path, data_list, file_name):
        for file in os.listdir(folder_path):
            self.log('Analyzing file: ' + file)
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    self.log('...The file is a FILE')
                    if file == file_name:
                        self.log('......The file is a fourier data file')
                        fourier_data_list.append(Utility.load_numpy_data(file_path))
                        return fourier_data_list
                elif os.path.isdir(file_path):
                    self.log('...The file is a DIRECTORY.Analyzing the directory')
                    fourier_data_list = self.get_numpy_data(file_path, fourier_data_list, file_name)
            except Exception as e:
                self.log(e)
        return fourier_data_list
    def get_all_data_url(self, folder_path, data_list):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    for i in range(len(self.all_data_files)):
                        if file == self.all_data_files[i]:
                            data_list[i].append(file_path)
                elif os.path.isdir(file_path):
                    data_list = self.get_all_data_url(file_path, data_list)
            except Exception as e:
                self.log(e)
        return data_list


    def get_right_data_paths(self, folder, file_list):
        for file in os.listdir(folder):
            self.log('Analyzing File: ' + file)
            file_path = os.path.join(folder, file)
            try:
                if os.path.isdir(file_path):
                    self.log('...The file is a DIRECTORY')
                    if file == self.data_folder:
                        self.log('......The directory contains right data')
                        file_list.append(file_path)
                        return file_list
                    else:
                        self.log('...Searching the directory for the right data folder')
                        file_list = self.get_right_data_paths(file_path, file_list)

            except Exception as e:
                self.log(e)
        return file_list

    def get_right_data(self, shuffle_data=True, max_size=20, test=False):
        if self.right_data_urls is None:
            right_data_paths = []
            print('Acquiring data path of the right data folders containing the target data: ' + self.data_folder)
            if not test:
                right_data_paths = self.get_right_data_paths(self.train_dataset_dir, right_data_paths)
            else:
                right_data_paths = self.get_right_data_paths(self.test_dataset_dir, right_data_paths)
            print(str(len(right_data_paths)) + " folders found containing the target data")
            all_data_urls = []
            for i in range(len(self.all_data_files)):
                all_data_urls.append([])
            print("Collecting data of " + str(len(all_data_urls)) + " types")
            for path in right_data_paths:
                all_data_urls = self.get_all_data_url(path, all_data_urls)
            print(str(self.all_data_files))
            print(str(all_data_urls))

            print('Total '+ self.all_data_files[0] + ' files: ' + str(len(all_data_urls[0])))
            self.right_data_urls = all_data_urls
        if shuffle_data:
            random.shuffle(self.right_data_urls[0])
        print("Right data urls: " + str(self.right_data_urls))
        if max_size < len(self.right_data_urls[0]):
            result = []
            for data in self.right_data_urls:
                result.append(data[:max_size])
            return result
        else:
            return self.right_data_urls
    def get_wrong_data_urls(self, folder_name, wrong_data_urls):
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            try:
                if os.path.isfile(file_path) and (not (self.data_folder in file_path)):
                    for i in range(len(self.all_data_files)):
                        if file == self.all_data_files[i]:
                            wrong_data_urls[i].append(file_path)
                elif os.path.isdir(file_path):
                    wrong_data_urls = self.get_wrong_data_urls(file_path, wrong_data_urls)
            except Exception as e:
                self.log(e)
        return wrong_data_urls
    def get_wrong_data(self, shuffle_data=True, max_size=20, test=False):
        if self.wrong_data_urls is None:
            wrong_data_urls = []
            for i in range(len(self.all_data_files)):
                wrong_data_urls.append([])
            if not test:
                wrong_data_urls = self.get_wrong_data_urls(self.train_dataset_dir, wrong_data_urls)
            else:
                wrong_data_urls = self.get_wrong_data_urls(self.test_dataset_dir, wrong_data_urls)
            print("Wrong data Loaded: " + str(len(wrong_data_urls[0])) + " " +
                  self.all_data_files[0] + " files and " + str(len(wrong_data_urls[1])) + " " +
                  self.all_data_files[1] + " files")
            self.wrong_data_urls = wrong_data_urls
        if shuffle_data:
            random.shuffle(self.wrong_data_urls[0])
        if max_size < len(self.wrong_data_urls[0]):
            result = []
            for data in self.wrong_data_urls:
                result.append(data[:max_size])
            return result
        else:
            return self.wrong_data_urls

    # -------------------------------------------------------------
    def create_audio_tag(self, data, labels):
        meta_filename =labels[0]


if __name__ == "__main__":
    data_name = "english"
    dc = DatasetCollector(data_name)
    data = dc.get_wrong_data()
    print("Collected data size: " + str(len(data[0])))
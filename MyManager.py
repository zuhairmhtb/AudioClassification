import time, struct, wave, os
import numpy as np
from scipy import signal
import scipy.io.wavfile as wavfile
from Classifier import Classifier
class StreamManager:
    def __init__(self, audio_streamer, audio_queue, record_dir, classifier, update_classification_function, processing_update_fn):
        self.audio_queue = audio_queue
        self.task_interval = 10 # Minutes
        self.audio_streamer = audio_streamer
        self.BASE_DIR = record_dir
        self.predicted_output_dir = self.BASE_DIR
        self.update_prediction = update_classification_function
        self.update_processing_fn = processing_update_fn
        self.classifier = classifier



    def mkdir(self, folder_url):
        if not os.path.exists(folder_url):
            os.makedirs(folder_url)
    def log(self, text):
        print(text)
    def run(self):
        folder_name = str(int(round(time.time() * 1000))) + "\\"

        if len(self.audio_queue) > 0:
            merged_data = self.audio_queue[0]
            print("Saving data to folder")
            for i in range(1, len(self.audio_queue)):
                print("Adding data")
                merged_data = np.concatenate((merged_data, self.audio_queue[i]), axis=0)
            t, f, s = signal.spectrogram(merged_data.reshape((merged_data.shape[0])), fs=self.audio_streamer.sampling_rate)
            output = np.asarray([t, f, s])
            #fourier_data = np.fft.fft(merged_data, axis=0)
            #freq = np.fft.fftfreq(self.audio_streamer.audio_block_size, d=1/self.audio_streamer.sampling_rate)
            #print("Saving data")
            prediction = self.classifier.get_prediction(merged_data)
            if prediction[0]['classes'] == 0:
                pred = "Not Talking"
            elif prediction[0]['classes'] == 1:
                pred = "Talking"
            prob = "{0:.2f}%".format((100 * np.amax(prediction[0]['probabilities'])))
            print("PREDICTION: " + str(pred))
            folder_url = self.predicted_output_dir + str(pred) + "\\" + folder_name
            self.mkdir(folder_url)
            #np.save(folder_url+'fourier', fourier_data)
            #np.save(folder_url+'freq', freq)
            np.save(folder_url+'data', merged_data)
            #print("Saving wave file")

            wavfile.write(folder_url+'audio.wav', self.audio_streamer.sampling_rate, merged_data)
            np.save(folder_url + 'spectogram', output)
            #self.save_as_wav(merged_data, len(merged_data), 'NONE', 'NONE', self.audio_streamer.channels, 2,
            #                self.audio_streamer.sampling_rate, folder_url+'audio.wav', 'h')

            self.update_prediction(str(pred))
            self.update_processing_fn("Classification Complete [ Confidence: " + prob + " ]")
        print("Thread complete")
    @staticmethod
    def save_as_wav(wave_data, nframes, comptype, compname, nchannels, sampwidth, sampling_rate, file, bit_format='i'):
        wav_file = wave.open(file, 'wb')
        wav_file.setparams((nchannels, sampwidth, int(sampling_rate), int(nframes), comptype, compname))
        if nchannels == 1:
            for s in wave_data:
                byte_data = struct.pack(bit_format, int(s))
                print("Writing wave data")
                wav_file.writeframes(byte_data)
        elif nchannels == 2:
            for i in range(len(wave_data[0])):
                byte_data = struct.pack('<' + bit_format + bit_format, int(wave_data[0][i]), int(wave_data[1][i]))
                wav_file.writeframes(byte_data)
        wav_file.close()
import math, wave, time, queue, struct, sys, json
from win32api import GetSystemMetrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from threading import Thread
from PyQt5.QtWidgets import QApplication, QPushButton, QDialog,\
    QHBoxLayout, QLabel, QVBoxLayout, QGroupBox, QSlider, QMessageBox,\
    QLineEdit, QScrollArea, QWidget
from scipy import signal
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from MyNet.temp.MyManager import StreamManager
from MyNet.temp.Classifier import Classifier

class AudioStreamer:
    def __init__(self, audio_plotter, audio_blocksize, sampling_rate, channels, record_duration, update_duration_function, classification_type, classification_update_function, update_processing_fn):
        self.update_duration = update_duration_function
        self.audio_plotter = audio_plotter
        self.sampling_rate = sampling_rate
        self.audio_block_size = audio_blocksize
        self.channels = channels
        self.input_device = None
        self.downsample = 10
        self.amplitude = 1
        self.device_info = None

        self.recorded_sample_duration = record_duration # Seconds
        self.record_dir = None
        queue_size = int((self.recorded_sample_duration * self.sampling_rate)/self.audio_block_size)+1

        self.audio_queue = queue.Queue(maxsize=queue_size)
        self.is_updating = False
        self.audio_data = queue.Queue()
        self.stop_streaming = True
        self.audio_streamer = None
        self.recorded_duration = 0
        self.recorded_duration_total = 0
        self.mute = False
        self.classification_type = classification_type
        self.classification_update_function = classification_update_function
        self.processing_update_function = update_processing_fn
        self.classifier = None
        self.classifier = Classifier(self.classification_type)
       # self.output_stream = sd.Stream(channels=self.channels, dtype='int32',samplerate=self.sampling_rate,
        #                               latency='low')

    def log(self, text):
        #print(text)
        ...
    def init_streamer(self):
        self.device_info = sd.query_devices(self.input_device, kind='input')
        self.log('Device Info: ' + self.device_info)
        self.sampling_rate =self.device_info['default_samplerate']
        self.log('Sampling rate: ' + str(self.sampling_rate))

    def update_audio_feed(self, indata, outdata, frames, time, status):
        amplified_data = np.multiply(indata, self.amplitude)
        if not self.audio_queue.full():
            self.log('Received audio data of length ' + str(indata.shape)
                     + ", Total Frames: " + str(frames))
            if not self.is_updating:
                self.audio_queue.put(np.copy(amplified_data))
        elif not ((self.record_dir is None) and self.audio_queue.full()):
            print("Audio Queue is full and the audio sample of length " + str(indata.shape) + " could not be added")
            self.processing_update_function("Running Classification...")
            temp = []
            self.is_updating = True
            while not self.audio_queue.empty():
                temp.append(self.audio_queue.get())
            self.is_updating = False
            man = StreamManager(self, temp, self.record_dir, self.classifier, self.classification_update_function, self.processing_update_function)
            thread = Thread(target=man.run)
            thread.start()
        if self.mute:
            amplified_data = np.multiply(indata, 0)
        self.audio_plotter.add(indata)
        current_duration = indata.shape[0] / self.sampling_rate
        self.recorded_duration += current_duration
        self.recorded_duration_total += current_duration
        if not (self.update_duration is None):
            self.update_duration(self.recorded_duration, self.recorded_duration_total)
        outdata[:] = amplified_data

    def stream(self):
        self.stop_streaming = False
        self.audio_streamer = sd.Stream(device=self.input_device, channels=self.channels,
                                        samplerate=self.sampling_rate, dtype='int16',
                                        callback=self.update_audio_feed, blocksize=self.audio_block_size,
                                        latency='low')
        #self.output_stream.start()
        with self.audio_streamer:
            while not self.stop_streaming:
                time.sleep(5)
            self.recorded_duration = 0
        #self.output_stream.stop()

class StreamController(QDialog):
    def __init__(self):
        super().__init__()
        self.BASE_DIR = "D:\\thesis\\ConvNet\\MyNet\\temp\\"
        data = self.get_information()
        self.prediction_type = "talk"
        self.sampling_rate = int(data["sampling_rate"])
        self.sample_width = int(data["sample_width"])
        self.blocksize = int(data["blocksize"])
        self.channels = int(data["channels"])
        self.amplitude = int(data["amplitude"])
        self.record_duration = int(data["record_duration"])
        self.audio_plotter = AudioPlotter(self.blocksize, self.sampling_rate, self)
        self.audio_streamer = AudioStreamer(self.audio_plotter, self.blocksize,
                                            self.sampling_rate, self.channels, self.record_duration,
                                            self.set_duration, self.prediction_type, self.update_prediction, self.update_processing)
        #self.audio_processor = AudioProcessor(self.audio_streamer, self, self.sampling_rate, self.channels)

        self.audio_streamer_thread = None
        #self.audio_processor_thread = None
        self.audio_plotter_thread = None
        self.WINDOW_TOP = 100
        self.WINDOW_LEFT = 100
        self.WINDOW_WIDTH = GetSystemMetrics(0)-100
        self.WINDOW_HEIGHT = GetSystemMetrics(1)-100
        self.TITLE = "Audio Signal Processing"
        self.fig = plt.figure()
        self.ax=[]
        self.ax.append(self.fig.add_subplot(4, 1, 1))
        self.ax.append(self.fig.add_subplot(4, 1, 2))
        self.ax.append(self.fig.add_subplot(4, 1, 3))
        self.ax.append(self.fig.add_subplot(4, 1, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)


        self.root_layout = None

        self.prediction_type_label = QLabel('Type:', self)
        self.prediction_type_data = QLabel(self.prediction_type.upper(), self)
        self.prediction_processing_label = QLabel("Inactive", self)
        self.prediction_output_type = QLabel('Last Predicted:', self)
        self.prediction_output_data = QLabel('N/A', self)
        self.start_stream_button = QPushButton('Start Streaming', self)
        self.stop_stream_button = QPushButton('Stop Streaming', self)
        self.mute_audio_button = QPushButton('Mute Audio', self)
        self.record_dir_input = QLineEdit(self)
        self.record_dir_input.setText('other')
        #self.record_button = QPushButton('Start/Stop Record',self)
        #self.save_recorded_button = QPushButton('Save Recorded', self)
        self.amplitude_slider = QSlider()
        self.amplitude_slider.setMinimum(0)
        self.amplitude_slider.setMaximum(10)
        self.amplitude_slider.setTickInterval(1)
        self.amplitude_slider.setValue(1)

        self.recorded_duration_label = QLabel('Last Recorded Duration: 0 '.upper()+'seconds', self)
        self.recorded_duration_total_label = QLabel('Total Recorded Duration: 0 '.upper()+'seconds', self)
        self.audio_info_labels = []
        for key in data:
            self.audio_info_labels.append(QLabel(str(key).upper().replace('_', ' ') + ": " + str(data[key])))
        self.root_layout = QVBoxLayout()
        self.root_row_buttons_layout = QHBoxLayout()
        self.root_row_info_layout = QHBoxLayout()
        self.root_row_plot_layout = QHBoxLayout()
        self.root_row_classification_layout = QHBoxLayout()
        self.root_row_plot = QGroupBox("Plot Tab")
        self.root_row_plot.setLayout(self.root_row_plot_layout)
        self.root_row_data_scroller = QScrollArea()

        self.root_row_data = QWidget(self.root_row_data_scroller)
        self.root_row_data_layout = QVBoxLayout()
        self.root_row_data.setLayout(self.root_row_data_layout)
        self.root_row_info = QGroupBox("Information")
        self.root_row_info.setLayout(self.root_row_info_layout)
        self.root_row_buttons = QGroupBox('Buttons')
        self.root_row_buttons.setLayout(self.root_row_buttons_layout)
        self.root_row_classification = QGroupBox("Classification")
        self.root_row_classification.setLayout(self.root_row_classification_layout)
        self.initWindow()
    def get_information(self):
        file_path = self.BASE_DIR + 'audio_information.json'
        with open(file_path) as f:
            data = json.load(f)
        return data

    def log(self, text):
        #print(text)
        ...
    def set_duration(self, last_duration, total_duration):
        last_duration_minute = str(int(last_duration/60)) + '.' + str(int(last_duration%60))
        total_duration_minute = str(int(total_duration / 60)) + '.' + str(int(total_duration % 60))
        self.recorded_duration_label.setText('Last Recorded Duration: '.upper()+ last_duration_minute + " minutes")
        self.recorded_duration_total_label.setText('Total Recorded Duration: '.upper()+ total_duration_minute + " minutes")
    def update_processing(self, text):
        self.prediction_processing_label.setText(text)


    def initWindow(self):
        self.start_stream_button.setDisabled(False)
        self.stop_stream_button.setDisabled(True)
        #self.save_recorded_button.setDisabled(True)
        #self.record_button.setDisabled(True)
        self.start_stream_button.clicked.connect(self.start_streaming)
        self.stop_stream_button.clicked.connect(self.stop_streaming)
        self.mute_audio_button.clicked.connect(self.mute_audio)
        #self.record_button.clicked.connect(self.record_audio)
        #self.save_recorded_button.clicked.connect(self.save_recorded)
        self.amplitude_slider.valueChanged.connect(self.change_amplitude)

        self.root_row_plot_layout.addWidget(self.canvas)
        self.root_row_buttons_layout.addWidget(self.toolbar)
        self.root_row_buttons_layout.addWidget(self.start_stream_button)
        self.root_row_buttons_layout.addWidget(self.stop_stream_button)
        self.root_row_buttons_layout.addWidget(self.mute_audio_button)
        #self.root_row_buttons_layout.addWidget(self.record_button)
        #self.root_row_buttons_layout.addWidget(self.save_recorded_button)
        self.root_row_buttons_layout.addWidget(self.record_dir_input)
        self.root_row_buttons_layout.addWidget(self.amplitude_slider)

        self.root_row_info_layout.addWidget(self.recorded_duration_label)
        self.root_row_info_layout.addWidget(self.recorded_duration_total_label)
        for label in self.audio_info_labels:
            self.root_row_info_layout.addWidget(label)


        self.root_row_classification_layout.addWidget(self.prediction_type_label)
        self.root_row_classification_layout.addWidget(self.prediction_type_data)
        self.root_row_classification_layout.addWidget(self.prediction_output_type)
        self.root_row_classification_layout.addWidget(self.prediction_output_data)
        self.root_row_classification_layout.addWidget(self.prediction_processing_label)

        self.root_row_data_scroller.setWidget(self.root_row_data)

        self.root_row_data_scroller.setWidgetResizable(True)
        self.root_row_data_layout.addWidget(self.root_row_buttons)

        self.root_row_data_layout.addWidget(self.root_row_info)
        self.root_row_data_layout.addWidget(self.root_row_classification)
        self.root_layout.addWidget(self.root_row_plot)
        self.root_layout.addWidget(self.root_row_data_scroller)
        self.setLayout(self.root_layout)
        self.setGeometry(self.WINDOW_TOP, self.WINDOW_LEFT, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setWindowTitle('Audio Streamer')

        self.show()
    def change_amplitude(self):
        self.audio_streamer.amplitude = self.amplitude_slider.value()
    def mute_audio(self):
        self.audio_streamer.mute = not self.audio_streamer.mute
    def update_prediction(self, text):
        self.prediction_output_data.setText(text)
    def start_streaming(self):
        if self.audio_streamer.stop_streaming:
            self.start_stream_button.setDisabled(True)
            #self.save_recorded_button.setDisabled(True)
            self.log('Starting Audio Streamer...')
            self.audio_streamer.record_dir = self.BASE_DIR + "dataset\\predicted\\" + self.record_dir_input.text() + '\\'
            self.audio_streamer_thread = Thread(target=self.audio_streamer.stream)
            self.audio_streamer_thread.setDaemon(True)
            self.audio_streamer_thread.start()
            self.log('Audio Streamer Started')
            #if self.audio_processor.stop_processing:
             #   self.log('Starting Audio Processor...')
              #  self.audio_processor_thread = Thread(target=self.audio_processor.process)
               # self.audio_processor_thread.setDaemon(True)
               # self.audio_processor_thread.start()
               # self.log('Audio Processor started')

            #else:
             #   self.log('Audio Processor is already running...')

            if self.audio_plotter.stop_plotting:
                self.log('Starting Audio Plotter...')
                self.audio_plotter_thread = Thread(target=self.audio_plotter.plot)
                self.audio_plotter_thread.start()
                self.log('Audio Plotter Thread started')
            else:
                self.log('Audio Plotter is already running...')
            self.stop_stream_button.setDisabled(False)
            #self.record_button.setDisabled(False)
            self.log("Started Streaming")
        else:
            self.log('Already Streaming')
    def stop_streaming(self):
        if not self.audio_streamer.stop_streaming:
            self.prediction_processing_label.setText("Inactive")
            self.stop_stream_button.setDisabled(True)
            #self.record_button.setDisabled(True)
            self.audio_streamer.stop_streaming = True
            self.log('Stopping Audio Streamer...')
            if self.audio_streamer_thread.is_alive():
                self.audio_streamer_thread.join()
            self.log('Audio Streamer stopped')
            self.audio_streamer_thread = None
            #if not self.audio_processor.stop_processing:
             #  self.log('Stopping Audio Processor...')
              #  if self.audio_processor_thread.is_alive():
               #     self.audio_processor_thread.join()
               # self.log('Audio Processor stopped')
               # self.audio_processor_thread = None

           # else:
            #    self.log('Audio processor is already stopped...')

            if not self.audio_plotter.stop_plotting:
                self.audio_plotter.stop_plotting = True
                self.log('Stopping Audio plotter')
                if self.audio_plotter_thread.is_alive():
                    self.audio_plotter_thread.join()
                self.log('Audio Plotter stopped')
                self.audio_plotter_thread = None
            else:
                self.log('Audio plotter is already stopped...')
            self.start_stream_button.setDisabled(False)
            #self.save_recorded_button.setDisabled(False)
            self.log('Stopped Streaming')

        else:
            self.log('Audio Streamer is already stopped...')
    def record_audio(self):
        if not self.audio_processor.is_recording:
            self.audio_processor.recorded_data = []
            QMessageBox.about(self, 'Recording', 'Started Recording Audio')
        else:
            QMessageBox.about(self, 'Recording', 'Stopped Recording Audio. Total Length: ' + str(len(self.audio_processor.recorded_data)))

        self.audio_processor.is_recording = not self.audio_processor.is_recording
    def save_recorded(self):
        if len(self.audio_processor.recorded_data) > 0:
            file_name = str(int(round(time.time() * 1000)))
            file_path = self.BASE_DIR + file_name + '.wav'

            merged_data = self.audio_processor.recorded_data[0]
            for i in range(1, len(self.audio_processor.recorded_data)):
                merged_data = np.concatenate((merged_data, self.audio_processor.recorded_data[i]), axis=0)
            print("Merged data shape: ", merged_data.shape)

            buttonReply = QMessageBox.question(self, 'Save Audio',
                                               'Are you Sure you want to save the Audio?')
            if buttonReply == QMessageBox.Yes:
                thread = Thread(target=self.save_as_wav,
                                args=(merged_data, len(merged_data),
                                      'NONE', 'NONE',
                                      self.audio_processor.channels, self.sample_width,
                                      self.audio_processor.sampling_rate, file_path))
                thread.start()
                thread.join()
            QMessageBox.about(self, 'Save Audio', "Your audio has been saved at" + file_path)
            #self.audio_processor.save_as_wav(np.multiply(merged_data, self.amplitude_slider.value()), len(merged_data),
             #                                'NONE', 'NONE',
              #                               self.audio_processor.channels, 2, self.audio_processor.sampling_rate, file_path)

            print("File Saved")

    @staticmethod
    def save_as_wav(wave_data, nframes, comptype, compname, nchannels, sampwidth, sampling_rate, file, bit_format='i'):
        wav_file = wave.open(file, 'wb')
        wav_file.setparams((nchannels, sampwidth, int(sampling_rate), int(nframes), comptype, compname))
        if nchannels == 1:
            for s in wave_data:
                byte_data = struct.pack(bit_format, int(s))
                wav_file.writeframes(byte_data)
        elif nchannels == 2:
            for i in range(len(wave_data[0])):
                byte_data = struct.pack('<' + bit_format + bit_format, int(wave_data[0][i]), int(wave_data[1][i]))
                wav_file.writeframes(byte_data)
        wav_file.close()
class AudioProcessor:
    def __init__(self, audio_streamer, stream_controller, sampling_rate, channels):
        self.sampling_rate =sampling_rate
        self.channels = channels
        self.audio_streamer = audio_streamer
        self.stream_controller = stream_controller
        self.stop_processing = True
        self.recorded_data = []
        self.is_recording = False
    def process(self):
        if not (self.audio_streamer is None):
            self.stop_processing = False
            while (not self.stop_processing) or (not self.audio_streamer.audio_queue.empty()):
                if not self.audio_streamer.audio_queue.empty():
                    data = np.copy(self.audio_streamer.audio_queue.get())
                    self.stream_controller.amplitude = self.stream_controller.amplitude_slider.value()
                    if self.stream_controller.amplitude > 1:
                        data = np.multiply(data, self.stream_controller.amplitude)
                    if self.is_recording:
                        self.recorded_data.append(data)

                    #print("Processing data of shape: " + str(data.shape))
                    self.stream_controller.set_duration(self.audio_streamer.recorded_duration, self.audio_streamer.recorded_duration_total)
                time.sleep(10)




class AudioPlotter:
    def __init__(self, blocksize, sampling_rate, stream_Controller, channels=1):
        self.audio_queue = queue.Queue()
        self.is_adding = False
        self.is_plotting = False
        self.plot_data = None
        self.accumulated_data_time = 10
        self.channels = channels
        self.accumulated_data = np.zeros((sampling_rate * self.accumulated_data_time, channels))
        self.stop_plotting = True
        self.stream_controller = stream_Controller
        self.blocksize = blocksize
        self.sampling_rate = sampling_rate
        self.freq = np.fft.fftfreq(blocksize, d=1 / sampling_rate)
        self.a = math.log(math.pow(2, 1 / 12))

    def add(self, data):
        if not self.is_plotting:
            self.is_adding = True
            self.plot_data = np.copy(data)
            self.is_adding = False
    def use_mel_filter(self, signal, sample_rate, nfilt=40, NFFT=512, frame_duration=25, pre_emphasis=0.97, stride_duration=10):
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        frame_size = frame_duration / 1000
        frame_stride = stride_duration / 1000

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        # Make sure that we have at least 1 frame
        num_frames = int(
            np.ceil(
                float(np.abs(signal_length - frame_length)) / frame_step
            )
        )
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        pad_signal = np.append(emphasized_signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        low_freq_mel = 0
        # Convert Hz to Mel
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        # Equally spaced in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        # Convert Mel to Hz
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)


        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # Left
            f_m = int(bin[m])  # Center
            f_m_plus = int(bin[m + 1])  # Right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        return filter_banks

    def update_view(self):
        #print("Updating plot")

        if (not (self.plot_data is None)) and (not self.is_adding):
            self.is_plotting = True
            data = self.plot_data
            scaled_data = np.divide(data, np.amax(np.abs(data))).astype(np.float32)
            fourier_data = np.fft.fft(scaled_data, axis=0)
            t, freq, spectrum = signal.spectrogram(scaled_data.reshape((scaled_data.shape[0])), fs=self.sampling_rate,
                                                   window='hamming')
            print("Fourier data shape:", fourier_data.shape)

            shift = len(data)
            self.accumulated_data = np.roll(self.accumulated_data, -shift, axis=0)
            self.accumulated_data[-shift:, :] = data
            self.plot_data = None
            self.is_plotting = False


            self.stream_controller.ax[0].cla()
            self.stream_controller.ax[0].plot(self.accumulated_data)

            self.stream_controller.ax[1].cla()
            self.stream_controller.ax[1].plot(scaled_data)
            self.stream_controller.ax[1].set_xlabel('No. of input samples')
            self.stream_controller.ax[1].set_ylabel('Amplitude')

            self.stream_controller.ax[2].cla()

            # spectrum, freq, t, im = self.stream_controller.ax[1].specgram(self.accumulated_data[:, 0],
            #                                                             Fs=self.sampling_rate, NFFT=1024)

            self.stream_controller.ax[2].pcolormesh(freq, t, spectrum, cmap='gist_earth')

            self.stream_controller.ax[2].set_xlabel("Time")
            self.stream_controller.ax[2].set_ylabel("Frequency")

            self.stream_controller.ax[3].cla()
            self.stream_controller.ax[3].plot(self.freq, fourier_data.real, label='Real')
            self.stream_controller.ax[3].legend()
            self.stream_controller.ax[3].set_xlabel('Frequency')
            self.stream_controller.ax[3].set_ylabel('Density')
            max_freq = int(self.freq[np.argmax(np.abs(fourier_data.real))])
            self.stream_controller.ax[3].text(10, 0.8,
                                              'Frequency: ' + str(max_freq) + " Hz" + ", Note: " + self.get_note(
                                                  max_freq))

            self.stream_controller.canvas.draw()
    def get_note(self, freq):
        print(freq)
        notes=['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        sign = 1
        if freq < 0:
            sign = -1
        known_freq = 440
        freq = np.abs(freq)
        if freq > 0:
            ratio = math.log(freq/known_freq)
            print("Ratio:", str(ratio))
            note_step = sign * int(math.floor(ratio/self.a))
            print(str(note_step))
            return notes[note_step%len(notes)]
        return ''
    def plot(self):
        if self.stop_plotting:
            self.stop_plotting = False
            while not self.stop_plotting:
                self.update_view()
                time.sleep(0.5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = StreamController()
    sys.exit(app.exec_())

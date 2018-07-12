# AudioClassification and SignalProcessing
This software is a demonstration of Audio Signal Processing and Machine Learning using Python and Tensorflow. The software architecture maintains flexibility for storing and processing audio input data i.e audio samples for which classification needs to be performed is saved inside a directory possessing a unique name. The folder name then needs to be mentioned in the classifier file after which a network is created/loaded automatically.  The software contains a GUI that can stream audio via webcams or external audio devices connected to the computer and process(perform classification and store data accordingly) the audio in real time using a Convolutional and/or a Recurrent Neural Network in order to perform audio classification like speech recognition, music classification, etc. (Depending on how the network was trained). The data set can be arranged in directories where the name of a parent directory represents a classification class. In this way a single network can be trained for multiple types of binary independent audio data eventually building a complex neural network. As a result any classifier can be made just by typing the name of a directory for which classification has to be performed. Audio data inside the directory represents the positive(correct) output class and rest of the data in other directories represent the negative(incorrect) ouput class.
The settings used in the software are as follows:

1. Audio Stream Settings/Parameters (audio_information.json):
a. Sampling rate: 44100Hz
b. Sample width: 2
c. Audio Block Size: 11000
d. Each sample record duration: 3 seconds (44100 *3 + extra) = 132,300 + 10,700 = 143000
e. Audio Channels: 1

2. Adjustable Classifier Settings/Parameters (classifier_information.json):
a. Number of layers (Convolutional Neural Network) : Input(1), Convolution(1), Pool(1), Dense(1), Output(1)
b. Input Type: Frequency Spectogram of Audio Waveform(wav format) sliced into 'n' pieces to create a sense of temporal dimension passed through Mel Filter Bank.
c. Input Width: 143000
d. Input Height and Channel/Depth: 1
e. Input data format: 2D(WidthxChannel) Numpy array of 'wav' audio data (saved in data.npy file)
f. Filters for Convolution Layer: 3
g. Activation Function: ReLU
g. Convolution Stride: 2
h. Convolution Kernel Width: 4
i. Pool Kernel Size: 2
j. Pool Stride: 2
k. Dense Layer Units: 512
l. Dense Dropout rate: 0.4
m. Output Classes: 2
n. Train batch size: 200
o. Train Steps: 80
p. Train epoch/iteration: 50

3. Files, Folders and their Functions:

a. MyStreamer.py: Creates a GUI using PyQt5 and Matplotlib in order to display output for audio stream, classification and in order to provide an interface to interact with the software. The audio stream is recorded using 'sounddevice' module of python. The interface displays 'Audio waveform for total input stream', 'Audio waveform for current audio input sample', 'Spectogram for current audio sample', 'DFT Output Graph for Frequency Vs Amplitude with identification of Audio Note with highest Frequency', 'Buttons for controlling Audio Stream', 'Information about Audio Settings' and 'Classification Output for current audio sample'.

b. MyManager.py: A threaded background object that receives each audio input sample from queue and creates blocks of Audio samples with specified duration(3 seconds) in order to save the output as a numpy array and perform classification using a pretrained CNN. It finaly displays output of the prediction and saves the audio samples in the following directory structure:
    -- Predicted/other/ (Directory)
      -- Parent Directory/ (According to Classification output)
        -- Data Directory/ (Named according to current time in millis)
          -- data.npy (Audio data in nupy format)
          -- audio.wav (Audio sample)
          -- spectogram.npy (Spectogram of the audio data-Obsolete)
          
c. DatasetManager.py: Manages the input dataset for the Classifier. It collects input data as URL from 'Train' and 'Test' directoris of the input dataset for a specific Classifier. As for example, if the training directory contains a folder named 'Sound' for which a network needs to be created and trained then the name of the folder should be mentioned in 'classifier_name' variable of Classifier.py. The audio input data present inside directories and sub-directories of the folder are then gathered for 'Correct' output type of the classifier and rest of the dataset are considered as 'Incorrect' output of the classifier. The network is then trained accordingly.

d. Classifier.py: Creates/Loads a Convolutional Neural Network for classification of audio input signals. The model for each classifier is saved in 'networks' directory.

e. dataset(Directory): Contains input dataset for Train, Predicted and Test type data of the network. Each of the three folders contains the following data hierarchy:
  -- Test/ (Root directory)
    -- FolderName/ (A binary CNN can be created for classification of data inside each of the folder-Only the name of folder for which                      classification has to be performed needs to be mentioned in Classifier.py)
      -- Sub-Folders/
      -- Data Folder/ (The name of each data folder is time in millis when the data was saved)
        -- audio.wav (The audio sample which is saved)
        -- data.npy (The numpy array of the input data)
        -- spectogram.py (Spectogram of the input data-Obsolete: is now performed inside classifiers)
        
 f. networks(Directory): Contains saved model for each trained classifier. The folder name of each model is the type of data(in dataset directory) for which binary classification was performed). It contains two child directories (cnn and rnn) inside which the models are saved. The directory structure is as follows:
  -- cnn/ (Root directory)
    -- ModelName/ (Model directory-TensorFlow)
      -- eval/ (Directory that contains model and log data for evaluation of the network)
      -- training model data (checkpoint, events, graphs, etc.)
  g. audio_information.json: Contains parameters for Audio Input Stream.
  h. classifier_information.json: Contains parameters and hyperparameters for CNN.
  

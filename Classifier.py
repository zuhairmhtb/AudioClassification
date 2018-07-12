import tensorflow as tf
import sounddevice as sd
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os
from MyNet.temp.DatasetManager import DatasetCollector
tf.logging.set_verbosity(tf.logging.INFO)
class Classifier:
    def __init__(self, classifier_name):
        self.classifier_type = "cnn"
        self.classifier_name = classifier_name
        self.classification_input_type = "mfb"
        self.dm = DatasetCollector(classifier_name)
        self.inp_w = self.dm.classifier_information["input_layer"]["width"]
        self.inp_chan = self.dm.classifier_information["input_layer"]["depth"]
        self.conv_filters = self.dm.classifier_information["convolution_layer"]["filters"]
        self.conv_padding = self.dm.classifier_information["convolution_layer"]["padding"]
        self.conv_activation = tf.nn.relu
        self.conv_stride = self.dm.classifier_information["convolution_layer"]["stride"]
        self.conv_kernel_width = self.dm.classifier_information["convolution_layer"]["kernel_width"]
        self.pool_size = self.dm.classifier_information["pool_layer"]["kernel_width"]
        self.pool_stride = self.dm.classifier_information["pool_layer"]["stride"]
        self.pool_padding = self.dm.classifier_information["pool_layer"]["padding"]
        self.dense_units = self.dm.classifier_information["dense_layer"]["units"]
        self.dense_activation = tf.nn.relu
        self.dense_dropout = self.dm.classifier_information["dense_layer"]["dropout"]
        self.output_classes = self.dm.classifier_information["output_layer"]["classes"]

        self.prediction_class_key = "classes"
        self.prediction_probability_key = "probabilities"
        self.prediction_probability_name = "softmax_tensor"
        self.input_feature_tag = "x"
        if self.classifier_type == "rnn":
            self.model_dir = self.dm.network_base_dir + "rnn\\" + self.classifier_name + "\\"
        else:
            self.model_dir = self.dm.network_base_dir + "cnn\\" + self.classifier_name + "\\"
        self.logger = self.get_logger()
        self.batch_size = 200
        self.train_epochs = None
        self.train_steps = 80
        self.train_iteration = 50
        self.log_iterations = 10
        self.train_classifier = self.get_training_classifier()
        self.predict_classifier = self.get_prediction_classifier()
        self.evaluation_classifier = self.get_evaluation_classifier()
        self.pool_output_size = 71498
        if self.classification_input_type == "mfb":
            self.inp_w = 322 * 40
            self.pool_output_size = 6438
        # Recurrent Neural Network(RNN) params
        elif self.classifier_type == "rnn":
            self.inp_w = 44100
        self.rnn_chunks = 10
        self.rnn_chunk_size = int(self.inp_w/self.rnn_chunks)
        self.rnn_size = int(self.inp_w/self.rnn_chunks)
        print("Chunk size: " + str(self.rnn_chunk_size))
        self.x_rnn = tf.placeholder('float', [None, self.rnn_chunks, self.rnn_chunk_size])
        self.y_rnn = tf.placeholder('int32')

        print("Initiating weights and variables of the RNN")
        self.rnn_layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.output_classes])),
                     'biases': tf.Variable(tf.random_normal([self.output_classes]))}



    def get_logger(self):
        iterations = 10
        tensors_to_log = {self.prediction_probability_key: self.prediction_probability_name}
        return tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=iterations)
    def get_training_classifier(self):
        return tf.estimator.Estimator(
            model_fn=self.train, model_dir=self.model_dir)
    def get_prediction_classifier(self):
        return tf.estimator.Estimator(
            model_fn=self.predict, model_dir=self.model_dir)
    def get_evaluation_classifier(self):
        return tf.estimator.Estimator(
            model_fn=self.evaluate, model_dir=self.model_dir)
    def get_input_function(self, input_data, input_labels, batch_size, epochs, shuffle=False):
        # Train the model
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={self.input_feature_tag: input_data},
            y=input_labels,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=shuffle)
        return input_fn
    def train(self, features, labels):
        mode = tf.estimator.ModeKeys.TRAIN
        if self.classifier_type == "rnn":
            output_res, predictions = self.recurrent_neural_network_model(features, mode)
        else:
            output_res, predictions = self.execute(features, mode)
        # Calculate Loss (for both TRAIN and EVAL modes)
        print("Tensor Label :" + str(labels))
        print("Predictions:" + str(predictions[self.prediction_class_key]))
        one_hot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.output_classes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_res)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        print("Loss, Global step = ", str(loss), ", ", str(tf.train.get_global_step()))
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
    def predict(self, features, params):
        print("Received Parameters:", str(params))
        mode = tf.estimator.ModeKeys.PREDICT
        if self.classifier_type == "rnn":
            output_res, predictions = self.recurrent_neural_network_model(features, mode)
        else:
            output_res, predictions = self.execute(features, mode)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    def evaluate(self, features, labels):
        mode = tf.estimator.ModeKeys.EVAL
        if self.classifier_type == "rnn":
            output_res, predictions = self.recurrent_neural_network_model(features,mode)
        else:
            output_res, predictions = self.execute(features, mode)
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_res)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)

    def recurrent_neural_network_model(self, data, mode):
        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_chunk_size, state_is_tuple=True)
        print("Transposing input data to split data")
        print("Input data shape: ", str(data[self.input_feature_tag]))
        x_rnn = tf.transpose(data[self.input_feature_tag], [1, 0, 2])
        print("Transposed data shape: ", str(x_rnn))
        x_rnn = tf.reshape(x_rnn, [-1, self.rnn_chunk_size])
        print("Reshaped data shape: ", str(x_rnn))
        x_rnn = tf.split(x_rnn, self.rnn_chunks, 0)
        print("Split data shape: ", str(x_rnn))
        outputs, states = rnn.static_rnn(lstm_cell, x_rnn, dtype=tf.float32)

        dense = tf.layers.dense(inputs=outputs[-1], units=self.dense_units, activation=self.dense_activation)
        dense_dropout = tf.layers.dropout(inputs=dense, rate=self.dense_dropout,
                                          training=mode == tf.estimator.ModeKeys.TRAIN)
        output = tf.layers.dense(inputs=dense_dropout, units=self.output_classes)
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            self.prediction_class_key: tf.argmax(input=output, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            self.prediction_probability_key: tf.nn.softmax(output, name=self.prediction_probability_name)
        }

        return output, predictions
    def execute(self, data, mode):
        outputs = []
        print("Input feature: " + str(data[self.input_feature_tag]))
        print("Input feature:" + str(type(data[self.input_feature_tag])))
        print(data[self.input_feature_tag])
        input_layer_output = tf.reshape(data[self.input_feature_tag], [-1, self.inp_w, self.inp_chan])
        conv_output = tf.layers.conv1d(inputs=input_layer_output, filters=self.conv_filters, kernel_size=self.conv_kernel_width)
        print("Conv output shape: " + str(conv_output))
        pool_output = tf.layers.max_pooling1d(conv_output, self.pool_size, self.pool_stride, self.pool_padding)
        print("Pool shape: " + str(pool_output))
        flattened_pool = tf.reshape(pool_output, [-1, self.conv_filters*self.pool_output_size])
        print("Flattened pool: " + str(flattened_pool))
        dense_output = tf.layers.dense(inputs=flattened_pool, units=self.dense_units, activation=self.dense_activation)
        dense_dropout = tf.layers.dropout(inputs=dense_output, rate=self.dense_dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
        final_output = tf.layers.dense(inputs=dense_dropout, units=self.output_classes)
        print("Final Output: " + str(final_output))
        outputs.append(input_layer_output)
        outputs.append(conv_output)
        outputs.append(pool_output)
        outputs.append(dense_output)
        outputs.append(final_output)
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            self.prediction_class_key: tf.argmax(input=final_output, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            self.prediction_probability_key: tf.nn.softmax(final_output, name=self.prediction_probability_name)
        }
        return final_output, predictions

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
    def convert_all_url_to_tensor(self, input_data_urls, input_data_labels):
        c = list(zip(input_data_urls, input_data_labels))
        np.random.shuffle(c)
        input_data_urls, input_data_labels = zip(*c)
        print("Shuffled data labels: " + str(input_data_labels))
        input_data_np = np.empty((0, self.inp_w, self.inp_chan), dtype=np.float32)
        input_label_tensor = tf.one_hot(input_data_labels, self.output_classes)
        for i in range(len(input_data_urls)):
            url = input_data_urls[i]
            data = np.load(url, allow_pickle=True)
            data = np.divide(data, np.amax(np.abs(data), axis=0)).astype(np.float32)
            if self.classification_input_type == "slope":
                slope_list = [y - x for x, y in zip(data, data[1:])]
                slope_list = slope_list + [0]
                data = np.asarray(slope_list).reshape((-1, data.shape[1])).astype(np.float32)
            elif self.classification_input_type == "mfb":
                filter = self.use_mel_filter(data, 44100)
                print("Filter Shape: " + str(filter.shape))
                data = filter.reshape((-1, data.shape[1])).astype(np.float32)
            # input_data_np[i, :, :] = data
            print("Input data shape: " + str(data.shape))
            print("Numpy data shape: " + str(input_data_np.shape))
            if data.shape[0] == input_data_np.shape[1]:
                input_data_np = np.concatenate((input_data_np, np.reshape(data, (1, data.shape[0], data.shape[1]))))
            elif data.shape[0] > input_data_np.shape[1]:
                diff = data.shape[0] - input_data_np.shape[1]
                input_data_np = np.concatenate(
                    (input_data_np, np.reshape(data[:-diff, :], (1, input_data_np.shape[1], data.shape[1]))))
        print("Numpy input data shape: " + str(input_data_np.shape) + ", type: " + str(input_data_np.dtype))
        print(input_label_tensor)
        # init = tf.global_variables_initializer()
        # with tf.Session() as sess:
        #   sess.run(init)
        label_np = np.asarray(input_data_labels)
        print("Label Shape: " + str(label_np.shape))
        return input_data_np, label_np
    def start_train(self):
        for i in range(self.train_iteration):
            right_data = self.dm.get_right_data(shuffle_data=True, max_size=self.batch_size)
            wrong_data = self.dm.get_wrong_data(shuffle_data=True, max_size=self.batch_size)

            right_raw_data_urls = right_data[0]
            right_raw_data_labels = [[1]] * len(right_raw_data_urls)
            wrong_raw_data_urls = wrong_data[0]
            wrong_raw_data_labels = [[0]] * len(wrong_raw_data_urls)


            print("Right data batch size: " + str(len(right_raw_data_urls)),
                  ", Type: " + str(type(right_raw_data_urls)))
            print('Right data batch label size: ' + str(len(right_raw_data_labels)))
            print("Right data url: " + str(right_raw_data_urls) + ", Type: " + str(type(right_raw_data_urls)))
            print('Wrong data batch label size: ' + str(len(wrong_raw_data_labels)))

            input_data_urls = right_raw_data_urls + wrong_raw_data_urls
            input_data_labels = right_raw_data_labels + wrong_raw_data_labels
            input_data_np, label_np = self.convert_all_url_to_tensor(input_data_urls, input_data_labels)
            train_queue = self.get_input_function(input_data_np, label_np, input_data_np.shape[0],
                                                  self.train_epochs, shuffle=True)
            result = self.train_classifier.train(input_fn=train_queue, steps=self.train_steps, hooks=[self.logger])
            print("Result:", str(type(result)), ", Shape:", str(np.shape(result)))
            self.demo_prediction(evaluate=True, max_size=50)


    def get_prediction(self, data):
        input_data_np = np.empty((0, self.inp_w, self.inp_chan))
        data = np.divide(data, np.amax(np.abs(data), axis=0)).astype(np.float32)
        if self.classification_input_type == "slope":
            data_list = data.tolist()
            slope_list = [y - x for x, y in zip(np.asarray(data_list), np.asarray(data_list[1:]))]
            slope_list = slope_list + [0]
            data = np.asarray(slope_list).reshape(-1, data.shape[1]).astype(np.float32)
        elif self.classification_input_type == "mfb":
            filter = self.use_mel_filter(data, 44100)
            print("Filter Shape: " + str(filter.shape))
            data = filter.reshape((-1, data.shape[1])).astype(np.float32)
        # input_data_np[i, :, :] = data
        print("Input data shape: " + str(data.shape))
        print("Numpy data shape: " + str(input_data_np.shape))
        if data.shape[0] == input_data_np.shape[1]:
            input_data_np = np.reshape(data, (1, data.shape[0], data.shape[1]))
        elif data.shape[0] > input_data_np.shape[1]:
            diff = data.shape[0] - input_data_np.shape[1]
            input_data_np = np.reshape(data[:-diff, :], (1, input_data_np.shape[1], data.shape[1]))
        print("Input tensor shape: " + str(input_data_np.shape))
        #print(input_data_np)

        predict_queue = self.get_input_function(input_data_np.astype(np.float32), None, 1,
                                                1, shuffle=False)
        result = self.predict_classifier.predict(predict_queue, hooks=None, predict_keys=None)
        return list(result)
    def get_evaluation(self, data, label):
        input_data_np = np.empty((0, self.inp_w, self.inp_chan))
        data = np.divide(data, np.amax(np.abs(data), axis=0)).astype(np.float32)
        # input_data_np[i, :, :] = data
        #print("Input data shape: " + str(data.shape))
        #print("Numpy data shape: " + str(input_data_np.shape))
        if data.shape[0] == input_data_np.shape[1]:
            input_data_np = np.concatenate((input_data_np, np.reshape(data, (1, data.shape[0], data.shape[1]))))
        elif data.shape[0] > input_data_np.shape[1]:
            diff = data.shape[0] - input_data_np.shape[1]
            input_data_np = np.concatenate(
                (input_data_np, np.reshape(data[:-diff, :], (1, input_data_np.shape[1], data.shape[1]))))
        #print("Input tensor shape: " + str(input_data_np.shape))
        # print(input_data_np)

        evaluate_queue = self.get_input_function(input_data_np.astype(np.float32), np.asarray(label), 1,
                                                 1, shuffle=False)
        result = self.evaluation_classifier.evaluate(evaluate_queue, hooks=[self.logger])
        prediction = self.get_prediction(data)
        return result, prediction

    def demo_prediction(self, evaluate=False, max_size=2):
        if not evaluate:
            fnames = ["data_talk1", "data_talk2", "data_sound", "data_talk3"]
            file = "D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\test\\"
            data = []
            labels = []
            for i in range(len(fnames)):
                file_path = file + fnames[i] + ".npy"
                np_data = np.load(file_path)
                print("Data " + str(i + 1) + " shape: " + str(np_data.shape))
                data.append(np_data)
                if self.classifier_name in fnames[i]:
                    labels.append([1])
                else:
                    labels.append([0])
            for i in range(len(data)):
                evaluation, prediction = classifier.get_prediction(data[i])
                print("Prediction : " + str(list(prediction)))
        else:
            wrong_data = self.dm.get_wrong_data(test=True, max_size=max_size)[0]
            wrong_label = [[0]] * len(wrong_data)
            right_data = self.dm.get_right_data(test=True, max_size=max_size)[0]
            right_label = [[1]] * len(right_data)
            all_data = wrong_data + right_data
            labels = wrong_label + right_label
            data_np, label_np = self.convert_all_url_to_tensor(all_data, labels)
            evaluate_queue = self.get_input_function(data_np, label_np, data_np.shape[0],
                                                     1, shuffle=False)
            evaluation = self.evaluation_classifier.evaluate(evaluate_queue, hooks=[self.logger])
            print("Evaluation Accuracy : " + str(evaluation['accuracy']))
            print('Provided target: ' + str(labels))






if __name__ == "__main__":
    classifier = Classifier("talk")
    classifier.start_train()
    #classifier.demo_prediction(evaluate=True, max_size=20)










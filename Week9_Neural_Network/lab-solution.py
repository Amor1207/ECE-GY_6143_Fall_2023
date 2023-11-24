# -*- coding: utf-8 -*-
"""8-lab-neural-net-music-classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16VowRHvWeRLiveTsCif2iYapjJMyNRTF

# Assignment: Neural Networks for Music Classification

*Fraida Fund*

**TODO**: Edit this cell to fill in your NYU Net ID and your name:

-   **Net ID**:
-   **Name**:

⚠️ **Note**: This experiment is designed to run on a Google Colab **GPU** runtime. You should use a GPU runtime on Colab to work on this assignment. You should not run it outside of Google Colab. However, if you have been using Colab GPU runtimes a lot, you may be alerted that you have exhausted the “free” compute units allocated to you by Google Colab. If that happens, you do not have to purchase compute units - use a CPU runtime instead, and modify the experiment as instructed for CPU-only runtime.

In this assignment, we will look at an audio classification problem. Given a sample of music, we want to determine which instrument (e.g. trumpet, violin, piano) is playing.

*This assignment is closely based on one by Sundeep Rangan, from his [IntroML GitHub repo](https://github.com/sdrangan/introml/).*
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
# %matplotlib inline

"""## Audio Feature Extraction with Librosa

The key to audio classification is to extract the correct features. The `librosa` package in python has a rich set of methods for extracting the features of audio samples commonly used in machine learning tasks, such as speech recognition and sound classification.
"""

import librosa
import librosa.display
import librosa.feature

"""In this lab, we will use a set of music samples from the website:

<http://theremin.music.uiowa.edu>

This website has a great set of samples for audio processing.

We will use the `wget` command to retrieve one file to our Google Colab storage area. (We can run `wget` and many other basic Linux commands in Colab by prefixing them with a `!` or `%`.)
"""

!wget "http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/SopSax.Vib.pp.C6Eb6.aiff"

"""Now, if you click on the small folder icon on the far left of the Colab interface, you can see the files in your Colab storage. You should see the “SopSax.Vib.pp.C6Eb6.aiff” file appear there.

In order to listen to this file, we’ll first convert it into the `wav` format. Again, we’ll use a magic command to run a basic command-line utility: `ffmpeg`, a powerful tool for working with audio and video files.
"""

aiff_file = 'SopSax.Vib.pp.C6Eb6.aiff'
wav_file = 'SopSax.Vib.pp.C6Eb6.wav'

!ffmpeg -y -i $aiff_file $wav_file

"""Now, we can play the file directly from Colab. If you press the ▶️ button, you will hear a soprano saxaphone (with vibrato) playing four notes (C, C#, D, Eb)."""

import IPython.display as ipd
ipd.Audio(wav_file)

"""Next, use `librosa` command `librosa.load` to read the audio file with filename `audio_file` and get the samples `y` and sample rate `sr`."""

y, sr = librosa.load(aiff_file)

"""Feature engineering from audio files is an entire subject in its own right. A commonly used set of features are called the Mel Frequency Cepstral Coefficients (MFCCs). These are derived from the so-called mel spectrogram, which is something like a regular spectrogram, but the power and frequency are represented in log scale, which more naturally aligns with human perceptual processing.

You can run the code below to display the mel spectrogram from the audio sample.

You can easily see the four notes played in the audio track. You also see the 'harmonics' of each notes, which are other tones at integer multiples of the fundamental frequency of each note.
"""

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.amplitude_to_db(S),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

"""## Downloading the Data

Using the MFCC features described above, [Prof. Juan Bello](http://steinhardt.nyu.edu/faculty/Juan_Pablo_Bello) at NYU Steinhardt and his former PhD student Eric Humphrey have created a complete data set that can used for instrument classification. Essentially, they collected a number of data files from the website above. For each audio file, the segmented the track into notes and then extracted 120 MFCCs for each note. The goal is to recognize the instrument from the 120 MFCCs. The process of feature extraction is quite involved. So, we will just use their processed data.

To retrieve their data, visit

<https://github.com/marl/dl4mir-tutorial/tree/master>

and note the password listed on that page. Click on the link for “Instrument Dataset”, enter the password, click on `instrument_dataset` to open the folder, and download it. (You can “direct download” straight from this site, you don’t need a Dropbox account.) Depending on your laptop OS and on how you download the data, you may need to “unzip” or otherwise extract the four `.npy` files from an archive.

Then, upload the files to your Google Colab storage: click on the folder icon on the left to see your storage, if it isn’t already open, and then click on “Upload”.

🛑 Wait until *all* uploads have completed and the orange “circles” indicating uploads in progress are *gone*. (The training data especially will take some time to upload.) 🛑

Then, load the files with:
"""

Xtr = np.load('uiowa_train_data.npy')
ytr = np.load('uiowa_train_labels.npy')
Xts = np.load('uiowa_test_data.npy')
yts = np.load('uiowa_test_labels.npy')

"""Examine the data you have just loaded in:

-   How many training samples are there?
-   How many test samples are there?
-   What is the number of features for each sample?
-   How many classes (i.e. instruments) are there?

Write some code to find these values and print them.
"""

len(np.unique(ytr))

# TODO -  get basic details of the data
# compute these values from the data, don't hard-code them
n_tr    = Xtr.shape[0]
n_ts    = Xts.shape[0]
n_feat  = Xtr.shape[1]
n_class = len(np.unique(ytr))

# now print those details
print("Num training= %d" % n_tr)
print("Num test=     %d" % n_ts)
print("Num features= %d" % n_feat)
print("Num classes=  %d" % n_class)

"""Then, standardize the training and test data, `Xtr` and `Xts`, by removing the mean of each feature and scaling to unit variance.

You can do this manually, or using `sklearn`'s [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). (For an example showing how to use a `StandardScaler`, you can refer to the notebook on regularization.)

Although you will scale both the training and test data, you should make sure that both are scaled according to the mean and variance statistics from the *training data only*.

<small>Standardizing the input data can make the gradient descent work better, by making the loss function “easier” to descend.</small>
"""

# TODO - Standardize the training and test data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtr_scale = scaler.fit_transform(Xtr)
Xts_scale = scaler.transform(Xts)

"""## Building a Neural Network Classifier

Following the example in the demos you have seen, clear the keras session. Then, create a neural network `model` with:

-   `nh=256` hidden units in a single dense hidden layer
-   `sigmoid` activation at hidden units
-   select the input and output shapes, and output activation, according to the problem requirements. Use the variables you defined earlier (`n_tr`, `n_ts`, `n_feat`, `n_class`) as applicable, rather than hard-coding numbers.

Print the model summary.
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# TODO - construct the model
nh = 256

# Initialize the model
model = Sequential()

# Add a dense hidden layer
model.add(Dense(nh, input_shape=(n_feat,), activation='sigmoid'))

# Add the output layer
model.add(Dense(n_class, activation='softmax'))

# show the model summary
model.summary()

# you can also visualize the model with
tf.keras.utils.plot_model(model, show_shapes=True)

"""Create an optimizer and compile the model. Select the appropriate loss function for this multi-class classification problem, and use an accuracy metric. For the optimizer, use the Adam optimizer with a learning rate of 0.001"""

# TODO - create optimizer and compile the model
opt = Adam(learning_rate=0.001)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""Fit the model for 10 epochs using the scaled data for both training and validation, and save the training history in \`hist.

Use the `validation_data` option to pass the *test* data. (This is OK because we are not going to use this data as part of the training process, such as for early stopping - we’re just going to compute the accuracy on the data so that we can see how training and test loss changes as the model is trained.)

Use a batch size of 128. Your final accuracy should be greater than 99%.
"""

# TODO - fit model and save training history
hist = model.fit(Xtr_scale, ytr,
                 epochs=10,
                 batch_size=128,
                 validation_data=(Xts_scale, yts))

"""Plot the training and validation accuracy saved in `hist.history` dictionary, on the same plot. This gives one accuracy value per epoch. You should see that the validation accuracy saturates around 99%. After that it may “bounce around” a little due to the noise in the stochastic mini-batch gradient descent.

Make sure to label each axis, and each series (training vs. validation/test).
"""

# TODO - plot the training and validation accuracy in one plot
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""Plot the training and validation loss values saved in the `hist.history` dictionary, on the same plot. You should see that the training loss is steadily decreasing. Use the [`semilogy` plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.semilogy.html) so that the y-axis is log scale.

Make sure to label each axis, and each series (training vs. validation/test).
"""

# TODO - plot the training and validation loss in one plot
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Plotting training and validation loss with a logarithmic scale for the y-axis
plt.figure(figsize=(10, 6))
plt.semilogy(epochs, train_loss, 'bo-', label='Training Loss')
plt.semilogy(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()

plt.show()

"""## Varying training hyperparameters

One challenge in training neural networks is the selection of the **training hyperparameters**, for example:

-   learning rate
-   learning rate decay schedule
-   batch size
-   optimizer-specific hyperparameters (for example, the `Adam` optimizer we have been using has `beta_1`, `beta_2`, and `epsilon` hyperparameters)

and this challenge is further complicated by the fact that all of these training hyperparameters interact with one another.

(Note: **training hyperparameters** are distinct from **model hyperparameters**, like the number of hidden units or layers.)

Sometimes, the choice of training hyperparameters affects whether or not the model will find an acceptable set of weights at all - i.e. whether the optimizer converges.

It’s more often the case, though, that **for a given model**, we can arrive at a set of weights that have similar performance in many different ways, i.e. with different combinations of optimizer hyperparameters. However, the \*training cost** in both **time** and **energy\*\* will be very much affected.

In this section, we will explore these further.

Repeat your model preparation and fitting code, but try four learning rates as shown in the vector `rates`. In each iteration of the loop:

-   use `K.clear_session()` to free up memory from models that are no longer in scope. (Note that this does not affect models that are still “in scope”!)
-   construct the network
-   select the optimizer. Use the Adam optimizer with the learning rate specific to this iteration
-   train the model for 20 epochs (make sure you are training a *new* model in each iteration, and not *continuing* the training of a model created already outside the loop)
-   save the history of training and validation accuracy and loss for this model
"""

rates = [0.1, 0.01,0.001,0.0001]

# To store the history of each model
histories = {}

for lr in rates:
    # Clearing the Keras session to free up memory
    K.clear_session()

    # Construct the network
    model = Sequential()
    model.add(Dense(nh, input_shape=(n_feat,), activation='sigmoid'))
    model.add(Dense(n_class, activation='softmax'))

    # Select the optimizer with the current learning rate
    opt = Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(Xtr_scale, ytr, epochs=20, batch_size=128, validation_data=(Xts_scale, yts))

    # Save the history
    histories[lr] = history

"""Plot the training loss vs. the epoch number for all of the learning rates on one graph (use `semilogy` again). You should see that the lower learning rates are more stable, but converge slower, while with a learning rate that is too high, the gradient descent may fail to move towards weights that decrease the loss function.

Make sure to label each axis, and each series.

**Comment on the results.** Given that all other optimizer hyperparameters are fixed, what is the effect of varying learning rate on the training process?

A learning rate that is too high can lead to instability and prevent the model from converging to a good solution.
A learning rate that is too low can lead to slow convergence, requiring more epochs and hence more computational resources and time.
An appropriately chosen learning rate provides a balance between the speed of convergence and the stability of the training process.
"""

# TODO - plot showing the training process for different learning rates
plt.figure(figsize=(12, 8))

for lr, history in histories.items():
    plt.semilogy(history.epoch, history.history['loss'], label=f'LR = {lr}')

plt.title('Training Loss vs. Epoch Number for Different Learning Rates')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (log scale)')
plt.legend()
plt.grid(True)
plt.show()

"""In the previous example, we trained each model for a fixed number of epochs. Now, we’ll explore what happens when we vary the training hyperparameters, but train each model to the same validation **accuracy target**. We will consider:

-   how much *time* it takes to achieve that accuracy target (“time to accuracy”)
-   how much *energy* it takes to achieve that accuracy target (“energy to accuracy”)
-   and the *test accuracy* for the model, given that it is trained to the specified validation accuracy target

#### Energy consumption

To do this, first we will need some way to measure the energy used to train the model. We will use [Zeus](https://ml.energy/zeus/overview/), a Python package developed by researchers at the University of Michigan, to measure the GPU energy consumption.

**Note**: if you are running this experiment in a CPU-only runtime, you should skip this section on energy comsumption. Continue with the ” `TrainToAccuracy` callback” section.

First, install the package:
"""

!pip install zeus-ml

"""Then, import it, and tell it to monitor your GPU:"""

# from zeus.monitor import ZeusMonitor

# monitor = ZeusMonitor(gpu_indices=[0])

"""When you want to measure GPU energy usage, you will:

-   start a “monitoring window”
-   do your GPU-intensive computation (e.g. call `model.fit`)
-   stop the “monitoring window”

and then you can get the time and total energy used by the GPU in the monitoring window.

Try it now - this will just continue fitting whatever `model` is currently in scope from previous cells:
"""

# monitor.begin_window("test")
# model.fit(Xtr_scale, ytr, epochs=5)
# measurement = monitor.end_window("test")
# print("Measured time (s)  :" , measurement.time)
# print("Measured energy (J):" , measurement.total_energy)

"""#### `TrainToAccuracy` callback

Next, we need a way to train a model until we achieve our desired validation accuracy. We will [write a callback function](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks) following these specifications:

-   It will be called `TrainToAccuracy` and will accept two arguments: a `threshold` and a `patience` value.
-   If the model’s validation accuracy is higher than the `threshold` for `patience` epochs in a row, stop training.
-   In the `on_epoch_end` function, which will be called at the end of every epoch during training, you should get the current validation accuracy using `currect_acc = logs.get("val_accuracy")`. Then, set `self.model.stop_training = True` if the condition above is met.
-   The default values of `threshold` and `patience` are given below, but other values may be passed as arguments at runtime.

Then, when you call `model.fit()`, you will add the `TrainToAccuracy` callback as in

    callbacks=[TrainToAccuracy(threshold=0.98, patience=5)]
"""

# TODO - write a callback function
class TrainToAccuracy(callbacks.Callback):
    def __init__(self, threshold=0.9, patience=3):
        super(TrainToAccuracy, self).__init__()
        self.threshold = threshold  # The desired accuracy threshold
        self.patience = patience  # How many epochs to wait once hitting the threshold
        self.wait = 0  # Counter for the number of epochs where threshold is met

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get("val_accuracy")
        if current_acc and current_acc > self.threshold:
            self.wait += 1
        else:
            self.wait = 0

        if self.wait >= self.patience:
            self.model.stop_training = True
            print(f"\nReached {self.threshold*100}% accuracy, so stopping training after {epoch+1} epochs!")

"""Try it! run the following cell to test your `TrainToAccuracy` callback. (This will just continue fitting whatever `model` is currently in scope.)"""

model.fit(Xtr_scale, ytr, epochs=100, validation_split = 0.2, callbacks=[TrainToAccuracy(threshold=0.95, patience=5)])

"""Your model shouldn’t *really* train for 100 epochs - it should stop training as soon as 95% validation accuracy is achieved for 5 epochs in a row! (Your “test” is not graded, you may change the `threshold` and `patience` values in this “test” call to `model.fit` in order to check your work.)

Note that since we are now using the validation set performance to *decide* when to stop training the model, we are no longer “allowed” to pass the test set as `validation_data`. The test set must never be used to make decisions during the model training process - only for evaluation of the final model. Instead, we specify that 20% of the training data should be held out as a validation set, and that is the validation accuracy that is used to determine when to stop training.

### See how TTA/ETA varies with learning rate, batch size

Now, you will repeat your model preparation and fitting code - with your new `TrainToAccuracy` callback - but in a loop. First, you will iterate over different learning rates.

In each iteration of each loop, you will prepare a model (with the appropriate training hyperparameters) and train it until:

-   either it has achieved **0.95 accuracy for 3 epoches in a row** on a 20% validation subset of the training data,
-   or, it has trained for 500 epochs

whichever comes FIRST.

For each model, you will record:

-   the training hyperparameters (learning rate, batch size)
-   the number of epochs of training needed to achieve the target validation accuracy
-   the accuracy on the *test* data (not the validation data!). After fitting the model, use `model.evaluate` and pass the scaled *test* data to get the test loss and test accuracy
-   **GPU runtime**: the GPU energy and time to train the model to the desired validation accuracy, as computed by a `zeus-ml` measurement window that starts just before `model.fit` and ends just after `model.fit`.
-   **CPU runtime**: the time to train the model to the desired validation accuracy, as computed by the difference in `time.time()` just before `model.fit` and just after `model.fit`.
"""

# TODO - iterate over learning rates and get TTA/ETA

# default learning rate and batch size -
lr = 0.001
batch_size = 128

metrics_vs_lr = []

# Iterating over different learning rates
for lr in [0.1, 0.01, 0.001, 0.0001]:

    # Clearing the Keras session to free up memory
    K.clear_session()

    # Construct the model
    model = Sequential()
    model.add(Dense(nh, input_shape=(n_feat,), activation='sigmoid'))
    model.add(Dense(n_class, activation='softmax'))  # Assuming ytr.shape[1] is the number of classes

    # Compile the model with the current learning rate
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Start measurement

    start_time = time.time()

    # Fit the model
    history=model.fit(Xtr_scale, ytr, epochs=500, batch_size=batch_size, validation_split=0.2, callbacks=[TrainToAccuracy(threshold=0.95, patience=5)])

    # End measurement

    total_time = time.time() - start_time

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(Xts_scale, yts)

    # Save metrics
    model_metrics = {
        'batch_size': 128,
        'learning_rate': lr,
        'epochs': len(history.history['accuracy']),
        'test_accuracy': test_accuracy,
        'train_time': total_time
    }

    metrics_vs_lr.append(model_metrics)

metrics_vs_lr

"""Next, you will visualize the results.

**GPU runtime instructions**: Create a figure with four subplots. In each subplot, create a bar plot with learning rate on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.

**CPU runtime instructions**: Create a figure with three subplots. In each subplot, create a bar plot with learning rate on the horizontal axis and (1) Time to accuracy, (2) Test accuracy, (3) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.
"""

# TODO - visualize effect of varying learning rate, when training to a target accuracy
# Extracting the data for plotting
learning_rates = [m['learning_rate'] for m in metrics_vs_lr]
time_to_accuracy = [m['train_time'] for m in metrics_vs_lr]
test_accuracies = [m['test_accuracy'] for m in metrics_vs_lr]
epochs = [m['epochs'] for m in metrics_vs_lr]
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Converting learning rates to strings for better display on the x-axis
learning_rate_labels = [str(lr) for lr in learning_rates]

# Subplot 1: Time to Accuracy
axes[0].bar(learning_rate_labels, time_to_accuracy, color='blue')
axes[0].set_title('Time to Accuracy vs Learning Rate')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Time to Accuracy (seconds)')

# Subplot 2: Test Accuracy
axes[1].bar(learning_rate_labels, test_accuracies, color='green')
axes[1].set_title('Test Accuracy vs Learning Rate')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Test Accuracy')

# Subplot 3: Epochs
axes[2].bar(learning_rate_labels, epochs, color='red')
axes[2].set_title('Epochs vs Learning Rate')
axes[2].set_xlabel('Learning Rate')
axes[2].set_ylabel('Epochs')

# Adjusting x-axis and y-axis for better readability
for ax in axes:
    ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels for clarity

plt.tight_layout()
plt.show()

"""**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the learning rate on the training process?

Note: because of the stochastic nature of neural network training AND in the compute resource, these measurements can be very “noisy”. Look for overall trends, but don’t be concerned with small differences from one experiment to the next, or with occasional “outlier” results. Also note that if the number of epochs is 500, this is an indication that the target validation accuracy was *not* reached in 500 epochs!

Training Time:



A higher learning rate can lead to faster convergence, meaning the model may reach the target validation accuracy in fewer epochs. However, if the learning rate is too high, it might cause the model to overshoot the minimum of the loss function or even diverge, leading to increased training time or failure to converge.

A lower learning rate ensures more gradual and potentially more stable convergence. However, it may require more epochs to reach the target accuracy, resulting in longer training times. Too low a learning rate can lead to excessively slow convergence, also increasing training time.



Energy Consumption (GPU Runtime):



When using GPUs, the energy consumption is also an important consideration. A higher learning rate might reduce the number of epochs needed to train, potentially lowering total energy consumption. However, this is contingent on the model converging properly.

A lower learning rate, while potentially more stable, could increase the number of epochs needed and thus the overall energy consumption.



Finding the Balance:



The key is to find a balanced learning rate that allows for efficient convergence without overshooting or getting stuck in local minima.

Adaptive learning rate methods (like Adam, RMSprop, etc.) can dynamically adjust the learning rate during training, potentially offering a more efficient path to convergence.

Now, you will repeat, with a loop over different batch sizes -
"""

# TODO - iterate over batch size and get TTA/ETA

# default learning rate and batch size -
lr = 0.001

metrics_vs_bs = []
for batch_size in [64, 128, 256, 512, 1024, 2048]:

    # Clearing the Keras session to free up memory
    K.clear_session()

    # Construct the model
    model = Sequential()
    model.add(Dense(nh, input_shape=(n_feat,), activation='sigmoid'))
    model.add(Dense(n_class, activation='softmax'))  # Assuming ytr.shape[1] is the number of classes

    # Compile the model with the current learning rate
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Start measurement

    start_time = time.time()

    # Fit the model
    history=model.fit(Xtr_scale, ytr, epochs=500, batch_size=batch_size, validation_split=0.2, callbacks=[TrainToAccuracy(threshold=0.95, patience=5)])

    # End measurement

    total_time = time.time() - start_time

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(Xts_scale, yts)

    # Save metrics
    model_metrics = {
        'batch_size': 128,
        'batch_size': batch_size,
        'epochs': len(history.history['accuracy']),
        'test_accuracy': test_accuracy,
        'train_time': total_time
    }

    metrics_vs_bs.append(model_metrics)

"""Next, you will visualize the results.

**GPU runtime instructions**: Create a figure with four subplots. In each subplot, create a bar plot with batch size on the horizontal axis and (1) Time to accuracy, (2) Energy to accuracy, (3) Test accuracy, (4) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.

**CPU runtime instructions**: Create a figure with three subplots. In each subplot, create a bar plot with batch size on the horizontal axis and (1) Time to accuracy, (2) Test accuracy, (3) Epochs, on the vertical axis on each subplot, respectively. Use an appropriate vertical range for each subplot. Label all axes.
"""

# TODO - visualize effect of varying batch size, when training to a target accuracy

batch_size = [m['batch_size'] for m in metrics_vs_bs]
time_to_accuracy = [m['train_time'] for m in metrics_vs_bs]
test_accuracies = [m['test_accuracy'] for m in metrics_vs_bs]
epochs = [m['epochs'] for m in metrics_vs_bs]
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Converting learning rates to strings for better display on the x-axis
batch_size_labels = [str(bs) for bs in batch_size]

# Subplot 1: Time to Accuracy
axes[0].bar(batch_size_labels, time_to_accuracy, color='blue')
axes[0].set_title('Time to Accuracy vs Learning Rate')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Time to Accuracy (seconds)')

# Subplot 2: Test Accuracy
axes[1].bar(batch_size_labels, test_accuracies, color='green')
axes[1].set_title('Test Accuracy vs Learning Rate')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('Test Accuracy')

# Subplot 3: Epochs
axes[2].bar(batch_size_labels, epochs, color='red')
axes[2].set_title('Epochs vs Learning Rate')
axes[2].set_xlabel('Batch Size')
axes[2].set_ylabel('Epochs')

# Adjusting x-axis and y-axis for better readability
for ax in axes:
    ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels for clarity

plt.tight_layout()
plt.show()

"""**Comment on the results**: Given that the model is trained to a target validation accuracy, what is the effect of the batch size on the training process?

Note: because of the stochastic nature of neural network training AND in the compute resource, these measurements can be very “noisy”. Look for overall trends, but don’t be concerned with small differences from one experiment to the next, or with occasional “outlier” results. Also note that if the number of epochs is 500, this is an indication that the target validation accuracy was *not* reached in 500 epochs!

Time to Accuracy: The time required to reach the target validation accuracy appears to be highest for the smallest batch size (64) and decreases as the batch size increases to 256. Beyond this point, the time taken does not decrease significantly with further increases in batch size. This suggests that there is a diminishing return on reducing training time after a certain batch size threshold.

Test Accuracy: The test accuracy remains relatively stable across different batch sizes. This indicates that the batch size does not have a significant effect on the model's generalization to the test data, within the range of batch sizes provided.

Epochs: The number of epochs required to reach the target validation accuracy tends to decrease as the batch size increases. This is likely due to the fact that larger batch sizes provide a more accurate estimate of the gradient, leading to more efficient learning steps.
"""
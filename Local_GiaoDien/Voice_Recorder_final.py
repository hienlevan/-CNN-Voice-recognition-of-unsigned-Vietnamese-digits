import sounddevice as sd
from tkinter import *
import queue
import soundfile as sf
import threading
from tkinter import messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json 


data_path = 'C:/Users/HIEN/Desktop/temp/'
wavs_path = data_path + "wavs/"
metadata_path = data_path + "test.csv"

# Read metadata file and parse it
df_val = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
df_val.columns = ["file_name", "transcription", "normalized_transcription"]
df_val = df_val[["file_name", "normalized_transcription"]]
df_val = df_val.sample(frac=1).reset_index(drop=True)
print(df_val.head(1))

# The set of characters accepted in the transcription.
characters = [x for x in "abcghikmnostuy "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384


def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

batch_size = 2

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search

    #results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


model_architecture = data_path + "model.json"
model_weights = data_path + "model.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

# Optimizer
opt = keras.optimizers.Adam(learning_rate=1e-4)
# Compile the model and return
model.compile(optimizer=opt, loss=CTCLoss)

model.summary(line_length=110)


#Define the user interface
voice_rec = Tk()
voice_rec.geometry("360x300")
voice_rec.title("Voice Recorder")
voice_rec.config(bg="#107dc2")

#Create a queue to contain the audio data
q = queue.Queue()
#Declare variables and initialise them
recording = False
file_exists = False    

#Fit data into queue
def callback(indata, frames, time, status):
    q.put(indata.copy())

#Functions to play, stop and record audio
#The recording is done as a thread to prevent it being the main process
def threading_rec(x):
    if x == 1:
        #If recording is selected, then the thread is activated
        t1=threading.Thread(target= record_audio)
        t1.start()
    elif x == 2:
        #To stop, set the flag to false
        global recording
        recording = False
        messagebox.showinfo(message="Recording finished")
        data, fs = sf.read(wavs_path + "test.wav", dtype='int16') 
        ty_le_vertical = 100.0/32767.0
        n_sample = data.shape[0]
        WINDOW = n_sample // 340
        data_draw = []
        for i in range(0, 340):
            x = data[i*WINDOW]
            x = x * ty_le_vertical
            data_draw.append(i)
            data_draw.append(100 - x)
        wav_cvs.delete('all')
        wav_cvs.update()
        wav_cvs.create_line(data_draw)

    elif x == 3:
        #To play a recording, it must exist.
        if file_exists:
            #Read the recording if it exists and play it
            data, fs = sf.read(wavs_path + "test.wav", dtype='float32') 
            sd.play(data,fs)
            sd.wait()
        else:
            #Display and error if none is found
            messagebox.showerror(message="Record something to play")
    elif x == 4:
        # Define the validation dataset
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
        )
        validation_dataset = (
            validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        predictions = []
        targets = []
        for batch in validation_dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
        '''
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
        '''
        L = len(predictions)
        print('Bat dau')
        for i in range(L):
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
        print('Xong')

#Recording function
def record_audio():
    #Declare global variables    
    global recording 
    #Set to True to record
    recording= True   
    global file_exists 
    #Create a file to save the audio
    messagebox.showinfo(message="Recording Audio. Speak into the mic")
    with sf.SoundFile(wavs_path + "test.wav", mode='w', samplerate=22050,
                        channels=1) as file:
    #Create an input stream to record audio without a preset time
            with sd.InputStream(samplerate=22050, dtype='int16', channels=1, callback=callback):
                while recording == True:
                    #Set the variable to True to allow playing the audio later
                    file_exists =True
                    #write into file
                    file.write(q.get())

    
#Label to display app title
title_lbl  = Label(voice_rec, text="Voice Recorder", bg="#107dc2").grid(row=0, column=0, columnspan=3)

#Button to record audio
record_btn = Button(voice_rec, text="Record Audio", command=lambda m=1:threading_rec(m))
#Stop button
stop_btn = Button(voice_rec, text="Stop Recording", command=lambda m=2:threading_rec(m))
#Play button
play_btn = Button(voice_rec, text="Play Recording", command=lambda m=3:threading_rec(m))

#Recognition button
recog_btn = Button(voice_rec, text="Recognition", command=lambda m=4:threading_rec(m))

wav_cvs = Canvas(voice_rec, width = 340, height = 200, relief = SUNKEN, bd = 1) 

#Position buttons
record_btn.grid(row=1,column=1)
stop_btn.grid(row=1,column=0)
play_btn.grid(row=1,column=2)
recog_btn.grid(row=1,column=3)

wav_cvs.grid(row=2,column=0, columnspan = 4, pady = 10)

voice_rec.mainloop()


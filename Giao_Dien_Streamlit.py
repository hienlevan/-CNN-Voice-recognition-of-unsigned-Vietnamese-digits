
import streamlit as st
from Nhan_Dien_Audio import Nhan_Dien
from Ghi_Am_Audio import Ghi_Am
from Play_Audio import Phat_Am_Thanh



Header = "Project Nhận Diện Tiếng Nói Tiếng Việt"
st.header(Header)

Url_Image = "https://khaosat.hcmute.edu.vn/assets/img/login-banner.png"
st.image(Url_Image)

Student_Name = "Họ và tên: Lê Văn Hiền"
st.subheader(Student_Name)

Student_ID = "MSSV: 20110475"
st.subheader(Student_ID)

Teacher_Name = "Giáo viên hướng dẫn: Trần Tiến Đức"
st.title(Teacher_Name)

with st.form("Form_Ghi_Am"):
    if st.form_submit_button("Ghi Âm"):
        Ghi_Am()

with st.form("Form_Play_Am_Thanh"):
    if st.form_submit_button("Phát âm thanh vừa ghi"):
        Phat_Am_Thanh()        

with st.form("Form_Nhan_Dien"):
    if st.form_submit_button("Nhận diện"):
        Nhan_Dien()


def Chon_De_Xem():
    st.balloons()
    st.write("Hãy lựa chọn code mà bạn muốn xem !")        

def Show_Code_Ghi_Am():
    code = '''from tkinter import *
def Ghi_Am():
    import sounddevice as sd
    
    import queue
    import soundfile as sf
    import threading
    from tkinter import messagebox


    path_name = 'C:/Users/HIEN/Desktop/temp/wavs/'
    file_name = path_name + "test.wav"

    #Define the user interface
    voice_rec = Tk()
    voice_rec.geometry("360x200")
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
       
    #Recording function
    def record_audio():
        #Declare global variables    
        global recording 
        #Set to True to record
        recording= True   
        global file_exists 
        #Create a file to save the audio
        messagebox.showinfo(message="Recording Audio. Speak into the mic")
        path_name = 'C:/Users/HIEN/Desktop/temp/wavs/'
        file_name = path_name + "test.wav"

        with sf.SoundFile(file_name, mode='w', samplerate=22050,
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
    
    #Position buttons
    record_btn.grid(row=1,column=1)
    stop_btn.grid(row=1,column=0)
    voice_rec.mainloop()

'''
    st.code(code, language='python')  

def Show_Code_Phat_Am_Thanh():  
    code = '''

def Phat_Am_Thanh():
    import sounddevice as sd
    import soundfile as sf
    import streamlit as st

    wavs_path = 'C:/Users/HIEN/Desktop/temp/wavs/'
    file_exists = True
    if file_exists:
        #Read the recording if it exists and play it
        data, fs = sf.read(wavs_path + "test.wav", dtype='float32') 
        st.audio(data, format="audio/wav",start_time=0, sample_rate=fs)
        st.line_chart(data)
        #sd.play(data,fs)
        sd.wait()
    else:
        #Display and error if none is found
        st.info("Record something to play")'''  
    st.code(code, language='python')

def Show_Code_Nhan_Dien():
    code = '''def Nhan_Dien():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.models import model_from_json 

    import matplotlib.pyplot as plt
    from IPython import display
    from jiwer import wer

    import streamlit as st

    data_path = 'C:/Users/HIEN/Desktop/temp/'
    wavs_path = data_path + "wavs/"
    metadata_path = data_path + "test.csv"

    # Read metadata file and parse it
    df_val = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    df_val.columns = ["file_name", "transcription", "normalized_transcription"]
    df_val = df_val[["file_name", "normalized_transcription"]]
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    #print(df_val.head(1))
    st.info(df_val.head(1))
    #print(f"Size of the validation set: {len(df_val)}")
    st.info(f"Size of the validation set: {len(df_val)}")

    """
    ## Preprocessing
    We first prepare the vocabulary to be used.
    """

    # The set of characters accepted in the transcription.
    characters = [x for x in "abcghikmnostuy "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    st.info(
        f"The vocabulary is: {char_to_num.get_vocabulary()} "
        f"(size ={char_to_num.vocabulary_size()})"
    )

    """
    Next, we create the function that describes the transformation that we apply to each
    element of our dataset.
    """

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


    """
    ## Creating `Dataset` objects
    We create a `tf.data.Dataset` object that yields
    the transformed elements, in the same order as they
    appeared in the input.
    """

    batch_size = 2

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


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

    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
   
    L = len(predictions)
    st.info('Bắt đầu !')
    for i in range(L):
        st.balloons()
        st.success(f"Prediction: {predictions[i]}", icon="ℹ️")
        #st.info("-" * 100)
    st.success('Xong !', icon="ℹ️")'''
    st.code(code, language='python')

def Show_Code_Training():
    code = '''import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras import layers 
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer 

data_path = 'C:/Users/HIEN/Desktop/temp/Recording_And_Training/Dataset/TiengNoiTiengViet'
wavs_path = data_path + "/wavs/"
metadata_path = data_path + "/metadata.csv"

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
print(metadata_df.head(3))

"""
We now split the data into training and validation set.
"""

split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the validation set: {len(df_val)}")

"""
## Preprocessing
We first prepare the vocabulary to be used.
"""

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

"""
Next, we create the function that describes the transformation that we apply to each
element of our dataset.
"""
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

    """
## Creating `Dataset` objects
We create a `tf.data.Dataset` object that yields
the transformed elements, in the same order as they
appeared in the input.
"""

batch_size = 2
# Define the trainig dataset
ttd_train_filename = list(df_train["file_name"])
ttd_train_mo_ta = list(df_train["normalized_transcription"])


train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


"""
## Visualize the data
Let's visualize an example in our dataset, including the
audio clip, the spectrogram and the corresponding label.
"""

fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = batch[1][0]
    # Spectrogram
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")
    # Wav
    file = tf.io.read_file(wavs_path + list(df_train["file_name"])[0] + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = audio.numpy()
    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose(audio), rate=16000))
plt.show()

"""
## Model
We first define the CTC Loss function.
"""

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


"""
We now define our model. We will define a model similar to
[DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html).
"""


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)

"""
## Training and Evaluating
"""

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
      
        L = len(predictions)
        for i in range(L):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

"""
Let's start the training process.
"""

# Define the number of epochs.
epochs = 50
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')'''
    st.code(code, language='python')

def Download_Source():
    url = "https://drive.google.com/drive/u/0/folders/1dhdiyTurY2q2W4c6PGJlKHKKRUyYrWRX"
    st.info("Nhấp Download để tải: [Download](%s)" % url)

def Colab():
    url = "https://www.youtube.com/watch?v=TEHnfsqpUPc&t=1260s"
    st.info("Video hướng dẫn sử dụng Colab: [Link Video](%s)" % url)

def Welcome():
    st.success("HI ! WELCOME TO MY WEBSITE")


def Tab_Show_Code():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["[Code: Ghi âm]", "[Code: Phát âm thanh]", "[Code: Nhận diện]", "[Code: Training]", "[Source code]", "[Video Colab]"])
    with tab1:
        Show_Code_Ghi_Am()

    with tab2:
        Show_Code_Phat_Am_Thanh()

    with tab3:
        Show_Code_Nhan_Dien()

    with tab4:
        Show_Code_Training()

    with tab5:
        Download_Source()

    with tab6:
        Colab()

page_names_to_funcs = {
    "WELCOME": Welcome,
    "SHOW CODE":Tab_Show_Code,
}

demo_name = st.sidebar.selectbox("CHỌN SHOW CODE ĐỂ XEM !", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


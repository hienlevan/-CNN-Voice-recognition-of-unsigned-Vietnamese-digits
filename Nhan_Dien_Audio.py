def Nhan_Dien():
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
    '''
    for i in np.random.randint(0, len(predictions), 2):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)
    '''
    L = len(predictions)
    st.info('Bắt đầu !')
    for i in range(L):
        st.balloons()
        st.success(f"Prediction: {predictions[i]}", icon="ℹ️")
        #st.info("-" * 100)
    st.success('Xong !', icon="ℹ️")
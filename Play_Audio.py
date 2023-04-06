

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
        st.info("Record something to play")
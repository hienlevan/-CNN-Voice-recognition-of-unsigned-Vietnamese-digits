from tkinter import *
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


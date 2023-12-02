import pyaudio
import numpy as np
import wave

class Microphone:
    def __init__(self, sample_rate, buffer_size, source_name):
        self.format = pyaudio.paFloat32
        self.sample_rate = sample_rate
        self.pyaud = pyaudio.PyAudio()
        self.buffer_size = buffer_size
        self.chunk_size = buffer_size
        self.source_name = source_name

        device_index = -1
        info = self.pyaud.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.pyaud.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.pyaud.get_device_info_by_host_api_device_index(0, i).get('name'))
                if source_name in self.pyaud.get_device_info_by_host_api_device_index(0, i).get('name'):
                    device_index = i
                    break

        if device_index == -1:
            print("ERROR: Couldn't find audio device " + source_name)
            exit(-1)

        self.stream = self.pyaud.open(format=self.format,
                                  channels=1,
                                  rate=sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self.get_callback(),
                                  input_device_index=device_index)
        self.chunks_per_buffer = int(buffer_size/self.chunk_size)
        self.count = 0
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.ready_buffer = np.fromstring(in_data, dtype=np.float32)
            self.buffer_ready = True
            return (in_data, pyaudio.paContinue)
        return callback

    def start_stream(self):
        self.stream.start_stream()

    def stop_stream(self):
        self.buffer_ready = False
        self.stream.stop_stream()

    def get_audio(self):
        while 1:
            if self.buffer_ready:
                break
        self.buffer = self.ready_buffer[int(len(self.ready_buffer)/4):]
#        np.save("./pickled/adversary2_audio_" + str(self.count) + ".npy", self.buffer)

        np.savetxt(self.source_name + ".csv", self.buffer, delimiter=",")

        self.count += 1
        self.buffer_ready = False
        return self.buffer

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


time = np.linspace(start=0, stop=0.5, num=2000)
mu, sigma = 0.25, 0.01
sinewave = np.sin(time * 250 * np.pi)
gaussian = (1 / (np.sqrt(2 * np.pi * np.square(sigma))) *
            np.exp(-(np.square(time - mu) /np.square(2 * sigma))))
ripple = gaussian * sinewave

np.random.seed(0)
trace = ripple + gaussian*6
# repeat the trace 4 times
# add noise
trace = np.tile(trace, 4)
trace += np.random.normal(0, 3, size=trace.shape)
# add time to the trace
time = np.arange(start=0, step=time[1], stop=len(trace)*time[1])
plt.plot(time, trace)
plt.show()


class Event:
    def __init__(self, time, trace, threshold=100):
        from scipy.signal import find_peaks
        self.time = time
        self.trace = trace
        self.event = None
        self.sampling_rate = 1 / (time[1] - time[0])
        print('Sampling rate: ', self.sampling_rate)
        event_index = find_peaks(self.trace, height=threshold, distance=100)[0]
        print('Event index: ', event_index)
        if len(event_index) == 0:
            raise ValueError('No events detected')
        self.event_time = time[event_index]
        # make an array with zeros
        self.event_trace = np.zeros((len(event_index), 900))
        for i, index in enumerate(event_index):
            left_index = index - 450
            right_index = index + 450    
            self.event_trace[i, :] += trace[left_index:right_index]
        self.filter_ripple()

    def plot_events(self, show=True):
        import matplotlib.pyplot as plt
        from numpy import arange
        plt.plot(arange(start=0, stop=self.event_trace.shape[1]/self.sampling_rate, step=1/self.sampling_rate), self.event_trace.T)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Events')
        if show:
            plt.show()

    def filter_ripple(self):
        # filter the trace with lowpass filter
        from scipy import signal
        nyquist = self.sampling_rate / 2
        b, a = signal.butter(N=5, Wn=[100/nyquist, 250/nyquist], btype='band')
        self.ripple = np.zeros_like(self.event_trace)
        for i in range(self.event_trace.shape[0]):
            self.ripple[i, :] = signal.filtfilt(b, a, self.event_trace[i, :])
    
    def plot_ripple(self, show=True):
        import matplotlib.pyplot as plt
        from numpy import arange
        plt.plot(arange(start=0, stop=self.event_trace.shape[1]/self.sampling_rate, step=1/self.sampling_rate), self.ripple.T)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Ripple')
        if show:
            plt.show()

detected_events = Event(time, trace, threshold=100)
detected_events.plot_events(show=False)
detected_events.plot_ripple()
detected_events.ripple

event_list = []
for i in range(len(detected_events.event_time)):
    event_list.append(Event(time, trace, threshold=100))

event_list[0].trace
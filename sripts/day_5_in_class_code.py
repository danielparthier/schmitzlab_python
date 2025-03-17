import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit



# import data from charact_data.json as dictionary
import json

with open('data/charact_data.json') as f:
    data = json.load(f)
# print the keys
data = {key: np.array(data[key]) for key in data.keys()}
plt.plot(data['D1'][10])
plt.show()

for i in data['D1']:
    print(i.shape)

time_s = np.linspace(0, data['D1'].shape[1]/20_000, data['D1'].shape[1])

peak_location, peak_properties = signal.find_peaks(data['D1'][10], height=0, width=[10, 50], distance=10)

plt.plot(time_s, data['D1'][10])
plt.plot(time_s[peak_location], peak_properties['peak_heights'], '.')
plt.show()

def find_peaks(data, height=0, width=[10, 50], distance=10):
    peaks = []
    for i in range(data.shape[0]):
        peak_location, peak_properties = signal.find_peaks(data[i], height=height, width=width, distance=distance)
        peaks.append(peak_location)
    return peaks

# find the peaks in the data
peaks = find_peaks(data['D1'], height=0, width=[10, 50], distance=10)
peak_count = [len(peak) for peak in peaks]
current_inj = [-300, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300]

for i in range(len(peaks)):
    plt.plot(data['D1'][i])
    plt.plot(peaks[i], data['D1'][i][peaks[i]], '.')
plt.show()

plt.plot(current_inj, peak_count, '.')
plt.xlabel('Current Injection (pA)')
plt.ylabel('Number of Spikes')
plt.title('Number of Spikes vs Current Injection')
plt.show()


# Example 2 for sensor data
# load the data from csv
sensor_trace = np.loadtxt('data/calcium/condition_2.csv', delimiter=',', skiprows=0) # use this one for everything
time = np.linspace(0, len(sensor_trace)/8, len(sensor_trace))

# plot the data
plt.plot(time, sensor_trace, label='raw trace')
plt.show()


plt.plot(time, sensor_trace[:,0], label='raw trace')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# fit an exponential to the data
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# make a function which fits the exponential and subtracts it from the data
def bleach_correction(traces, plot=False):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    corrected_traces = np.zeros_like(traces)
    xdata = np.arange(0, len(traces)/8, 1/8)
    for sweep in range(traces.shape[1]):
        ydata = traces[:,sweep].flatten()
        xdata = np.arange(0, len(ydata))
        popt, pcov = curve_fit(func, xdata[60:], ydata[60:],
                               p0=(ydata.min()-(ydata.max()-ydata.min()),
                                   .001, ydata.max()-ydata.min()))
        corrected_traces[:,sweep] = ydata - func(xdata, *popt)
        if plot:
            plt.plot(xdata, ydata, 'b-', label='data')
            plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    if plot:
        plt.show()
    return corrected_traces

corrected_traces = bleach_correction(sensor_trace, plot=True)
corrected_traces.shape

# plot the corrected traces
plt.plot(time, corrected_traces, label='corrected traces')
plt.show()

# fd/f = (ft-f0)/f0
# f0 is baseline fluorescence, can usually be taken as the mean of the first frames.
# here we take the mean of a short time during the absence of stimulation

def df_over_f(corrected_traces, start_index, end_index):
    f0 = corrected_traces[start_index:end_index,:].mean(axis=0)
    df_over_f = (corrected_traces-f0)/np.abs(f0)
    return df_over_f

df_f = df_over_f(corrected_traces, 120, 140)


plt.plot(time, df_f, label='df/f')
plt.show()


# make a step by step function to find the peaks
peaks_single_sweep = signal.find_peaks(df_f[:,0], height=1, width=10)

trace_peaks = [signal.find_peaks(df_f[:,sweep], height=1, width=10)[0] for sweep in range(df_f.shape[1])]


# make a template for GCaMP6 with rise time of 6.9ms, time to peak of 18.6ms, and half decay time of 149ms
# rise time = 6.9ms
# time to peak = 18.6ms
# half decay time = 149ms
# decay time = 149ms


def gcamp_template(x, rise_time, tau_decay):
    return  -np.exp(-x/rise_time) + np.exp(-x/tau_decay)


plt.plot(np.arange(0, 1,0.001), gcamp_template(np.arange(0, 1,0.001), .05,0.14))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('GCaMP Template')
plt.show()

# use the gcamp_template for deconvolution
template_8hz = gcamp_template(np.arange(1/8, 2,1/8), .05,0.14)

trace_recorded = df_f[:,0]
recovered, remainder = signal.deconvolve(df_f[:,0].flatten(), template_8hz)

plt.plot(recovered, label='recovered')
plt.show()

plt.plot(df_f[:,0], label='corrected traces')
plt.plot(trace_peaks[0], df_f[trace_peaks[0],0], ".", label='peaks')
plt.show()

# make a subplot of the original trace, the corrected trace, and the deconvolved trace
plt.subplots(3,1, sharex=True)
plt.subplot(3,1,1)
plt.plot(sensor_trace[:,0], label='raw traces')
plt.legend(loc='center right')
plt.subplot(3,1,2)
plt.plot(df_f[:,0], label='df/f traces')
plt.legend(loc='center right')
plt.subplot(3,1,3)
plt.plot(recovered, label='deconvolved trace')
plt.legend(loc='center right')
plt.show()



# filter signals with scipy.signal
# create noisy signal
time = np.linspace(start=0, stop=0.5, num=2000)
mu, sigma = 0.25, 0.01
sinewave = np.sin(time * 250 * np.pi)
gaussian = (1 / (np.sqrt(2 * np.pi * np.square(sigma))) *
            np.exp(-(np.square(time - mu) /np.square(2 * sigma))))
ripple = gaussian * sinewave
np.random.seed(0)
trace = ripple + gaussian*6 + np.random.normal(0, 3, size=time.shape)
plt.plot(time, trace)
plt.show()

# filter the data
# bandpass filter
sampling_rate = 1/time[1]
nyquist = sampling_rate / 2

from scipy.signal import butter, filtfilt

b, a = signal.butter(5, [100/nyquist, 250/nyquist], btype='band')
ripple_filtered_trace = signal.filtfilt(b, a, trace)

# plot filter response
w, h = signal.freqz(b, a, fs=sampling_rate, worN=2000)
plt.plot(w, abs(h))
plt.title('Filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.axvline(100, color='r', linestyle='--')
plt.axvline(250, color='r', linestyle='--')
plt.show()



b, a = signal.butter(5, 80/nyquist, btype='low')
lowpass_filtered_trace = signal.filtfilt(b, a, trace)


plt.plot(time, ripple_filtered_trace)
plt.plot(time, lowpass_filtered_trace)
plt.plot(time, trace)
plt.plot(time, ripple, '--')
plt.show()




# Power spectrum transform of the ripple
f, Pxx_den = signal.welch(trace, fs=1/time[1], nperseg=1024, average='median')
plt.semilogy(f, Pxx_den)
plt.vlines(125, Pxx_den.min(), Pxx_den.max(),color='r', linestyle='--')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density')
plt.show()  



from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def mini_template(x, rise_time, tau_decay, amplitude = 15, centered=False):
    min_x = -np.log(tau_decay / rise_time) / (1 / tau_decay - 1 / rise_time)
    amp_scaling = amplitude/-(-np.exp(-min_x/rise_time) + np.exp(-min_x/tau_decay))
    if centered:
        x_neg = np.arange(-np.max(x),np.min(x), np.abs(x[0]-x[1]))
        if np.max(x) > np.max(x_neg):
            x = np.append(x_neg, x)
        else:
            x = np.append(x, x_neg)
        event = (-np.exp(-x/rise_time) + np.exp(-x/tau_decay)) * amp_scaling
        # first differential of event
        np.log(tau_decay / rise_time) / (1 / tau_decay - 1 / rise_time)
        event[x < 0] = 0
        return event
    else:
        return (-np.exp(-x/rise_time) + np.exp(-x/tau_decay)) * amp_scaling

#def mini_template(x, rise_time, tau_decay):
#    return -(-np.exp(-x/rise_time) + np.exp(-x/tau_decay)) * 21.5

def template_maximum(rise_time, tau_decay):
    # calculate the maximum of the template
    # maximum of the template is at t = rise_time * tau_decay / (tau_decay - rise_time)
    return -np.log(tau_decay / rise_time) / (1 / tau_decay - 1 / rise_time)


x = np.arange(0, 0.2, 0.001)
template_maximum(0.001, 0.012)

plt.plot(x, mini_template(x, 0.001, 0.012))
plt.vlines(x=-np.log(0.012 / 0.001) / (1 / 0.012 - 1 / 0.001),ymin=-20, ymax=0)
plt.show()

for i in range(10, 20):
    plt.plot(x, mini_template(x, 0.001, 0.012, amplitude=i))
#    plt.plot(x, mini_template(x, i/1000, 0.012))
plt.show()

sampling_freq = 2e4
x = np.arange(0, 0.2, 1/sampling_freq)
#x = np.linspace(0, len(binary_vesicle)*1/2e4, len(binary_vesicle))
np.random.seed(0)
vesicle_timing = np.cumsum(np.random.gamma(1, 0.1, 100))
binary_vesicle = np.zeros(int(2e5))
vesicle_timing = np.round(vesicle_timing * sampling_freq).astype(int)
binary_vesicle[vesicle_timing] = 1
binary_vesicle *= np.random.normal(1, 0.1, len(binary_vesicle))

plt.plot(binary_vesicle)
plt.title("Vesicle timing and content")
plt.ylabel("Vesicle content (normalised)")
plt.show()

plt.plot(mini_template(x, 0.001, 0.006))
plt.plot(mini_template(x, 0.001, 0.012))
plt.plot(mini_template(x, 0.004, 0.018))
plt.plot(mini_template(x, 0.008, 0.018))
plt.show()

plt.plot(x, mini_template(x, 0.001, 0.012))
plt.title("Template for rise_time=0.001 and tau_decay=0.012")
plt.ylabel("Template amplitude")
plt.show()

trace = mini_template(x, 0.001, 0.012, amplitude=15, centered=True)
trace = signal.convolve(in1=binary_vesicle, in2=mini_template(x, 0.001, 0.012, amplitude=15, centered=True), mode='same') + \
np.random.normal(0, 6, len(binary_vesicle)) + \
np.sin(np.linspace(0, 3, len(binary_vesicle))) * 0.1
plt.plot(trace)
plt.plot(binary_vesicle)
plt.show()

# filter the trace with lowpass filter
b, a = signal.butter(N=2, Wn=[100], btype='low', fs=sampling_freq)
filtered_trace = signal.filtfilt(b, a, trace)
plt.plot(filtered_trace)
#plt.plot(trace)
plt.show()

filtered_trace = trace

# deconvolution
# use deconvolve from scipy
# notice that the function takes time to run and is not very fast
deconvolved_trace, remainder = signal.deconvolve(filtered_trace, mini_template(x, 0.001, 0.012)[1:])
# normalise the deconvolved trace
deconvolved_trace *= len(filtered_trace)/(len(x)-1)


# lets make our own deconvolution function using Fourier transform
def deconvolve_trace(trace, template):
    # deconvolve the trace with the template
    N = np.int64(2**(np.ceil(np.log2(len(template) + len(trace)))))
    from scipy import fftpack
    # construct trace which is padded with zeros and the length of N
    trace_padded = np.zeros(N)
    trace_padded[int((N-len(trace))/2):int(len(trace)+(N-len(trace))/2)] = trace
    return np.real(fftpack.ifft(fftpack.fft(x=trace_padded, n=N) / fftpack.fft(x=template, n=N))[int((N-len(trace))/2):int(len(trace)+(N-len(trace))/2)])

def filter_trace(trace, b, a, pad_len=5000, sampling_freq=2e4):
    b, a = signal.butter(N=2, Wn=[100], btype='low', fs=sampling_freq)
    # filter the trace with the lowpass filter
    padded_signal = np.zeros(len(trace) + pad_len * 2)
    padded_signal[pad_len:(len(trace)+pad_len)] = trace
    padded_signal[0:pad_len] = 0
    padded_signal[-pad_len:] = 0
    return signal.filtfilt(b, a, padded_signal)[pad_len:(len(trace)+pad_len)]

# Notice the faster FFT (Fast Fourier Transform) deconvolution. 
# In the function we use N as a power of 2 to speed up the FFT. The length of the trace is padded with zeros
# and the template is applied in the frequency domain. This means we divide the FFT of the trace with the
# FFT of the template using the same length. The resulting FFT is then transformed back to the time domain.
deconvolved_trace_fft = deconvolve_trace(trace, mini_template(x,  0.001, 0.012))


# We can now adjust the template to see how it affects the deconvolution
vs_count_parameter_change = []
for i in range(1, 30):
    deconvolved_trace_fft = deconvolve_trace(trace, mini_template(x, 0.001*(i/10), 0.01+i/1000))
    print(i)
    deconvolved_trace_fft = filter_trace(deconvolved_trace_fft, b, a, pad_len=100000)
    vesicle_detection = signal.find_peaks(deconvolved_trace_fft, height=np.std(deconvolved_trace_fft[0:285000])*3)
    vesicle_count = np.int64(vesicle_detection[1]['peak_heights'] // np.min(vesicle_detection[1]['peak_heights']))
    vesicle_trace = np.zeros_like(trace)
    vesicle_trace[vesicle_detection[0]] = vesicle_count
    vs_count_parameter_change.append(len(vesicle_detection[0]))
    plt.plot(x, mini_template(x, 0.001*(i/10), 0.01+i/1000))

plt.show()

plt.plot(vs_count_parameter_change)
plt.title("Vesicle count")
plt.xlabel("Template parameter")
plt.ylabel("Vesicle count")
plt.show()

vs_count_amplitude_change = []
for i in range(5, 100):
    deconvolved_trace_fft = deconvolve_trace(trace, mini_template(x, 0.001, 0.012, amplitude=i))
    print(i)
    deconvolved_trace_fft = filter_trace(deconvolved_trace_fft, b, a, pad_len=100000)
    vesicle_detection = signal.find_peaks(deconvolved_trace_fft, height=np.std(deconvolved_trace_fft[0:285000])*3)
    vesicle_count = np.int64(vesicle_detection[1]['peak_heights'] // np.min(vesicle_detection[1]['peak_heights']))
    vesicle_trace = np.zeros_like(trace)
    vesicle_trace[vesicle_detection[0]] = vesicle_count
    vs_count_amplitude_change.append(len(vesicle_detection[0]))
    plt.plot(x, mini_template(x, 0.001, 0.012, amplitude=i))

plt.show()

plt.plot(vs_count_amplitude_change)
plt.title("Vesicle count")
plt.xlabel("Template parameter")
plt.ylabel("Vesicle count")
plt.show()

# As you can see the count does not change when changing the amplitude of the template.
# This is because of our dynamic thresholding in the peak detection. We use the standard
# deviation for every single trace.

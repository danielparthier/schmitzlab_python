import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit


time = np.linspace(start=0, stop=0.5, num=2000)
mu, sigma = 0.25, 0.01
sinewave = np.sin(time * 250 * np.pi)
gaussian = (1 / (np.sqrt(2 * np.pi * np.square(sigma))) *
            np.exp(-(np.square(time - mu) /np.square(2 * sigma))))
ripple = gaussian * sinewave
plt.plot(time, ripple)
plt.show()

# wavelet transform of the ripple
f, Pxx_den = signal.welch(ripple, fs=1/time[1], nperseg=1024, average='median')
plt.semilogy(f, Pxx_den)
plt.vlines(125, Pxx_den.min(), Pxx_den.max(),color='r', linestyle='--')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of the Ripple')
plt.show()  


# load the data from csv
condition_1 = np.loadtxt('data/calcium/condition_1.csv', delimiter=',', skiprows=0) # skip these
condition_2 = np.loadtxt('data/calcium/condition_2.csv', delimiter=',', skiprows=0) # use this one for everything
condition_3 = np.loadtxt('data/calcium/condition_3.csv', delimiter=',', skiprows=0) # still a bit weird
time_1 = np.linspace(0, len(condition_1)/8, len(condition_1))
time_2 = np.linspace(0, len(condition_2)/8, len(condition_2))

# plot the data
plt.plot(time_1,condition_1, label='condition 1')
plt.plot(time_2, condition_2, label='condition 2')
plt.plot(condition_3, label='condition 3')
plt.legend()
plt.show()


plt.plot(time_2, condition_2[:,0], label='condition 2')
np.median(condition_2[:,0])
# add horizontal line at the median
plt.axhline(y=np.median(condition_2[:,0]), color='r', linestyle='--', label='median')
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
        #    plt.plot(xdata, ydata-func(xdata, *popt), 'g-', label='data_corrected')
    if plot:
       # plt.legend()
        plt.show()
    return corrected_traces

corrected_traces = bleach_correction(condition_2, plot=True)
corrected_traces.shape

# just for testing the apply_along_axis function
def inner_loop(ydata, plot=False):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    xdata = np.arange(0, len(ydata))
    popt, pcov = curve_fit(func, xdata[60:], ydata[60:],
                           p0=(ydata.min()-(ydata.max()-ydata.min()),
                               .001, ydata.max()-ydata.min()))
    if plot:
        plt.plot(xdata, ydata, 'b-', label='data')
        plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    return ydata - func(xdata, *popt)

# apply the function to each trace
# time this function using timeit
# the apply function is slightly slower. not enough iterations to be faster compared to for loop
import timeit
def time_function(input):
    return np.apply_along_axis(inner_loop, 0, input)

# time the function
timeit.timeit('time_function(condition_2)', globals=globals(), number=1000)
timeit.timeit('bleach_correction(condition_2)', globals=globals(), number=1000)




# apply the bleach correction to the data
corrected_traces = bleach_correction(condition_2)

# plot the corrected traces
plt.plot(time_2, corrected_traces, label='corrected traces')
plt.show()


# fd/f = (ft-f0)/f0
# f0 is baseline fluorescence, can usually be taken as the mean of the first frames.
# here we take the mean of a short time during the absence of stimulation

def df_over_f(corrected_traces, start_index, end_index):
    f0 = corrected_traces[start_index:end_index,:].mean(axis=0)
    df_over_f = (corrected_traces-f0)/np.abs(f0)
    return df_over_f

df_f = df_over_f(corrected_traces, 120, 140)


plt.plot(time_2, df_f, label='df/f')
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
    x_max = (rise_time * tau_decay) / (tau_decay - rise_time)
    y_max = -np.exp(-x_max/rise_time) + np.exp(-x_max/tau_decay)
    return  (-np.exp(-x/rise_time) + np.exp(-x/tau_decay))/ y_max




plt.plot(np.arange(0, 1,0.001), gcamp_template(np.arange(0, 1,0.001), .05,0.14))


plt.plot(np.arange(0, 1,0.001), gcamp_template(np.arange(0, 1,0.001), .05,0.14))
plt.vlines((.05*0.14)/(0.14-0.05), 0, 1, color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('GCaMP Template')
plt.show()

# use the gcamp_template for deconvolution
df_f[:,0]
template_8hz = gcamp_template(np.arange(1/8, 2,0.1/8), .05,0.14)

trace_recorded = df_f[:,0]
recovered, remainder = signal.deconvolve(df_f[:,0].flatten(), template_8hz)
plt.plot(recovered, label='recovered')
plt.show()

plt.plot(df_f[:,0], label='corrected traces')
plt.plot(trace_peaks[0], df_f[trace_peaks[0],0], ".", label='peaks')
trace_peaks

plt.show()

# make a subplot of the original trace, the corrected trace, and the deconvolved trace
plt.subplot(3,1,1)
plt.plot(condition_2[:,0], label='raw traces')
plt.legend(loc='center right')
plt.subplot(3,1,2)
plt.plot(df_f[:,0], label='df/f traces')
plt.legend(loc='center right')
plt.subplot(3,1,3)
plt.plot(recovered, label='deconvolved trace')
plt.legend(loc='center right')
plt.show()

trace_peaks[0]


def gcamp_template(x, rise_time, a_rise, a_decay, tau_decay):
    return a_rise * np.exp(rise_time * x) + a_decay * np.exp(-tau_decay * x)


import numpy as np
import pywt
from neo import AnalogSignal, SpikeTrain
from quantities import second
from scipy import signal
from tqdm import tqdm


def wavelet_denoise(in_arr, threshold=0.4, level=None, wlet="sym7", noise_arr=None):
    """
    perform wavelet denoising on an input signal
    """
    if noise_arr is None:
        noise_arr = in_arr
    threshold_vals = calculate_wavelet_denoise_thresholds(
        noise_arr, threshold=threshold, wlet=wlet, level=level
    )
    recon = wavelet_denoise_prespecified(in_arr, threshold_vals, wlet=wlet)
    return recon


def calculate_wavelet_denoise_thresholds(
    in_arr, threshold=1, level=None, wlet="sym7", method="NEW"
):
    """
    calculates the threshold values for the wavelet denoise filter
    method=NEW:
        T=σ * sqrt(2ln(N))
        where σ is std of gaus noise for each level.

    """
    if type(wlet) is str:
        wlet = pywt.Wavelet(wlet)

    coeffs = pywt.wavedec(in_arr, wlet, level=level)
    factor = threshold * np.sqrt(2 * np.log(in_arr.shape[-1]))  # np.log = ln
    coef_thresholds = []
    for i in coeffs[1:]:  # coeffs[0] is approximations (cA)
        if method == "NEW":
            # https://ieeexplore.ieee.org/document/1179130 - although not quite?
            sigma = np.median(np.abs(i)) / 0.6745  # 0.6745 = 75th percentile
            t = sigma * factor
        else:
            t = np.max(
                threshold * max(i)
            )  # This is incorrect... but i cant figure out the original logic.

        coef_thresholds.append(t)

    return coef_thresholds


def wavelet_denoise_prespecified(in_arr, threshold_arr=None, wlet="sym7"):
    """
    A generic class to perform wavelet denoising given an array of thresholds.
    """
    if type(wlet) is str:
        wlet = pywt.Wavelet(wlet)

    coeffs = pywt.wavedec(in_arr, wlet, level=len(threshold_arr))
    coeffs2 = []
    for threshold, c in zip(
        threshold_arr, coeffs[1:]
    ):  # coeffs[0] is approximations (cA)
        coeffs2.append(pywt.threshold(c, threshold, mode="hard"))
    recon = pywt.waverec([coeffs[0], *coeffs2], wlet)
    return recon


def create_erp(signal_chan, idxs, offset=-1000, length=30000):
    arr = np.zeros((len(idxs), length))
    for i, x in enumerate(idxs):
        arr[i].flat[:] = signal_chan[int(x + offset) : int(x + offset + length)]
    return arr



# def create_erp_signals(signal, event, offset, length, channel=0):
#     start = int(np.array(signal.sampling_rate.base) // ((1 / offset)))
#     end = int(np.array(signal.sampling_rate.base) // ((1 / length)))
#     erp = create_erp(
#         signal.as_array()[:, channel],
#         (event.as_array() - signal.t_start.base)
#         * np.array(signal.sampling_rate, dtype=int),
#         start,
#         end,
#     )
#     return erp
import quantities as pq
def create_erp_signals(signal_chan, idxs, offset=-1000*pq.ms, length=30000*pq.ms):
    arr = np.zeros((len(idxs), int(np.ceil(length.rescale(pq.second).magnitude*signal_chan.sampling_rate.magnitude))))
    for i, x in enumerate(idxs):
        arr[i].flat[:] = signal_chan.time_slice(x + offset, x + offset + length).as_array()
    return arr

# a neo class that creates an api like analogsignal for multiple short signals with different start/stop times. they must not overlap.
from neo.core.basesignal import BaseSignal
from typing import List
from neo.core import Group, Segment
class MultiAnalogSignal:
    
    def __init__(self, signals:List[AnalogSignal]):
        self.signals = signals
        self.t_start = min([x.t_start for x in signals])
        self.t_stop = max([x.t_stop for x in signals])
        self.sampling_rate = signals[0].sampling_rate
        self.units = signals[0].units
        self.shape = (len(signals), signals[0].shape[0])
        self.array_annotations = {}
        self.annotations = {}
        self.name = signals[0].name

        # ensure all the signals have the same sample rate
        for x in signals:
            assert x.sampling_rate == self.sampling_rate

        # ensure that the signals dont overlap
        for i, x in enumerate(signals[:-1]):
            assert x.t_stop <= signals[i + 1].t_start

        self.t_starts = [x.t_start for x in signals]
    
    def time_slice(self, start, stop):
        # find the signals that overlap with the start/stop
        # then slice them and concatenate them
        sig = np.searchsorted(self.t_starts, start, side="left")  -1

        x = self.signals[sig]
        try:
            start_idx = x.time_index(start)
            stop_idx = x.time_index(stop)
        except ValueError:
            # Handle the case where start or stop is outside the range of x
            raise ValueError(f"No signal found for the time slice {start}-{stop}")


        return x[start_idx:stop_idx]
            

    @property
    def times(self):
        return np.concatenate([x.times for x in self.signals])

    

from neo import Event
import quantities as pq
import warnings
# function to create multianalogsignal like erp
def create_neo_multianalogsignal(signal_chan:AnalogSignal, idxs:Event, offset=-1, length=2):

    # if the offset or length is not a pyquantity assume it is seconds otherwise rescale it to seconds
    if not hasattr(offset, "rescale"):
        offset = offset * pq.second

    if not hasattr(length, "rescale"):
        length = length * pq.second
    
    offset = offset.rescale("s")
    length = length.rescale("s")

    # create a list of signals
    signals = []
    # create the (start, stop) for segments, where start= idx+offset and end=idx+offset+length ensuring there are no overlaps of idx-offset and idx-offset+length. 
    regions = []
    start = idxs[0] + offset
    stop = idxs[0] + offset + length
    for i in range(1, len(idxs)):
        next_start = idxs[i] + offset
        if next_start < stop:
            stop = next_start + offset + length
        else:
            regions.append((start, stop))
            start = next_start
            stop = next_start + offset + length
    regions.append((start, stop))
    
    for start,stop in regions:
        if start < signal_chan.t_start or stop > signal_chan.t_stop:
            warnings.warn("start and/or stop are outside the range of signal_chan")
            stop = min(signal_chan.t_stop, stop)
            start = max(signal_chan.t_start, start)
        if start==stop:
            warnings.warn("start and stop are the same, skipping")
            continue

        signals.append(signal_chan.time_slice(start, stop))

  

    return MultiAnalogSignal(signals)


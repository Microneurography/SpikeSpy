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


def create_erp_signals(signal, event, offset, length, channel=0):
    start = int(np.array(signal.sampling_rate.base) // ((1 / offset)))
    end = int(np.array(signal.sampling_rate.base) // ((1 / length)))
    erp = create_erp(
        signal.as_array()[:, channel],
        (event.as_array() - signal.t_start.base)
        * np.array(signal.sampling_rate, dtype=int),
        start,
        end,
    )
    return erp

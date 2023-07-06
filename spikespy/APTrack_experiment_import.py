# Code to load data generated using APTrack into neo
# ensure works with 04/04 data

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import neo
import numpy as np
import quantities as pq
from neo.core import AnalogSignal, Block, Event, Group, Segment
from scipy import signal


def apply_bandpass(data: AnalogSignal, lowHz=500, highHz=7000, sampleRate=None):
    """
    apply bandpass filter to the data
    """
    out = np.array(data).copy()
    sampleRate = sampleRate or data.sampling_rate
    a, b = signal.butter(
        2, [lowHz / sampleRate / 2, highHz / sampleRate / 2], "bandpass"
    )
    ch1_filt = signal.filtfilt(a, b, data.T).T
    out[:] = ch1_filt
    return out * data.units


class TypeID(Enum):
    ANALOG = "AS"  # analogsignal channel
    TTL = "TTL"  # electricalstimulus - convert to timestamps
    EVENTS = "EVT"  # events files to convert into timestamps


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace("\n", "").replace("header.", "")
    for i, item in enumerate(h.split(";")):
        if "=" in item:
            header[item.split(" = ")[0]] = item.split(" = ")[1]
    return header


def readContinous(filename):
    NUM_HEADER_BYTES = 1024
    openephysdtype = np.dtype(
        [
            ("timestamp", "<i8"),
            ("N", np.uint16),
            ("recording", np.uint16),
            ("data", ">i2", 1024),
            ("rec", np.uint8, 10),
        ]
    )
    header = readHeader(open(filename, "rb"))
    mmap = np.memmap(filename, mode="r", dtype=openephysdtype, offset=NUM_HEADER_BYTES)
    return (header, mmap)


@dataclass
class APTrackRecording:
    filename: str
    probe_type: TypeID
    name: str
    details: str = ""
    summarize = None


def as_neo(mng_files: List[APTrackRecording], aptrack_events: str = None, record_no=1):
    """
    Takes the openephys files and creates a single neo object containing the experiment data

    mng_files: defines the categories of mng files
    mng_events: the filename of the messages.events file to use (should identify spikes)
    """

    seg = Segment()
    for f in mng_files:

        header, ch_mmap = readContinous(f.filename)
        sampling_rate = float(header["sampleRate"]) * pq.Hz
        ch_units_probe = pq.UnitQuantity(
            "kmicroVolts", pq.uV / float(header["bitVolts"]), symbol="kuV"
        )
        # header['']
        if record_no is not None and record_no >= 1:
            ch_mmap2 = ch_mmap[ch_mmap["recording"] == record_no - 1]
        else:
            ch_mmap2 = ch_mmap

        t_start = (ch_mmap2["timestamp"][0] / float(header["sampleRate"])) * pq.s
        if f.probe_type == TypeID.ANALOG:
            asig = AnalogSignal(
                ch_mmap2["data"].flat,
                sampling_rate=sampling_rate,
                units=ch_units_probe,
                type_id=f.probe_type.value,
                name=f.name,
                description=f.details,
                t_start=t_start,
            )
            seg.analogsignals.append(asig)
            if f.name.lower().startswith("rd"):
                seg.analogsignals.append(
                    AnalogSignal(
                        apply_bandpass(asig, 500, 7000),
                        sampling_rate=sampling_rate,
                        type_id=f.probe_type.value,
                        name=f.name + ".bp",
                        description=f.details + "\n bandpass 100-7000Hz",
                        t_start=t_start,
                    )
                )
        elif f.probe_type == TypeID.TTL:
            m = np.mean(ch_mmap2["data"])
            # find events from continuous file
            idxs = find_square_pulse_numpy(
                ch_mmap2["data"].flat,
                int(int(header["sampleRate"]) * 0.0004),
                (2 * np.std(ch_mmap2["data"] - m)) + m,
            )  # 2sd from mean

            idxs_rising = idxs[
                0
            ]  # [idxs[1]==1].astype(int) # only take the rising edge
            seg.events.append(
                Event(
                    (idxs_rising / sampling_rate).rescale("s") + t_start,
                    type_id=TypeID.TTL.value,
                    name=f.name,
                    array_annotations={
                        "duration": (
                            np.array(idxs[1] - idxs[0] - 1) / sampling_rate
                        ).rescale("s"),
                        "maximum": (idxs[2] * ch_units_probe).rescale("mV"),
                    },
                    description=f.details,
                )
            )
        elif f.probe_type == TypeID.EVENTS:
            raise Exception("WIP")
        else:
            raise Exception(f"Unsupported probe type '{f.probe_type}'")

    if aptrack_events is not None:
        spike_events, stimChange_events, protocol_events = parse_APTrackEvents(
            aptrack_events
        )

        units = []
        try:
            for x in np.unique(spike_events.array_annotations["spikeGroup"]):
                e = spike_events[spike_events.array_annotations["spikeGroup"] == x]
                e.name = f"unit {x}"
                units.append(e)
        except:
            pass
        seg.events += units  # TODO: include the protocol info
        seg.events += [stimChange_events, protocol_events]
    return seg


def parse_APTrackEvents(filename):
    f = open(filename, "r")
    stimulation_volts = []
    spikeInfo = []
    protocolInfo = []
    curProtocol = ""
    curProtocolNo = 0
    curProtocolStep = 0
    sampleRate = 30000  # TODO: this should be found... somewhere? or neo may be able to handle timestamps
    for l in f.readlines():
        l2 = l.split(" ", 1)
        if len(l2) == 1:
            continue
        timestamp = int(l2[0])
        message = l2[1].strip()
        if message.find("Processor: ") >= 0:
            sampleRate = int(re.findall(r"(\d*)Hz", message)[0])
        if message.find("setStimVoltage") >= 0:
            voltage = float(message.split(":")[1])
            stimulation_volts.append([timestamp, {"voltage": voltage}])
            continue
        elif message.find("spikeSampleNumber") >= 0:
            d = json.loads(
                message.replace("'", '"')
            )  # TODO: fix the json encoding in the plugin
            spikeInfo.append([timestamp, d])
        elif message.find("starting stimulus") >= 0:
            curProtocol = message.split("starting stimulus protocol ")[1]
            curProtocolNo += 1
            curProtocolStep = 0

        elif message.find("voltage:") >= 0:  # TODO: convert this into json encoded.
            elements = message.split(";")
            curProtocolStep += 1
            d = {e.split(":")[0].strip(): float(e.split(":")[1]) for e in elements}
            d["protocol"] = curProtocol
            d["protocolIdx"] = curProtocolNo
            d["step"] = curProtocolStep
            protocolInfo.append([timestamp, d])

        else:
            logging.warning(f"did not parse log line: {message}")

    def make_evt(arr):
        array_annotations = {}
        for _, evt in arr:
            for k, v in evt.items():

                array_annotations[k] = array_annotations.get(k, []) + [v]
        for k, v in array_annotations.items():
            array_annotations[k] = np.array(v)

        return Event(  # A bit clunky - for spikeEvents this uses spikeSampleNumber, otherwise use the current timestamp
            np.array(
                [
                    (((x[1].get("spikeSampleNumber", 0)) or x[0]) / sampleRate)
                    for x in arr
                ]
            )
            * pq.s,
            array_annotations=array_annotations,
        )

    evt_spikeInfo = make_evt(
        spikeInfo
    )  # TODO: this needs to split by spikeGroup to work in viewer

    evt_stimulation_volts = make_evt(stimulation_volts)
    evt_stimulation_volts.name = "log.stimVolt"
    evt_protocolInfo = make_evt(protocolInfo)
    evt_protocolInfo.name = "log.protocol"

    return [evt_spikeInfo, evt_stimulation_volts, evt_protocolInfo]


def process_folder(foldername: str, record_no=1, config=None):
    """
    taking a given folder convert to neo using the as_neo function

    default setup:
    CH1 = main
    ADC4 = stimVolt
    ADC5 = TTL
    ADC7 = Temperature
    ADC8 = Button

    messages.events = APTrack spike info.
    config paramter can override these to suit older data.
    """
    all_files = list(Path(foldername).glob(f"*.continuous"))

    def find_channel(chname, foldername=None):
        channo = int(re.sub("[^\d]", "", chname))
        if chname.startswith("ADC"):
            channo += 16  # sometimes there is just the channel number for ADC
        rex = f"_(((CH)?{channo})|({chname}))"
        if record_no > 1:
            rex += f"(_{record_no})?"
        rex += ".continuous"

        matches = [x for x in all_files if len(re.findall(rex, str(x.name))) == 1]
        return matches[0]

    if config is None:
        config = {}
    signals = [
        APTrackRecording(
            find_channel(config.get("rd.0", "CH1")),
            TypeID.ANALOG,
            "rd.0",
            "microneurography probe",
        ),  # main,
        APTrackRecording(
            find_channel(config.get("stimVolt", "ADC4")),
            TypeID.TTL,
            "env.stimVolt",
            "A TTL of the stimulation voltage",
        ),
        APTrackRecording(
            find_channel(config.get("stim", "ADC5")),
            TypeID.TTL,
            "env.stim",
            "A TTL of the stimulation",
        ),
        APTrackRecording(
            find_channel(config.get("stim", "ADC5")),
            TypeID.ANALOG,
            "env.stim",
            "A TTL of the stimulation",
        ),
        APTrackRecording(
            find_channel(config.get("thermode", "ADC7")),
            TypeID.ANALOG,
            "env.thermode",
            "Thermal stimulation temperatures",
        ),
        APTrackRecording(
            find_channel(config.get("button", "ADC8")),
            TypeID.TTL,
            "rec.button",
            "Manual button press, usually to signify a change in protocol or mechanical stimulation",
        ),
    ]

    messages = Path(foldername) / "messages.events"
    neo = as_neo(
        signals, str(messages) if messages.exists() else None, record_no=record_no
    )

    return neo


def process_oe_binary(folder):
    """
    CH1 = main
    ADC4 = stimVolt
    ADC5 = TTL
    ADC7 = Temperature
    ADC8 = Button
    """
    from open_ephys.analysis import Session

    folder = "/Users/xs19785/Library/CloudStorage/OneDrive-UniversityofBristol/Microneurography/Data/2023-01-31_11-47-16/Record Node 103/"

    sess = Session(folder)
    recording = sess.recordings[0]

    pass


# def process_folder2(foldername:str, chnum):

#     signals = [
#         APTrackRecording(
#             Path(foldername)/"101_1.continuous", TypeID.ANALOG, "rd.0", "microneurography probe"
#         ),  # main,
#         APTrackRecording(
#             Path(foldername)/"101_2.continuous",
#             TypeID.TTL,
#             "env.stimVolt",
#             "A TTL of the stimulation voltage",
#         )
#     ]

#     messages = Path(foldername)/"messages.events"
#     neo = as_neo(signals, str(messages) if messages.exists() else None)
#     return neo

from numpy.lib.stride_tricks import sliding_window_view


def find_square_pulse_numpy(arr, width, threshold):

    # rolling_mean = np.mean(sliding_window_view(arr,width),-1)
    # transitions = arr[:-width+1] - rolling_mean

    up = arr > threshold

    transitions_up = np.where(~up[:-1] & up[1:])[0]
    transitions_down = np.where(up[:-1] & ~up[1:])[0]

    transitions_down_idx = 0
    starts = []
    ends = []
    maxes = []
    for start in transitions_up:
        if start < transitions_down[transitions_down_idx]:
            continue  # this should not be possible

        while transitions_down[transitions_down_idx] < start:
            transitions_down_idx += 1

        end = transitions_down[transitions_down_idx]

        if end - start > width:
            maxval = np.max(arr[start:end])
            starts.append(start)
            ends.append(end)
            maxes.append(maxval)

    return np.array(starts, dtype=int), np.array(ends, dtype=int), np.array(maxes)


def find_square_pulse(arr, width, threshold):
    """
    convert analog arr into timepoints.
    arr = a 1d array of the signal
    width = number of points used for a local average to use when identifying step changes
    threshold = the difference between the signal and average to indicate a change

    returns (
            indexes,
            direction (1 for step up, 0 for step down),
            maxima (the maximum value since the previous index
        )
    """
    maxno = 20000
    out = np.zeros(maxno, dtype=np.int32)
    direction = np.zeros(maxno, dtype=np.int8)
    maxima = np.zeros(maxno, dtype=np.float32)
    buffer = np.zeros(width, dtype=arr.dtype)

    out_i = 0
    prev = 0
    max_val = 0
    i = 0
    sum_val = np.sum(arr[0:width])
    for x in range(arr.shape[0]):
        start = x - width
        if start < 0:
            start = 0
        # sum_val = np.sum(buffer)
        sum_val = (
            sum_val + arr[x] - arr[start]
        )  # recalculate average. increase in coding complexity, reduces runtime dramatically
        v = np.abs(arr[x] - (sum_val / buffer.shape[0]))

        if v > threshold and prev < threshold:
            out[out_i] = x
            direction[out_i] = 0
            maxima[out_i] = max_val
            max_val = 0
            out_i += 1

        if v < threshold and prev > threshold:
            out[out_i] = x
            direction[out_i] = 1
            maxima[out_i] = max_val
            max_val = 0
            out_i += 1
        if max_val < v:
            max_val = v
        prev = v
        buffer[i] = v
        i += 1

        if i > width - 1:
            i = 0
    return (out[:out_i], direction[:out_i], maxima[:out_i])


try:
    import numba

    find_square_pulse = numba.jit(find_square_pulse)
except:
    logging.info("optional dependency numba will speed up find_square_pulse method")

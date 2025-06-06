import logging
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from tempfile import tempdir
from turtle import update
from typing import Any, List, Optional, Union

import neo
import numpy as np
import quantities as pq
from neo import Event
from neo.io import NixIO, NixIOFr
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDialogButtonBox,
    QMessageBox,
)
from functools import lru_cache
from neo.core.objectlist import ObjectList

from .APTrack_experiment_import import (
    process_folder as open_aptrack,
    APTrackRecording,
    TypeID,
)
from .NeoOpenEphyisIO import open_ephys_to_neo
from .processing import create_erp_signals


class lru_numpy_memmap:
    def __init__(self):
        self.cache_dir = tempfile.gettempdir()
        self.cache_dict = {}
        self.data_cache = {}

    def clear_cache(self):
        for k, v in self.cache_dict.items():
            assert Path(v).parent == Path(self.cache_dir)
            Path(v).unlink()
        self.cache_dict = {}
        self.data_cache = {}

    def __call__(
        self,
        func,
        keyfunc=None,
    ):
        if keyfunc is None:
            keyfunc = lambda *args, **kwargs: ";".join(
                [str(hash(a)) for a in args] + [f"{k}={hash(v)}" for k, v in kwargs]
            )

        def _(*args, **kwargs):
            key = keyfunc(*args, **kwargs)
            if key in self.cache_dict:
                if key in self.data_cache:
                    return self.data_cache[key]
                try:
                    data = np.load(self.cache_dict[key], mmap_mode="r+")
                    self.data_cache[key] = np.load(self.cache_dict[key], mmap_mode="r+")
                    return data
                except:

                    logging.warn("cache appears to be removed")

            res = func(*args, **kwargs)
            tmpfile_loc = tempfile.mkstemp(dir=self.cache_dir)[1] + ".npy"
            self.cache_dict[key] = tmpfile_loc

            np.save(tmpfile_loc, res)
            del res
            return np.load(self.cache_dict[key], mmap_mode="r+")

        _.cache_clear = self.clear_cache
        return _


@dataclass
class tracked_neuron_unit:
    """
    a UI class for tracked neurons.
    This is to store the current state of neurons tracked

    #TODO: this abstraction is quite annoying. idx_arr in particular makes this hard to use
    """

    @property
    def idx_arr(self):
        logging.warning(
            "Accessing idx_arr is deprecated and will be removed in future versions. use get_latencies instead"
        )
        return self._idx_arr

    @idx_arr.setter
    def idx_arr(self, value):
        logging.warning(
            "Setting idx_arr is deprecated and will be removed in future versions."
        )
        self._idx_arr = value

    fibre_type: str = ""
    notes: str = ""
    idx_arr: List[int] = None
    event: Event = None

    def get_number_of_events(self):
        return len([x for x in self.idx_arr if x is not None])

    def get_window(self):
        # return the min-time & max-time of the events
        idx_arr_filt = [x[0] for x in self.idx_arr if x is not None]
        if len(idx_arr_filt) == 0:
            return None
        return (int(min(idx_arr_filt)), int(max(idx_arr_filt)))

    def get_idx_arr(self, event_signal, signal_chan):
        p = [None for x in range(len(event_signal))]
        for e in x.times:
            i = int(np.searchsorted(event_signal, e).base) - 1
            x = int(((e - event_signal[i]) * signal_chan.sampling_rate).base)
            p[i] = (
                x,
                signal_chan.base[event_signal[i].base * signal_chan.sampling_rate.base],
            )
        return p

    def get_latencies(self, event_signal: pq.UnitTime) -> pq.UnitTime:
        """
        returns the latencies of the events in the spikegroup in ms.
        if the event is not in the spikegroup, it returns np.nan

        note: this will only return one latency per event, if there are multiple events in the same time, it will only return the first one.

        """

        next_spikes = np.searchsorted(self.event.times, event_signal).magnitude
        next_spikes = np.clip(next_spikes, 0, len(self.event.times) - 1)
        if len(self.event.times) == 0:
            return np.ones(len(event_signal)) * np.nan * pq.ms
        spike_times = self.event.times[next_spikes] - (
            np.asarray(event_signal) * event_signal.units
        )

        spike_times = spike_times.rescale(pq.ms)
        spike_times[spike_times < (0 * pq.s)] = np.nan * pq.ms
        return spike_times
        for i, e in enumerate(event_signal):
            next_spike = np.searchsorted(self.event.times, e)
            if next_spike == len(self.event.times):
                continue
            p[i] = (self.event.times[next_spike] - e).rescale(pq.ms)

        # for e in self.event.times:
        #     i = np.searchsorted(event_signal, e) - 1
        #     p[i] = (e - event_signal[i]).rescale(pq.ms)
        return p


class ViewerState(QObject):
    onLoadNewFile = Signal()
    onUnitChange = Signal(int)
    onUnitGroupChange = Signal()
    onStimNoChange = Signal(int)

    def __init__(self, parent=None, segment=None, spike_groups=None):
        super().__init__(parent)

        self.segment: neo.Segment = segment
        self.spike_groups: List[tracked_neuron_unit] = spike_groups or []
        self.stimno = 0
        self.cur_spike_group = 0
        self.trace_count = 3
        self.analog_signal: neo.AnalogSignal = None
        self.sampling_rate = None
        self.event_signal: neo.Event = None
        self.window_size = 0.5 * pq.s
        self.undo_queue = []
        self.MAX_UNDO = 10
        self.title = ""

    def save_undo(self, func):
        self.undo_queue.append(func)
        if len(self.undo_queue) > self.MAX_UNDO:
            self.undo_queue.pop(0)

    def undo(self):
        if len(self.undo_queue) == 0:
            return
        func = self.undo_queue.pop()
        func()

    @property
    def analog_signal_erp(self):
        return self.get_erp()

    @Slot(int)
    def setUnitGroup(self, unitid: int):
        if unitid > len(self.spike_groups):
            return
        self.cur_spike_group = unitid
        self.onUnitGroupChange.emit()

    @Slot(int, int)
    def setUnit(self, latency, stimno=None):
        if stimno is None:
            stimno = self.stimno
        evt = self.spike_groups[self.cur_spike_group].event

        def undo(
            x=self.cur_spike_group,
            evt=self.spike_groups[self.cur_spike_group].event.copy(),
            stimno=self.stimno,
            self=self,
        ):
            evt.sort()
            self.spike_groups[self.cur_spike_group].event = evt
            self.update_idx_arrs()
            self.onUnitChange.emit(stimno)

        self.save_undo(undo)

        if len(evt) > 0:
            to_keep = (evt < self.event_signal[stimno]) | (
                (evt > self.event_signal[stimno + 1])
                if self.stimno < len(self.event_signal) - 1
                else False
            )  # Prevent crash on final index
            # TODO: we should remove by index, and also use purely the timestamps not idx_arr
            self.spike_groups[self.cur_spike_group].event = evt[to_keep]
            evt = evt[to_keep]
        if latency is None:
            self.spike_groups[self.cur_spike_group].idx_arr[stimno] = None

        else:
            self.spike_groups[self.cur_spike_group].idx_arr[stimno] = (
                latency,
                1,
            )  # TODO: fix
            evt2 = evt.merge(
                Event(
                    ((latency / self.analog_signal.sampling_rate).rescale(pq.s))
                    + self.event_signal[[stimno]],
                )
            )
            evt2.name = evt.name
            self.spike_groups[self.cur_spike_group].event = evt2

        self.spike_groups[
            self.cur_spike_group
        ].event.sort()  # TODO: not sure if this breaks the array annotations.

        self.onUnitChange.emit(stimno)

    @Slot(Event)
    def updateUnit(self, event, merge=False):
        def undo(
            x=self.cur_spike_group,
            evt=self.spike_groups[self.cur_spike_group].event,
            stimno=self.stimno,
        ):
            self.spike_groups[self.cur_spike_group].event = evt
            self.update_idx_arrs()
            self.onUnitChange.emit(stimno)

        self.save_undo(undo)

        old_evt = self.spike_groups[self.cur_spike_group].event.rescale(pq.s)
        new_event = event.rescale(pq.s)
        if merge:
            # update the existing events with the new ones, removing ones within 0.5s of the other

            newEvents = []
            new_evt_idxs = self.event_signal.rescale(pq.s).searchsorted(new_event) - 1
            old_evt_idxs = self.event_signal.rescale(pq.s).searchsorted(old_evt) - 1

            for i, t in enumerate(self.event_signal):
                time_gap = 1 * pq.s  # TODO get from state

                for x in [*new_event[new_evt_idxs == i], *old_evt[old_evt_idxs == i]]:
                    if x <= t + time_gap and x >= t:
                        newEvents.append(x)
                        break

            event = Event(np.array(newEvents) * pq.s,**old_evt.annotations, name=old_evt.name)
            #TODO: this does not maintain array_annotations.
        self.spike_groups[self.cur_spike_group].event = event
        del self.spike_groups[self.cur_spike_group].idx_arr
        self.update_idx_arrs()
        self.onUnitGroupChange.emit()

    def stimno_offset_to_event(self, stimno, offset):
        """
        convert arrays of stimno and offset to Event
        """
        t = self.event_signal[stimno] + (offset / self.analog_signal.sampling_rate)
        return Event(t)

    @Slot(int)
    def setStimNo(self, stimno: int):
        self.stimno = max(min(stimno, len(self.event_signal) - 1), 0)
        self.onStimNoChange.emit(self.stimno)

    @Slot(str, str)
    def loadFile(self, fname, type="h5", **kwargs):
        try:
            data, analog_signal, event_signal, spike_groups = load_file(fname, type)
        except:
            import traceback

            traceback.print_exc()
            qmb = QMessageBox(
                QMessageBox.Critical,
                "Sorry :(",
                f'Unable to load "{fname}" using {type}',
            )
            qmb.setModal(True)
            qmb.exec()

            return
        self.title = fname
        self.segment = data
        self.cur_spike_group = 0
        self.set_data(analog_signal, event_signal, spike_groups=spike_groups)

    def removeUnitGroup(self, unit):
        spike_groups = self.spike_groups
        self.spike_groups = [x for i, x in enumerate(spike_groups) if x.event.name != unit.event.name]
        self.update_idx_arrs()
        self.save_undo(
            lambda: self.set_data(
                spike_groups=spike_groups,
            )
        )
        self.onUnitGroupChange.emit()


    @Slot()
    def addUnitGroup(self):
        evt = Event()
        nom = f"unit"
        x = len(self.spike_groups)
        while(f"{nom}_{x}" in [sg.event.name for sg in self.spike_groups]):
            x+= 1
        evt.name =f"{nom}_{x}" 
        evt.annotations["type"] = "unit"
        evt.annotations["unit"] = len(self.spike_groups)
        evt.annotations["notes"] = ""
        evt.annotations["fibre_type"] = ""
        self.spike_groups.append(tracked_neuron_unit(event=evt))
        self.update_idx_arrs()
        self.onUnitGroupChange.emit()

    def update_idx_arrs(self):
        for sg in self.spike_groups:
            # if sg.idx_arr is not None:
            #    continue
            p = [None for x in range(len(self.event_signal))]
            sg.idx_arr = p
            for e in sg.event.rescale("s").times:
                i = (
                    int(
                        np.searchsorted(
                            self.event_signal.rescale("s"), e, side="right"
                        ).base
                    )
                    - 1
                )
                if i < 0:
                    continue
                x = int(
                    ((e - self.event_signal[i]) * self.analog_signal.sampling_rate).base
                )
                if e > self.analog_signal.t_stop:
                    continue
                try:
                    v = self.analog_signal.time_slice(e, e + 1 * pq.ms)[0]
                except:
                    v = None
                if (e - self.event_signal[i]) > self.window_size:
                    continue
                p[i] = (x, v)

            sg.idx_arr = p

    def get_erp(
        self,
        signal: neo.AnalogSignal = None,
        event_signal: neo.Event = None,
        channel=0,
        window_size=None,
    ):
        if signal is None:
            signal = self.analog_signal
        if event_signal is None:
            event_signal = self.event_signal
        if window_size is None:
            window_size = self.window_size

        signal_idx = next(
            i for i, x in enumerate(self.segment.analogsignals) if x.name == signal.name
        )
        events_idx = next(
            i for i, x in enumerate(self.segment.events) if x.name == event_signal.name
        )
        return self._get_erp(
            signal_idx, events_idx, channel, float(window_size.rescale(pq.second))
        )

    @lru_numpy_memmap()
    def _get_erp(
        self, signal_idx=None, event_signal_idx=None, channel=0, window_size=0.5
    ):
        signal = self.segment.analogsignals[signal_idx]
        event_signal = self.segment.events[event_signal_idx]
        erp = create_erp_signals(signal, event_signal, 0 * pq.s, window_size * pq.s)

        # s = int(np.array(signal.sampling_rate.base) // ((1 / window_size)))
        # erp = create_erp(
        #     signal.rescale("mV").as_array()[:, channel],
        #     (event_signal.as_array() - signal.t_start.base)
        #     * np.array(signal.sampling_rate, dtype=int),
        #     0,
        #     s,
        # )
        return erp

   

    _peaks = None

    def get_peaks(self):
        if self._peaks is not None:
            return self._peaks
        from scipy.signal import find_peaks

        dpts = self.get_erp()
        all_pts = []
        for d in dpts:
            pts, _ = find_peaks(d)
            pts_down, _ = find_peaks(-1 * d)
            pts = np.sort(np.hstack([pts, pts_down]).flatten())
            all_pts.append(pts)
        self._peaks = all_pts
        return all_pts

    def set_window_size(self, window_size):
        self.window_size = window_size * pq.ms
        self._get_erp.cache_clear()
        self.onLoadNewFile.emit()
    
    def get_unit_suggestions(self):
        # This will return a list of event suggestions provided by all the activated UnitSuggestion providers
        # This allows for custom algorithms to be implemented
        # The first one in the list will be the selected default.
        pass
    
    def set_data(
        self,
        analog_signal=None,
        event_signal=None,
        spike_groups=None,
    ):
        self._get_erp.cache_clear()
        self._peaks = None
        if analog_signal is not None:
            self.analog_signal = analog_signal
        if event_signal is not None:
            self.event_signal = event_signal
        if spike_groups is not None:
            self.spike_groups = spike_groups

        # ensure essentials are populated on state
        if self.spike_groups is None:
            spike_groups = [tracked_neuron_unit(event=Event())]
        if self.analog_signal is None:
            analog_signal = neo.AnalogSignal([], pq.mV, sampling_rate=1 * pq.Hz)
        if self.event_signal is None:
            event_signal = neo.Event()

        self.sampling_rate = int(np.array(self.analog_signal.sampling_rate.magnitude))
        if len(self.spike_groups) == 0:
            self.spike_groups.append(tracked_neuron_unit(event=Event()))

        self.update_idx_arrs()
        self.onLoadNewFile.emit()
        self.undo_queue = []

    def getUnitGroup(self, unit=None):
        if unit is None:
            unit = self.cur_spike_group
        if unit is None:
            return None
        return self.spike_groups[unit]

    def set_segment(self, data: neo.Segment):

        kwargs = {}
        cur_evt = [
            i for i, x in enumerate(data.events) if id(x) == id(self.event_signal)
        ]
        if len(cur_evt) == 0:
            event_signal = data.events[0]
            kwargs["event_signal"] = event_signal
        else:
            event_signal = data.events[cur_evt[0]]

        cur_analogsig = [
            i
            for i, x in enumerate(data.analogsignals)
            if id(x) == id(self.analog_signal)
        ]
        if len(cur_analogsig) == 0:
            analog_signal = data.analogsignals[0]
            kwargs["analog_signal"] = analog_signal
        else:
            analog_signal = data.analogsignals[cur_evt[0]]

        spike_event_groups = []

        for x in data.events:
            if not x.name.startswith("unit"):
                continue

            spike_event_groups.append(tracked_neuron_unit(event=x))

        cur_sel = [
            i
            for i, x in enumerate(spike_event_groups)
            if id(x.event) == id(self.spike_groups[self.cur_spike_group])
        ]
        if len(cur_sel) == 0:
            self.cur_spike_group = 0
        else:
            self.cur_spike_group = self.cur_sel[0]

        self.segment = data

        self.set_data(analog_signal, event_signal, spike_groups=spike_event_groups)


def load_file(data_path, type="h5", **kwargs):
    # currently this only loads the first segment
    if type == "h5":
        data = NixIO(data_path, mode="ro").read_block(0).segments
    # elif type == "dabsys":
    # data = import_dapsys_csv_files(data_path)[0].segments[0]
    elif type == "openEphys":
        # data = open_ephys_to_neo(data_path)
        data = neo.OpenEphysBinaryIO(data_path).read_block(0).segments[0]

    elif type == "nwb":
        data = neo.NWBIO(data_path).read_block(block_index=0).segments[0]

    elif type == "matlab":
        data = open_matlab_to_neo(data_path)

    elif type == "APTrack":
        d = APTrackDialog()
        val = d.exec_()

        config = d.get_config()
        t = config.pop("record")

        # HACK: Add DS4 channels
        DS4_channels = [
            APTrackRecording(
                "ADC2",
                TypeID.TTL,
                "env.ds4.stimVolt",
                "The DS5 1V per 10mA",
            ),
            APTrackRecording(
                "ADC2",
                TypeID.ANALOG,
                "env.ds4.stimVolt",
                "The DS5 1V per 10mA",
            ),
        ]

        data = open_aptrack(data_path, t, config, DS4_channels)
        # blk = neo.OpenEphysIO(data_path).read_block(0)
        # data = blk.segments[0]
    elif type == "spike2":
        data = neo.Spike2IO(data_path)
    if isinstance(data, (list, ObjectList)):  # multiple segments
        # create multianalogsignals if there are segments labeled as "ERP"
        final_sig = next(
            x
            for x in data
            if x.name != "ERP"
            and x.annotations.get("class_", "") != "multianalogsignal"
        )
        for seg in data:
            from .processing import MultiAnalogSignal

            if (
                seg.name == "ERP"
                or seg.annotations.get("class_", "") == "multianalogsignal"
            ):
                erp = MultiAnalogSignal(seg.analogsignals)
                final_sig.analogsignals.append(erp)

        data = final_sig
    if len(data.events) == 0:
        event_signal = Event()
    else:
        event_signal = data.events[0]
    signal_chan = data.analogsignals[0]

    spike_event_groups = []

    for x in data.events:
        x: Event = x
        if (not x.name.startswith("unit")) and (
            x.annotations.get("type", "") != "unit"
        ):
            continue
        # info = x.times[:,np.newaxis]-event_signal
        # p = [None for x in range(len(event_signal))]

        spike_event_groups.append(tracked_neuron_unit(event=x))

    data.file_origin = data_path
    return data, signal_chan, event_signal, spike_event_groups


def prompt_for_neo_file(type):
    if type is None:
        type = QInputDialog().getItem(
            None,
            "Select file type",
            "Filetype",
            ["h5", "openEphys", "APTrack", "matlab", "spike2", "nwb"],
        )[0]

    if type in ("h5", "spike2", "nwb"):
        fname = QFileDialog.getOpenFileName(None, "Open")[0]
    elif type in ("openEphys", "APTrack", "matlab"):
        fname = QFileDialog.getExistingDirectory(None, "Open OpenEphys")
    else:
        raise Exception(f"Unknown filetype: {type}")
    return fname, type


def open_matlab_to_neo(folder):
    from pathlib import Path

    from scipy.io import loadmat

    matfiles = Path(folder).glob("*.mat")
    seg = neo.Segment()
    for m in matfiles:
        mf = loadmat(m)
        asig = neo.AnalogSignal(
            mf["data"].T,
            pq.V,
            sampling_rate=mf["samplerate"][0, 0] * pq.Hz,
            name=m.stem,
        )
        seg.analogsignals.append(asig)
    return seg


def open_smrx_to_neo(file):
    pass


class APTrackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QFormLayout()
        self.setLayout(self.layout)

        self.layout.addRow(QLabel("custom parameters"))
        self.sbRecordNumber = QSpinBox()
        self.sbRecordNumber.setMinimum(0)

        self.layout.addRow(QLabel("recording"), self.sbRecordNumber)
        config = {
            "rd.0": "CH1",
            "stimVolt": "ADC4",
            "stim": "ADC5",
            "thermode": "ADC7",
            "button": "ADC8",
        }
        self.inputsConfig = {}
        for k, v in config.items():
            inputBox = QLineEdit(v)

            self.inputsConfig[k] = inputBox
            self.layout.addRow(QLabel(k), inputBox)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.layout.addRow(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_config(self):
        config = {}
        config["record"] = self.sbRecordNumber.value()
        for k, v in self.inputsConfig.items():
            config[k] = v.text()

        return config

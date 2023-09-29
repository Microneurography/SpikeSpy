import cProfile
import copy
import sys
from dataclasses import dataclass, field
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.style as mplstyle
import neo
import numpy as np
import PySide6
import quantities as pq
from matplotlib.widgets import PolygonSelector
from neo.io import NixIO
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
    QSettings,
)
from PySide6.QtGui import QAction, QColor, QShortcut, QKeySequence, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

# import PySide6QtAds as QtAds
from .mng_file_selector import QNeoSelector
from .MultiTraceView import MultiTraceView
from .NeoSettingsView import NeoSettingsView
from .SingleTraceView import SingleTraceView
from .SpikeGroupTable import SpikeGroupTableView
from .TrackingView import TrackingView
from .UnitView import UnitView
from .ViewerState import ViewerState, prompt_for_neo_file, tracked_neuron_unit
from .EventView import EventView

import PySide6QtAds as QtAds

mplstyle.use("fast")


class MdiView(QMainWindow):
    # signals
    loadFile = Signal(str, str)

    def savePerspectives(self):
        # TODO: this currently works if all the required window titles are open. need to:
        # 1. create unique window names
        # 2. save the details of the windows to reopen
        # 3. save the 'state' of the windows (zoom levels etc.)
        self.dock_manager.addPerspective("main")
        self.dock_manager.savePerspectives(self.settings_file)

    def loadPerspectives(self):
        self.dock_manager.loadPerspectives(self.settings_file)
        self.dock_manager.openPerspective("main")

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.settings_file = QSettings("spikespy.ini", QSettings.Format.IniFormat)
        self.window_options = {
            # 'TraceAnnotation': TraceView,
            "MultiTrace": MultiTraceView,
            "UnitView": UnitView,
            "SpikeGroupTable": SpikeGroupTableView,
            "SingleTraceView": SingleTraceView,
            "Settings": NeoSettingsView,
            "TrackingView": TrackingView,
            "Data": QNeoSelector,
            "Events": EventView,
        }
        self.cur_windows = []
        import pkg_resources

        version = pkg_resources.get_distribution("spikespy").version
        self.setWindowTitle(f"SpikeSpy - {version}")
        self.state = state or ViewerState(**kwargs)
        self.loadFile.connect(self.state.loadFile)
        self.state.onLoadNewFile.connect(
            lambda self=self: self.setWindowTitle(
                f"SpikeSpy ({version}) - {Path(self.state.title).name}"
            )
        )
        self.dock_manager = QtAds.CDockManager(self)
        self.dock_manager.addPerspective("main")
        self.mdi = QMdiArea()
        # self.setCentralWidget(self.mdi)

        # for k in [
        #     "MultiTrace",
        #     "UnitView",
        #     "SpikeGroupTable",
        #     "SingleTraceView",
        #     "TrackingView",
        # ]:
        #     self.newWindow(k)

        self.toolbar = self.addToolBar("")
        for k in self.window_options.keys():
            act = self.toolbar.addAction(k)
            act.triggered.connect(lambda *args, k=k: self.newWindow(k))

        # Menu bar
        self.menubar = self.menuBar()

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(
            QAction("Open", self, shortcut="Ctrl+O", triggered=lambda: self.open())
        )
        file_menu.addAction(
            QAction(
                "Save as nixio",
                self,
                shortcut="Ctrl+S",
                triggered=lambda: self.save_as(),
            )
        )
        file_menu.addAction(
            QAction(
                "Export spikes as csv",
                self,
                shortcut="Ctrl+E",
                triggered=lambda: self.export_csv(),
            )
        )
        file_menu.addAction(
            QAction(
                "import spikes csv",
                self,
                shortcut="Ctrl+I",
                triggered=lambda: self.import_csv(),
            )
        )
        edit_menu = self.menubar.addMenu("&Edit")
        edit_menu.addAction(
            QAction(
                "undo", self, shortcut="Ctrl+Z", triggered=lambda: self.state.undo()
            )
        )
        edit_menu.addAction(
            QAction("save perspective", self, triggered=self.savePerspectives)
        )
        edit_menu.addAction(
            QAction("load perspective", self, triggered=self.loadPerspectives)
        )

        # key shortcuts
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        self.shortcut_profile = QShortcut(QKeySequence(Qt.Key_F1), self)

        def profile():
            if self.is_profiling:
                self.profiler.disable()
                self.is_profiling = False
                nom = QFileDialog.getSaveFileName(self, "Export profile")
                if nom is None:

                    return
                self.profiler.dump_stats(nom[0])
                self.profiler = cProfile.Profile()
                import subprocess

                subprocess.run(["snakeviz", nom[0]])

            else:
                self.is_profiling = True
                self.profiler.enable()

        self.shortcut_profile.activated.connect(lambda: profile())

        self.shortcut_next = QShortcut(
            QKeySequence(Qt.Key_Down), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_next.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno + 1)
        )

        self.shortcut_prev = QShortcut(
            QKeySequence(Qt.Key_Up), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_prev.activated.connect(
            lambda: self.state.setStimNo(self.state.stimno - 1)
        )

        self.shortcut_del = QShortcut(
            QKeySequence(Qt.Key_Backspace), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_del.activated.connect(lambda: self.state.setUnit(None))

        self.move_mode = "snap"

        def move(dist=1):
            cur_point = (
                self.state.spike_groups[self.state.cur_spike_group].idx_arr[
                    self.state.stimno
                ]
            )[0]
            if self.move_mode == "snap":
                from scipy.signal import find_peaks

                dpts = self.state.get_erp()[self.state.stimno]
                pts, _ = find_peaks(dpts)
                pts_down, _ = find_peaks(-1 * dpts)
                pts = np.sort(np.hstack([pts, pts_down]).flatten())
                i = pts.searchsorted(cur_point)

                if dist > 0 and pts[i] != cur_point:
                    dist -= 1

                dist = pts[i + dist] - cur_point

            self.state.setUnit(cur_point + dist)  # TODO: make method

        self.shortcut_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_left.activated.connect(lambda: move(-1))

        self.shortcut_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_right.activated.connect(lambda: move(1))

        self.shortcut_numbers = [
            QShortcut(QKeySequence(Qt.Key_1), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_2), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_3), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_4), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_5), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_6), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_7), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_8), self, context=Qt.ApplicationShortcut),
            QShortcut(QKeySequence(Qt.Key_9), self, context=Qt.ApplicationShortcut),
        ]
        for i, x in enumerate(self.shortcut_numbers):
            x.activated.connect(lambda i=i: self.state.setUnitGroup(i))

        def toggle_snap():
            self.move_mode = None if self.move_mode == "snap" else "snap"

        self.shortcut_snap = QShortcut(
            QKeySequence(Qt.Key_S), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_snap.activated.connect(toggle_snap)

        def copy_previous():
            try:
                sg = self.state.spike_groups[self.state.cur_spike_group].idx_arr
                new_x = next(
                    sg[x][0]
                    for x in range(self.state.stimno - 1, -1, -1)
                    if sg[x] is not None
                )
                self.state.setUnit(new_x)
            except StopIteration:
                pass

        self.shortcut_copy_previous = QShortcut(
            QKeySequence(Qt.Key_C), self, context=Qt.ApplicationShortcut
        )
        self.shortcut_copy_previous.activated.connect(copy_previous)

    def export_csv(self):
        save_filename = QFileDialog.getSaveFileName(self, "Export")[0]
        from csv import writer

        if save_filename is None:
            return
        with open(save_filename, "w") as f:
            w = writer(f)
            # self.state.segment
            w.writerow(["SpikeID", "Stimulus_number", "Latency (ms)", "Timestamp(ms)"])
            for i, sg in enumerate(self.state.spike_groups):
                for timestamp in sg.event:
                    stim_no = self.state.event_signal.searchsorted(timestamp) - 1
                    latency = (timestamp - self.state.event_signal[stim_no]).rescale(
                        pq.ms
                    )
                    w.writerow(
                        [f"{i}", stim_no, latency.base, timestamp.rescale(pq.ms).base]
                    )

    def import_csv(self):
        open_filename = QFileDialog.getOpenFileName(self, "csv import")[0]
        if open_filename is None:
            return
        from csv import DictReader

        with open(open_filename, "r") as f:
            r = DictReader(f)
            try:
                k = next(x for x in r.fieldnames if "timestamp" in x.lower())
            except StopIteration:
                import logging

                logging.warn("StopIteration not found")
                return

            unit = pq.ms
            out = {}
            for row in r:
                spikeid = str(row.get("SpikeID", "0"))
                out[spikeid] = out.get(spikeid, []) + [row[k]]

            out2 = []
            for k, v in out.items():
                timestamps = neo.Event(
                    np.array(v, dtype=np.float64), units=unit, name=f"unit_{k}"
                )
                out2.append(tracked_neuron_unit(event=timestamps.rescale(pq.s)))

            self.state.set_data(spike_groups=(self.state.spike_groups or []) + out2)

    def newWindow(self, k, pos=QtAds.TopDockWidgetArea):
        w = self.window_options[k](parent=self, state=self.state)
        w2 = QtAds.CDockWidget(k)
        w2.setWidget(w)
        w2.setFeature(QtAds.CDockWidget.DockWidgetDeleteOnClose, True)
        self.dock_manager.addDockWidget(QtAds.NoDockWidgetArea, w2)
        self.cur_windows.append(w2)
        w2.show()

    @Slot()
    def open(self, type=None):
        """
        triggered on file->open
        """
        fname, type = prompt_for_neo_file(type)
        self.loadFile.emit(fname, type)

    @Slot()
    def save_as(
        self,
    ):  # TODO: move out
        """
        triggered on file->save
        """
        fname = QFileDialog.getSaveFileName(self, "Save as", filter=".h5")[0]
        # s = neo.Segment()
        if fname == "":
            return

        s = self.state.segment or neo.Segment()

        save_file(
            fname,
            self.state.spike_groups,
            s,
            event_signal=self.state.event_signal,
            signal_chan=self.state.analog_signal,
        )


def save_file(
    filename,
    spike_groups,
    data=None,
    metadata=None,
    event_signal=None,
    signal_chan=None,
):
    """
    Saves a file containing the spike groups
    """
    if metadata is None:
        metadata = {
            "current_user": environ.get("USERNAME", environ.get("USER")),
            "date": datetime.now(),
        }

    def create_event_signals():
        events = []

        # 1. create timestamps from them
        for i, sg in enumerate(spike_groups):
            ts = []
            for e, s in zip(event_signal, sg.idx_arr):
                if s is None:
                    continue
                s_in_sec = s[0] / signal_chan.sampling_rate
                ts.append(e + s_in_sec)
            events.append(
                neo.Event(
                    np.array(ts),
                    name=f"unit_{i}",
                    annotations=metadata,
                    units="s",
                )
            )  # TODO: add more details about the events

        return events

    from copy import deepcopy

    if data is not None:
        data2 = deepcopy(data)

        # remove all 'nix_names' which prevent saving the file
        for x in [*data2.analogsignals, *data2.events, data2]:
            if "nix_name" in x.annotations:
                del x.annotations["nix_name"]

        data2.analogsignals = [x.rescale("mV") for x in data2.analogsignals]

        # remove previous unit annotations
        data2.events = [x for x in data2.events if not x.name.startswith("unit_")]
    else:
        data2 = neo.Segment()

    for x in create_event_signals():
        data2.events.append(x)

    blk = neo.Block(name="main")
    blk.segments.append(data2)
    if Path(filename).exists():
        Path(filename).unlink()
    n = NixIO(filename, mode="rw")
    n.write_block(blk)
    n.close()


class EventHistoryView(QWidget):
    """
    #TODO: should show the recent history of the selected spike (in table?)
    """

    pass


def align_spikegroup(spikegroup, erp_arr):
    """given a spike group attempt to align using convolution - note this should go outside of the UI component"""
    # 1. get spike events as 2d matrix
    window = 200
    arr = [
        erp_arr[i, x[0] - window : x[0] + window]
        for i, x in enumerate(spikegroup.idx_arr)
    ]

    # 2. correlate event signals with eachother
    np.convolve(arr, arr, axis=-1)
    # 3. take n-best overlaps & create template

    # 4. align all other signals to this template.
    # TODO
    pass


def run():
    app = QApplication(sys.argv)
    icon_path = Path(sys.modules[__name__].__file__).parent.joinpath("ui/icon.svg")
    app.setWindowIcon(QIcon(QPixmap(str(icon_path))))
    # data, signal_chan, event_signal, spike_groups = load_file(
    # )
    # w = TraceView(
    #     analog_signal=signal_chan, event_signal=event_signal, spike_groups=spike_groups
    # )
    w = MdiView()
    if len(sys.argv) > 1:
        w.state.loadFile(sys.argv[1])
    # w = SpikeGroupView()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()

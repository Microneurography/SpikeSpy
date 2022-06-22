import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.style as mplstyle
import neo
import numpy as np
import PySide6
import quantities as pq
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import (FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from neo import Event
from neo.io import NixIO
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, Qt,
                            Signal, Slot)
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QFileDialog, QFormLayout,
                               QHBoxLayout, QInputDialog, QMainWindow,
                               QMdiArea, QMdiSubWindow, QMenu, QMenuBar,
                               QPushButton, QSpinBox, QTableView, QVBoxLayout,
                               QWidget)

from .APTrack_experiment_import import process_folder as open_aptrack
from .MultiTraceView import MultiTraceView
from .NeoSettingsView import NeoSettingsView
from .TrackingView import TrackingView
from .ViewerState import ViewerState, tracked_neuron_unit

mplstyle.use('fast')



class UnitView(QMainWindow):
    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):
        super().__init__(parent)
        self.state: ViewerState = None
        if state is not None:
            self.set_state(state)

        xsize = 640
        ysize = 480
        dpi = 100

        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
 
        #  create widgets

        self.view = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.view, self)
        self.addToolBar(self.toolbar)
        self.setCentralWidget(self.view)
        self.axes = None
        self.w = None
        self.selected_line = None
        self.lines = None
        self.setup_figure()

    def set_state(self, state):
        self.state = state
        self.state.onLoadNewFile.connect(self.updateAll)
        self.state.onStimNoChange.connect(self.update_curstim_line)
        self.state.onUnitGroupChange.connect(self.updateAll)
        # self.state.onUnitChange.connect(self.update_displayed_data)
        self.state.onUnitChange.connect(self.update_curstim_line)

    @Slot()
    def updateAll(self):
        self.fig.clear()
        self.setup_figure()

    def setup_figure(self):
        if self.state.analog_signal is None:
            return
        self.w = int(self.state.sampling_rate * 0.002)
        self.axes = {"main": self.fig.add_subplot(111)}
        ax = self.axes["main"]
        xarr = np.arange(-self.w, +self.w)
        self.selected_line = ax.plot(
            xarr, np.zeros(len(xarr)), color="purple", zorder=10
        )[0]

        ax = self.axes["main"]

        ax.axvline(0, ls="--", color="black")
        # ax.clear()

        sg = self.state.getUnitGroup()
        sig_erp = self.state.analog_signal_erp
        idx_arr = sg.idx_arr
        self.lines = {}
        for i, idx, d in zip(range(len(sig_erp)), idx_arr, sig_erp):
            if idx is None:
                ydata = np.zeros(self.w * 2)
            else:
                ydata = d[idx[0] - self.w : idx[0] + self.w]

            self.lines[i] = ax.plot(
                np.arange(-self.w, self.w), ydata, alpha=0.3, color="gray"
            )[0]
            if idx is None:
                self.lines[i].set_visible(False)

        self.update_curstim_line(self.state.stimno)
        self.view.draw()

    @Slot()
    def update_curstim_line(self, stimno):

        idx = self.state.getUnitGroup().idx_arr[stimno]
        grayline = self.lines[stimno]
        if idx is None:
            grayline.set_visible(False)
            if stimno == self.state.stimno:
                self.selected_line.set_visible(False)
        else:

            d = self.state.analog_signal_erp[stimno]
            yarr = d[idx[0] - self.w : idx[0] + self.w]

            grayline.set_visible(True)
            grayline.set_ydata(yarr)
            if stimno == self.state.stimno:
                self.selected_line.set_visible(True)
                self.selected_line.set_ydata(yarr)

            self.axes["main"].relim()
            self.axes["main"].autoscale_view(True, True, True)
        self.view.draw()

    @Slot()
    def view_clicked(self, event):
        pass

    def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
        return self.parentWidget().keyPressEvent(event)


class PolygonSelectorTool:  # This is annoyingly close - there are two styles of tools in matplotlib, and i cannot get this one to work embedded in QT (no toolmanager)
    """Polygon selector"""

    default_keymap = "S"
    description = "PolygonSelection"
    default_toggled = True

    def __init__(self, fig, *args, **kwargs):
        self.fig = fig
        self.poly = PolygonSelector(self.fig.axes[0], self.onselect)
        self.poly.disconnect_events()

    def enable(self, *args):
        self.poly.connect_default_events()

    def disable(self, *args):
        self.poly.disconnect_events()

    def onselect(self, verts):
        print(verts)



class SpikeGroupTableView(QWidget):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = ...,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.spike_tablemodel = SpikeGroupTableModel(lambda: state.spike_groups)

        self.tbl = QTableView(self)
        self.tbl.setModel(self.spike_tablemodel)
        self.state = state
        state.onUnitChange.connect(self.spike_tablemodel.update)
        state.onUnitGroupChange.connect(self.spike_tablemodel.update)
        state.onLoadNewFile.connect(self.spike_tablemodel.update)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.addButton = QPushButton("+")

        self.addButton.clicked.connect(self.state.addUnitGroup)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.tbl)
        vlayout.addWidget(self.addButton)
        self.setLayout(vlayout)

        self.tbl.selectionModel().selectionChanged.connect(self.set_selection)

    @Slot()
    def set_selection(self, x):
        y = x.toList()
        if len(y) > 0:
            sg = x.first().top()
        else:
            sg = 0
        self.state.setUnitGroup(sg)


class SingleTraceView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = ...,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.state = state

        xsize = 1024
        ysize = 480
        dpi = 100

        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        #  create widgets
        self.view = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar2QT(self.view, self)
        self.ax = self.fig.add_subplot(111)
        self.addToolBar(self.toolbar)

        self.identified_spike_line = self.ax.axvline(6000, zorder=0)
        self.trace_line_cache = None

        self.setupFigure()
        self.setCentralWidget(self.view)

        self.state.onLoadNewFile.connect(self.setupFigure)
        self.state.onUnitChange.connect(self.updateFigure)
        self.state.onUnitGroupChange.connect(self.updateFigure)
        self.state.onStimNoChange.connect(self.updateFigure)

        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.select_local_maxima_width = 1

        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

    @Slot()
    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1:
            return

        if e.inaxes == self.ax:
            self.set_cur_pos(e.xdata)

    @Slot()
    def setupFigure(self):
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x / self.state.sampling_rate)
        )
        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.set_xticks(
            np.arange(
                0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 100
            )
        )
        self.ax.set_xticks(
            np.arange(
                0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 1000
            ),
            minor=True,
        )
        self.ax.grid(True, which="both")

        if self.trace_line_cache is not None:
            self.trace_line_cache.remove()
            self.trace_line_cache = None

        self.fig.tight_layout()
        self.updateFigure()

    @Slot()
    def updateFigure(self):
        sg = self.state.getUnitGroup()
        dpts = self.state.analog_signal_erp[self.state.stimno]

        cur_point = sg.idx_arr[self.state.stimno]
        if self.trace_line_cache is None:
            self.trace_line_cache = self.ax.plot(dpts, color="purple")[0]
        else:
            self.trace_line_cache.set_data(np.arange(len(dpts)), dpts)

        if cur_point is not None:
            self.identified_spike_line.set_data(([cur_point[0], cur_point[0]], [0, 1]))
            self.identified_spike_line.set_visible(True)

        else:
            self.identified_spike_line.set_visible(False)

        self.view.draw()

    def set_cur_pos(self, x):
        x = round(x)
        dpts = self.state.analog_signal_erp[self.state.stimno]
        if self.select_local_maxima_width > 1:
            w = self.select_local_maxima_width
            if x < 0 + w or x > self.state.analog_signal_erp.shape[1] - w:
                return

            x += np.argmax(np.abs(dpts[x - w : x + w])) - w

        self.state.setUnit(x)

    def keyPressEvent(self, e):
        dist = max(self.select_local_maxima_width + 1, 1)

        if e.key() == Qt.Key_Down:
            self.state.setStimNo(self.state.stimno + 1)
        elif e.key() == Qt.Key_Up:
            self.state.setStimNo(self.state.stimno - 1)
        elif e.key() in (Qt.Key_Delete, Qt.Key_D):
            self.state.setUnit(None)

        elif e.key() == Qt.Key_Left:
            self.set_cur_pos(
                (
                    self.state.spike_groups[self.state.cur_spike_group].idx_arr[
                        self.state.stimno
                    ]
                    or [np.int(np.mean(self.ax.get_xlim()))]
                )[0]
                - dist  # TODO: make method
            )
        elif e.key() == Qt.Key_Right:
            self.set_cur_pos(
                (
                    self.state.spike_groups[self.state.cur_spike_group].idx_arr[
                        self.state.stimno
                    ]
                    or [np.int(np.mean(self.ax.get_xlim()))]
                )[0]
                + dist
            )
        elif e.key() == Qt.Key_C:
            try:
                sg = self.state.spike_groups[self.state.cur_spike_group].idx_arr
                new_x = next(
                    sg[x][0]
                    for x in range(self.state.stimno - 1, -1, -1)
                    if sg[x] is not None
                )
                self.set_cur_pos(new_x)
            except StopIteration:
                pass
        elif e.key() == Qt.Key_Z:
            pass  # TODO: zoom into current spike
        elif e.key() == Qt.Key_T:
            # automatically track (#TODO: make this less cryptic & more generic)
            from .basic_tracking import track_basic

            unit_events = self.state.getUnitGroup().event
            last_event = unit_events.searchsorted(
                self.state.event_signal[self.state.stimno] + (0.5 * pq.s)
            )  # find the most recent event

            starting_time = unit_events[max(last_event-1,0)]
            window = 0.02 * pq.s
            threshold = (
                self.state.analog_signal[
                    self.state.analog_signal.time_index(starting_time)
                ][0]
                * 0.8
            )  # 0.1 * pq.mV

            evt2 = track_basic(
                self.state.analog_signal,
                self.state.event_signal,
                starting_time=starting_time,
                window=window,
                threshold=threshold,
            )
            self.state.updateUnit(
                event=unit_events.merge(evt2)
            )


class MdiView(QMainWindow):
    # signals
    loadFile = Signal(str, str)

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
        **kwargs,
    ) -> None:
        super().__init__(parent)

        self.window_options = {
            # 'TraceAnnotation': TraceView,
            "MultiTrace": MultiTraceView,
            "UnitView": UnitView,
            "SpikeGroupTable": SpikeGroupTableView,
            "SingleTraceView": SingleTraceView,
            "Settings": NeoSettingsView,
            "TrackingView":TrackingView
        }
        self.cur_windows = []

        self.state = state or ViewerState(**kwargs)
        self.loadFile.connect(self.state.loadFile)

        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)

        for k, v in self.window_options.items():

            w = v(parent=self, state=self.state)
            w = self.mdi.addSubWindow(w)
            self.cur_windows.append(w)

        self.toolbar = self.addToolBar("")
        for k in self.window_options.keys():
            act = self.toolbar.addAction(k)
            act.triggered.connect(lambda *args, k=k: self.newWindow(k))

        # Menu bar
        self.menubar = self.menuBar()

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(
            QAction("Open", self, shortcut="Ctrl+O", triggered=self.open)
        )
        file_menu.addAction(
            QAction(
                "Open dapsys",
                self,
                shortcut="Ctrl+O",
                triggered=lambda: self.open("dabsys"),
            )
        )
        file_menu.addAction(
            QAction("Save as nixio", self, shortcut="Ctrl+S", triggered=self.save_as)
        )
        file_menu.addAction(
            QAction("Export as csv", self, shortcut= "Ctrl+E", triggered=self.export_csv )
        )
    def export_csv(self):
        save_filename = QFileDialog.getSaveFileName(self, "Export")[0]
        from csv import writer
        with open(save_filename, 'w') as f:
            w = writer(f)
            w.writerow(['SpikeID','Stimulus_number','Latency (ms)','Timestamp(ms)']) 
            for i, sg in enumerate(self.state.spike_groups):
                for timestamp in sg.event:
                    stim_no = self.state.event_signal.searchsorted(timestamp)-1
                    latency = (timestamp - self.state.event_signal[stim_no]).rescale(pq.ms)
                    w.writerow([f'{i}', stim_no, latency.base, timestamp.rescale(pq.ms).base ])

        


    def newWindow(self, k):
        w = self.window_options[k](parent=self, state=self.state)
        w = self.mdi.addSubWindow(w)
        self.cur_windows.append(w)
        w.show()

    @Slot()
    def open(self, type=None):
        """
        triggered on file->open
        """
        if type is None:
            type = QInputDialog().getItem(
                self,
                "Select file type",
                "Filetype",
                ["h5", "dabsys", "openEphys", "openEphysBinary"],
            )[0]

        if type == "h5":
            fname = QFileDialog.getOpenFileName(self, "Open")[0]
        if type == "openEphys":
            fname = QFileDialog.getExistingDirectory(self, "Open OpenEphys")
            # options = list(set(str(x).rsplit("_",1)[0] for x in Path(fname).glob("*.continuous")))
            # type = QInputDialog().getItem(self, "Select oe option","recording",options)

        if type in ("dabsys", "openEphysBinary"):
            fname = QFileDialog.getExistingDirectory(self, "Open Dabsys")

        print(fname)
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

        s = self.state.segment or neo.Segment()

        save_file(
            fname,
            self.state.spike_groups,
            s,
            event_signal=self.state.event_signal,
            signal_chan=self.state.analog_signal,
        )


import copy
from datetime import datetime
from os import environ
from pathlib import Path

import neo


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


class SpikeGroupTableModel(QAbstractTableModel):
    def __init__(self, spikegroups_func=None):
        """
        spikegroups_func is a functino which returns the spike groups
        """
        QAbstractTableModel.__init__(self)
        self.spikegroups_func = spikegroups_func
        self.spikegroups = self.spikegroups_func()
        self.headers = ["SpikeID", "start", "end", "notes", "fibre class"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.spikegroups_func() or [])

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(
        self, section: int, orientation: PySide6.QtCore.Qt.Orientation, role: int = ...
    ) -> Any:
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.headers[section]
        else:
            return section

    @Slot()
    def update(self):
        self.modelAboutToBeReset.emit()
        self.spikegroups = self.spikegroups_func()
        self.modelReset.emit()

    def data(
        self,
        index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex],
        role: int = ...,
    ) -> Any:
        column = index.column()
        row = index.row()
        sg = self.spikegroups_func()[row]
        if role == Qt.DisplayRole:
            if column == 0:  # SpikeID
                return row
            elif column in (1, 2):  # start/end
                if sg.get_window() is None:
                    return ""
                return sg.get_window()[column - 1]
            else:
                return ""
        elif role == Qt.BackgroundRole:
            return QColor(Qt.white)


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

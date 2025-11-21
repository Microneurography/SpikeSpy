import itertools
import sys
from dataclasses import dataclass, field

import matplotlib
import matplotlib.axes
from matplotlib.collections import PatchCollection
import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from neo import Event
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QMainWindow,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QDialog,
    QPushButton,
    QFormLayout,
    QDoubleSpinBox,
    QToolBar,
    QLabel,
    QSpacerItem,
    QSizePolicy,
)
from PySide6 import QtCore
from PySide6.QtCore import QTimer
from .NeoSettingsView import NeoSettingsView
from .ViewerState import ViewerState, tracked_neuron_unit

from .QMatplotlib import QMatplotlib
from matplotlib.lines import Line2D
import time

# mplstyle.use("fast")

from matplotlib.axes import Axes
from .helpers import qsignal_throttle_wrapper
from matplotlib.image import AxesImage
from .CachedAxesImage import CachedAxesImage


class falling_leaf_plotter:
    def __init__(self):
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.percentiles = []
        self.cb1 = None
        self.cb2 = None
        self.mode = None
        self.partial = False
        self.hline = None

    def setup(self, ax: Axes, erp, sampling_rate=1000, mode="heatmap"):
        self.percentiles = (
            np.arange(0, 100) * 2 * np.sqrt(np.mean(erp**2))
        )  # np.percentile(erp, np.arange(100))
        self.mode = mode

        # self.cb2 = ax.callbacks.connect("ylim_changed", self.setup)

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x / sampling_rate)
        )
        ax.xaxis.set_major_formatter(func_formatter)
        loc = matplotlib.ticker.MultipleLocator(
            base=sampling_rate / 100
        )  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

    def plot_main(self, mode, ax: Axes, erp, partial=True, clim=None):
        self.mode = mode

        ylim = ax.get_ylim()
        ylim = [max(int(np.floor(ylim[0])), 0), int(np.ceil(ylim[1]))]
        xlim = ax.get_xlim()
        xlim = [max(int(np.floor(xlim[0])), 0), int(np.ceil(xlim[1]))]
        if partial:
            im_data = erp[ylim[0] : ylim[1], xlim[0] : xlim[1]]
        else:
            im_data = erp
            xlim = [0, erp.shape[1]]
            ylim = [0, erp.shape[0]]
        im_data = im_data
        if mode == "heatmap":
            if self.ax_track_leaf is not None:
                try:
                    self.ax_track_leaf.remove()
                except:
                    pass
                self.ax_track_leaf = None

            if self.ax_track_cmap is not None:
                try:
                    self.ax_track_cmap.remove()
                    self.ax_track_cmap = None
                except:
                    pass
            #    self.ax_track_cmap.remove()
            ax.set_autoscale_on(False)

            self.ax_track_cmap = CachedAxesImage(
                ax,
                cmap="gray_r",
                interpolation="antialiased",
                rolling_max=True,
            )

            self.ax_track_cmap.set_data(np.clip(im_data, 0, np.max(im_data)))
            self.ax_track_cmap.set_extent((xlim[0], xlim[1], ylim[1], ylim[0]))
            if clim is None:
                clim = (self.percentiles[40], self.percentiles[95])
            self.ax_track_cmap.set_clim(*clim)
            self.ax_track_cmap._scale_norm(None, None, None)
            ax.add_artist(self.ax_track_cmap)
            # self.ax_track_cmap.set_animated(True)
            # ax.draw_artist(self.ax_track_cmap)
            # self.ax_track_cmap.set_visible(False)
        elif mode == "lines":
            if self.ax_track_cmap is not None:
                self.ax_track_cmap.set_visible(False)

            p90 = np.std(self.im_data) * 10  # self.percentiles[95] * 4
            analog_signal_erp_norm = np.clip(
                im_data,
                -p90,
                p90,
            ) / (p90 * 2)
            if self.ax_track_leaf is not None:
                # return
                self.ax_track_leaf.remove()
                self.ax_track_leaf = None

            x = np.arange(analog_signal_erp_norm.shape[1]) + (xlim[0] if partial else 0)
            ys = (analog_signal_erp_norm * -1) + (
                np.arange(analog_signal_erp_norm.shape[0]) + (ylim[0] if partial else 0)
            )[:, np.newaxis]
            segs = [np.column_stack([x, y]) for y in ys]
            from matplotlib.collections import LineCollection

            self.ax_track_leaf = LineCollection(
                segs, array=x, linestyles="solid", color="gray"
            )

            ax.add_artist(self.ax_track_leaf)

            # self.ax_track_leaf = ax.plot(  # could increase performance to just plot lines in view. but then no blitting...
            #     (
            #         (analog_signal_erp_norm * -1)
            #         + (np.arange(analog_signal_erp_norm.shape[0]) + ylim[0])[
            #             :, np.newaxis
            #         ]
            #     ).T,
            #     color="gray",
            #     zorder=10,
            # )

    def highlight_stim(self, ax, stimNo, partial=True):
        if stimNo is None:
            return
        if self.mode == "lines":
            from matplotlib import lines

            ax_leaf_paths = self.ax_track_leaf.get_paths()

            to_color_idx = stimNo - (np.floor(ax.get_ylim()[0]) if partial else 0)
            new_colors = [
                "purple" if x == to_color_idx else "gray"
                for x in range(len(ax_leaf_paths))
            ]
            self.ax_track_leaf.set_colors(new_colors)
            # ax.draw_artist(self.ax_track_leaf)
            return self.ax_track_leaf

            # op: lines.Line2D = self.mainPlotter.ax_track_leaf[stimNo]

            # self.hline = lines.Line2D(
            #     *self.mainPlotter.ax_track_leaf[to_color_idx].get_data()
            # )
            # self.hline.update_from(op)
            # self.hline.set_color("purple")
        else:
            # self.hline.set_visible(True)
            if self.hline is not None:
                try:
                    self.hline.remove()
                except:
                    pass
            self.hline = ax.axhline(stimNo)
            # self.hline.set_animated(True)

            # ax.draw_artist(self.hline)
            return self.hline

    def plot_spikegroup(self, ax, sg, **kwargs):
        kwargs = {"alpha": 0.5, **kwargs}

        points = np.array(
            [(x[0], i) for i, x in enumerate(sg.idx_arr) if x is not None]
        )
        if len(points) == 0:
            return
        scat = ax.scatter(points[:, 0], points[:, 1], s=4, **kwargs)
        # scat.set_animated(True)
        # ax.draw_artist(scat)
        return [scat]


class MultiTraceView(QMainWindow):
    right_ax_data = {}

    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):

        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.ClickFocus)
        self.state: ViewerState = None
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.points_spikegroup = None
        self.hline = None
        self.figcache = None
        self.references = []
        self.messages = None

        # throttle the update for upateAll to every 500ms when using the comboboxes.
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.updateAll)
        self.update_timer.setInterval(500)
        self.close_conn = self.destroyed.connect(lambda: self.closeEvent())

        xsize = 1024
        ysize = 480
        dpi = 80
        self.plotter = falling_leaf_plotter()

        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)

        self.mode = "heatmap"

        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])

        self.ax = self.fig.add_subplot(self.gs[0, 0])
        self.ax_right = []
        # self.ax_right_fig = self.fig.add_subfigure(self.gs[0,1])

        # self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # create widgets
        self.view = FigureCanvas(self.fig)
        # self.view.update()
        # self.ps = PolygonSelectorTool(self.fig)
        # self.ps.enable()
        self.toolbar = NavigationToolbar2QT(self.view, self)
        # act = self.toolbar.addAction("selector")

        self.lock_to_stim = False

        self.addToolBar(self.toolbar)
        layout = QVBoxLayout()
        self.lowerSpinBox = QDoubleSpinBox(self)
        self.lowerSpinBox.setDecimals(2)
        self.lowerSpinBox.setSingleStep(0.1)
        self.lowerSpinBox.setRange(0, 10)
        self.lowerSpinBox.valueChanged.connect(lambda x: self.update_timer.start())
        self.lowerSpinBox.setValue(0.5)
        self.lowerSpinBox.setMinimumWidth(100)

        self.upperSpinBox = QDoubleSpinBox(self)
        self.upperSpinBox.setRange(0, 10)
        self.upperSpinBox.setDecimals(2)
        self.upperSpinBox.setSingleStep(0.1)
        self.upperSpinBox.setValue(2)
        self.upperSpinBox.setMinimumWidth(100)

        self.lowerSpinBox.valueChanged.connect(self.upperSpinBox.setMinimum)
        self.upperSpinBox.valueChanged.connect(self.lowerSpinBox.setMaximum)
        self.upperSpinBox.valueChanged.connect(lambda x: self.update_timer.start())

        self.referneces = []
        self.lock_to_stimCheckBox = QCheckBox("lock")

        def set_lock(*args):
            self.lock_to_stim = self.lock_to_stimCheckBox.isChecked()
            self.update_ylim(self.state.stimno)

        self.lock_to_stimCheckBox.stateChanged.connect(set_lock)

        self.includeUnitCheckbox = QCheckBox("Tracked")
        self.includeUnitCheckbox.setChecked(True)
        self.includeUnitCheckbox.stateChanged.connect(lambda x: self.render())
        self.includeAllUnitsCheckBox = QCheckBox("Others")
        self.includeAllUnitsCheckBox.stateChanged.connect(lambda x: self.render())

        butgrp = QButtonGroup()

        linesRadio = QRadioButton("lines")
        heatmapRadio = QRadioButton("heatmap")
        unitOnlyRadio = QRadioButton("unit only")

        heatmapRadio.setChecked(True)
        butgrp.addButton(linesRadio)
        butgrp.addButton(heatmapRadio)
        butgrp.addButton(unitOnlyRadio)
        self.butgrp = butgrp

        def buttonToggled(id, checked):
            if heatmapRadio.isChecked():
                mode = "heatmap"
            elif linesRadio.isChecked():
                mode = "lines"
            elif unitOnlyRadio.isChecked():
                mode = "unitonly"
            if checked:
                self.mode = mode
                self.setup_figure()
            self.view.draw()

        butgrp.idToggled.connect(buttonToggled)

        self.polySelectButton = QPushButton("Selector")
        self.polySelectButton.clicked.connect(lambda *args: self.toggle_polySelector())

        self.settingsButton = QPushButton("Settings")
        self.settingsDialog = DialogSignalSelect()
        self.settingsButton.clicked.connect(lambda: self.settingsDialog.show())
        self.rightPlots = {}

        self.toggleMaxMeanButton = QPushButton("Show max", self)
        self.toggleMaxMeanButton.setCheckable(True)
        self.toggleMaxMeanButton.clicked.connect(lambda: self.toggleMaxMean())

        self.showMessages = QPushButton("Show messages", self)
        self.showMessages.setCheckable(True)
        self.showMessages.clicked.connect(lambda: self.toggleShowMessages())

        toolbar2 = QToolBar("Controls", self)
        toolbar2.addWidget(QLabel("Colour scale (std): "))
        toolbar2.addWidget(QLabel("Min"))
        toolbar2.addWidget(self.lowerSpinBox)
        toolbar2.addWidget(QLabel("Max"))
        toolbar2.addWidget(self.upperSpinBox)
        toolbar2.addSeparator()
        toolbar2.addWidget(QLabel("Units:"))
        toolbar2.addWidget(self.includeUnitCheckbox)
        toolbar2.addWidget(self.includeAllUnitsCheckBox)
        toolbar2.addSeparator()
        toolbar2.addWidget(QLabel("Mode:"))
        toolbar2.addWidget(self.lock_to_stimCheckBox)
        toolbar2.addWidget(linesRadio)
        toolbar2.addWidget(heatmapRadio)
        toolbar2.addWidget(unitOnlyRadio)
        toolbar2.addSeparator()

        toolbar2.addWidget(self.polySelectButton)
        toolbar2.addWidget(self.settingsButton)
        toolbar2.addWidget(self.toggleMaxMeanButton)
        toolbar2.addWidget(self.showMessages)

        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, toolbar2)
        # layout.addLayout(layout2)
        layout.addWidget(self.view)
        w = QWidget(self)
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.percentiles = []

        if state is not None:
            self.set_state(state)
        self.setup_figure()
        self.pg_selector = LineSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="purple", linestyle="none", linewidth=2, alpha=0.5),
            # useblit=True,
        )  # useblit does not work (for unknown reasons)
        self.pg_selector.set_active(False)
        self.update_axis()

        # @qsignal_throttle_wrapper(interval=33)
        # def draw_evt(evt):
        #     with self.fig.canvas.callbacks.blocked():
        #         # self.blit()
        #         self.render()

        # self.fig.canvas.mpl_connect("draw_event", draw_evt)
        # self.ax.callbacks.connect("xlim_changed", draw_evt)
        # self.ax.callbacks.connect("ylim_changed", draw_evt)

        self.dialogPolySelect = DialogPolySelect(self)
        self.dialogPolySelect.onSubmit.connect(self.polySelect)
        self.dialogPolySelect.finished.connect(
            lambda *args: self.toggle_polySelector(False)
        )
        self.dialogPolySelect.changeSelection.connect(
            lambda x, y: self.pg_selector.set_w(
                int(self.state.analog_signal.sampling_rate * x / 1000)
            )
        )

    def get_settings(self):
        return {
            "xlim": self.ax.get_xlim(),
            "yrange": np.diff(self.ax.get_ylim()),
            "mode": self.mode,
            "include_all_units": self.includeAllUnitsCheckBox.isChecked(),
            "lock_to_stim": self.lock_to_stim,
            "percentiles": (self.lowerSpinBox.value(), self.upperSpinBox.value()),
        }

    def set_settings(self, values):
        if "xlim" in values:
            self.ax.set_xlim(values["xlim"])
        if "mode" in values:
            self.mode = values["mode"]
        if "include_all_units" in values:
            self.includeAllUnitsCheckBox.setChecked(values["include_all_units"])
        if "lock_to_stim" in values:
            self.lock_to_stim = values["lock_to_stim"]
            self.lock_to_stimCheckBox.setChecked(values["lock_to_stim"])
        if "percentiles" in values:
            self.lowerSpinBox.setValue(values["percentiles"][0])
            self.upperSpinBox.setValue(values["percentiles"][1])
        if "yrange" in values:
            # cur= self.ax.get_ylim()[0]
            cur = max(self.state.stimno - np.abs(values["yrange"]) // 2, 0)
            self.ax.set_ylim(cur + np.abs(values["yrange"]), cur)

        self.setup_figure()

    def polySelect(self):
        """
        create new event based on the lineSelector
        """
        # find all the interpolated spikes

        window_size, min_peak = self.dialogPolySelect.getValues()
        xys = np.array(self.pg_selector._xys[:-1])
        window_size = int(self.state.analog_signal.sampling_rate * window_size / 1000)

        erp = self.state.get_erp()
        out = []

        for i in range(xys.shape[0] - 1):
            seg = np.array([xys[i], xys[i + 1]], dtype=int)
            seg = seg[np.argsort(seg[:, 1])]

            stimpos = np.arange(seg[0, 1], seg[1, 1])
            arr = np.interp(stimpos, seg[:, 1], seg[:, 0])
            for x, stimno in zip(arr, stimpos):
                x2 = int(x - (window_size // 2))
                vals = erp[stimno, x2 : x2 + window_size]
                x2_offset = np.argmax((1 if min_peak >= 0 else -1) * (vals))
                if (1 if min_peak >= 0 else -1) * vals[x2_offset] > abs(
                    min_peak
                ):  # if min_peak is negative, must be negative
                    out.append((x2 + x2_offset, stimno, vals[x2_offset]))

        out = np.array(out)
        evt = self.state.stimno_offset_to_event(
            out[:, 1].astype(int), out[:, 0].astype(int)
        )
        self.state.updateUnit(evt, merge=True)

    def toggle_polySelector(self, mode=None):
        self.pg_selector.set_active(mode or (not self.pg_selector.active))
        if self.pg_selector.active == 1:
            self.pg_selector.ax = self.ax
            if self.pg_selector._selection_artist not in self.ax.lines:
                self.ax.add_line(self.pg_selector._selection_artist)
            self.pg_selector.connect_default_events()
            self.pg_selector._selection_completed = False
            self.pg_selector.set_visible(True)
            self.dialogPolySelect.show()
            self.dialogPolySelect.change()
            # self.dialogPolySelect.activate()
        else:
            self.pg_selector._selection_completed = True
            self.pg_selector.clear()
            self.pg_selector._xys = [(0, 0)]  # TODO: move to clear
            self.dialogPolySelect.hide()
            self.pg_selector.set_visible(False)

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_P:
            self.toggle_polySelector()
        if (
            e.key() == Qt.Key_Return and self.pg_selector.active
        ):  # action when pg_selector is done. #TODO: move elsewhere

            self.polySelect()

        self.update()

    def closeEvent(self, *args):
        self.remove_references()
        self.dialogPolySelect.close()
        self.settingsDialog.close()
        self.pg_selector.set_active(False)

    def remove_references(self, *args, **kwargs):
        if self.state is not None:
            for ref in self.references:
                self.state.disconnect(ref)

    def set_state(self, state):
        self.state = state

        self.remove_references()

        if self.state is None:
            return

        self.references.append(
            self.state.onLoadNewFile.connect(self.reset_right_axes_data)
        )
        self.references.append(self.state.onLoadNewFile.connect(self.setup_figure))
        self.references.append(self.state.onLoadNewFile.connect(self.update_axis))

        self.references.append(
            self.state.onUnitGroupChange.connect(lambda *args: self.render())
        )
        self.references.append(
            self.state.onUnitChange.connect(lambda *args: self.render())
        )
        self.references.append(
            self.state.onUnitGroupChange.connect(
                lambda *args: self.reset_right_axes_data()
            )
        )

        self.references.append(self.state.onStimNoChange.connect(self.update_ylim))
        self.references.append(
            self.state.onStimNoChange.connect(
                lambda *args: self.plot_curstim_line(self.state.stimno)
            )
        )
        self.references.append(
            self.state.onUnitChange.connect(lambda x: self.reset_right_axes_data())
        )

        self.update_axis()
        # self.state.onUnitChange.connect(lambda x:self.reset_right_axes_data())
        self.reset_right_axes_data()

    # @qsignal_throttle_wrapper(1000)
    def reset_right_axes_data(self):
        if self.state is None:
            return
        if self.state.event_signal is None:
            return

        stimFreq_data = (
            1 / np.diff(self.state.event_signal.times).rescale(pq.second),
            np.arange(1, len(self.state.event_signal.times)),
        )
        ug = self.state.getUnitGroup()

        latencies = ug.get_latencies(self.state.event_signal)
        idx_na = np.isnan(latencies)

        current_spike_diffs = np.diff(latencies[~idx_na])
        out = np.ones(latencies.shape) * np.nan * pq.ms
        out[np.where(~idx_na)[0][1:]] = current_spike_diffs
        # out[np.isnan(out)] = 0 * pq.ms
        self.right_ax_data = {
            "Stim. Freq. (Hz)": stimFreq_data,
            "Latency Diff.": (out, np.arange(len(latencies))),
        }
        for x in self.state.segment.analogsignals:
            try:
                if self.toggleMaxMeanButton.isChecked():
                    self.right_ax_data[x.name] = (
                        np.max(self.state.get_erp(x, self.state.event_signal), axis=1),
                        np.arange(0, len(self.state.event_signal.times)),
                    )
                else:
                    self.right_ax_data[x.name] = (
                        np.mean(self.state.get_erp(x, self.state.event_signal), axis=1),
                        np.arange(0, len(self.state.event_signal.times)),
                    )

            except:
                pass

        if not self.rightPlots:
            self.rightPlots = {k: True for k, v in self.right_ax_data.items()}

        if self.settingsDialog is not None:
            self.settingsDialog.destroy()

        self.settingsDialog = DialogSignalSelect(options=self.rightPlots)

        def updateView(k, v):
            self.rightPlots[k] = v
            self.plot_right_axis()
            self.render()

        self.settingsDialog.changeSelection.connect(updateView)
        self.plot_right_axis()
        self.view.draw_idle()

    def setup_figure(self):
        mode = self.mode
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return

        # self.ax_right_fig.clear()
        erp = self.state.get_erp()
        # erp = np.clip(erp, 0, np.max(erp))
        self.std = np.std(erp)
        # self.percentiles = self.percentiles = np.linspace(0,10,101)*np.std((erp - np.mean(erp)))#np.percentile(erp, np.arange(100))#np.percentile(erp[erp>0], np.arange(101))
        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        if self.plotter is not None:
            try:
                self.plotter.ax_track_cmap.remove()
            except:
                pass
            try:
                self.plotter.ax_track_leaf.remove()
            except:
                pass
            try:
                self.plotter.hline.remove()
            except:
                pass

        self.plotter.setup(
            self.ax, erp, sampling_rate=self.state.sampling_rate, mode=mode
        )
        self.plotter.plot_main(
            mode=self.mode,
            ax=self.ax,
            erp=erp,
            partial=False,
            clim=(
                self.std * self.lowerSpinBox.value(),
                self.std * self.upperSpinBox.value(),
            ),
        )
        self.ax.set_ylim(ylim)
        self.ax.set_xlim(xlim)
        self.ax.set_autoscale_on(False)

        self.view.update()
        # self.blit()

        self.plotter.highlight_stim(self.ax, self.state.stimno)

        self.plot_spikegroups()
        # self.plot_curstim_line(self.state.stimno)

        self.plot_right_axis()

        self.fig.tight_layout()
        return [x for x in [self.ax_track_cmap, self.ax_track_leaf] if x is not None]
        # self.view.draw_idle()

    # @qsignal_throttle_wrapper(1000)
    def plot_right_axis(self):
        for ax in self.ax_right:
            ax.remove()
        self.ax_right = []
        count_axes = len([x for x, v in self.rightPlots.items() if v])
        # if count_axes == 0:
        #     self.ax.set_position([0,0,1,1])
        # TODO: when there are no plots, increase the width of the main plot to fill.

        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, max(count_axes, 1), subplot_spec=self.gs[0, 1]
        )
        # plot right axes
        colorwheel = itertools.cycle(iter(["r", "g", "b", "orange", "purple", "green"]))
        # self.fig.subfigures()
        for i, (label, data) in enumerate(
            [(k, v) for k, v in self.right_ax_data.items() if self.rightPlots[k]]
        ):

            self.ax_right.append(self.fig.add_subplot(gs00[0, i], sharey=self.ax))
            if i > 0:
                self.ax_right[i].tick_params(axis="y", labelleft=False)
            c = next(colorwheel)
            self.ax_right[i].plot(*data, label=label, color=c)
            self.ax_right[i].set_xlabel(label)
            self.ax_right[i].xaxis.label.set_color(c)
            self.ax_right[i].tick_params(axis="y", colors=c)
        for ax in self.ax_right:
            try:
                ax.draw(self.fig.canvas.get_renderer())
            except:
                pass
        # self.view.draw_idle()  # TODO - this slows things down as it re-renders the image plot also.

    def update_ylim(self, curStim):
        if self.lock_to_stim:
            cur_lims = self.ax.get_ylim()
            w = max(abs(cur_lims[1] - cur_lims[0]) // 2, 2)
            self.ax.set_ylim(curStim + w, curStim - w)
            # self.view.update()
            # self.view.draw_idle()
            self.view.draw()
            # self.view.update()

    def update_axis(self):  # TODO: plot these data using x as the time in millis.
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return
        self.ax.set_ylim(len(self.state.event_signal), 0)
        self.ax.set_xlim(0, len(self.state.analog_signal_erp[0]))

        class CustomFormatter(matplotlib.ticker.Formatter):
            def __init__(self, ax, func):
                super().__init__()
                self.set_axis(ax)
                self.func = func

            def __call__(self, x, pos=None):
                # Find the axis range
                vmin, vmax = self.axis.get_view_interval()
                major_locs = [
                    x
                    for x in self.axis.get_major_locator().tick_values(vmin, vmax)
                    if vmin <= x and x <= vmax
                ]
                if len(major_locs) >= 2:
                    return ""
                return self.func(
                    x, pos
                )  # "{0:g}".format(1000 * x / self.state.sampling_rate)

        func = lambda x, pos: "{0:g}".format(1000 * x / self.state.sampling_rate)
        func_formatter = matplotlib.ticker.FuncFormatter(func)

        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax, func))
        # loc = matplotlib.ticker.MaxNLocator(
        #     steps=[1, 5, 10],
        # )  # this locator puts ticks at regular intervals
        loc = matplotlib.ticker.MultipleLocator(self.state.sampling_rate / 10)
        self.ax.xaxis.set_major_locator(loc)
        loc = matplotlib.ticker.MultipleLocator(self.state.sampling_rate / 100)
        self.ax.xaxis.set_minor_locator(loc)

        self.ax.xaxis.set_major_formatter(func_formatter)

        # loc = matplotlib.ticker.MultipleLocator(
        #     base=self.state.sampling_rate / 100
        # )  # this locator puts ticks at regular intervals
        # self.ax.xaxis.set_major_locator(loc)
        # self.ax.set_xticks(
        #     np.arange(
        #         0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 100
        #     )
        # )
        # self.ax.set_xticks(
        #     np.arange(
        #         0, self.state.analog_signal_erp.shape[1], self.state.sampling_rate / 1000
        #     ),
        #     minor=True,
        # )
        # self.ax.grid(True, which="both")

    points_spikegroups = None

    def plot_spikegroups(self, sgidx=None):
        if self.points_spikegroups is None:
            pass  # TODO optimisation of setting x_data rather than replotting
        else:
            # try:
            for x in self.points_spikegroups:
                if x is not None:
                    x.remove()
            # except:
            #    pass
        points_spikegroups = []

        def plot(sgidx, **kwargs):
            sg = self.state.getUnitGroup(sgidx)

            points = np.array(
                [(x[0], i) for i, x in enumerate(sg.idx_arr) if x is not None]
            )
            if len(points) == 0:
                # self.view.draw_idle()
                return
            # scat = self.ax.scatter(points[:, 0], points[:, 1], s=10, **kwargs)
            points_spikegroups = []
            for point in points:
                rect = matplotlib.patches.Rectangle(
                    (point[0] - 5, point[1]), 10, 1, **kwargs
                )

                points_spikegroups.append(rect)

            # scat.set_animated(True)
            # self.ax.draw_artist(scat)
            points_spikegroups = PatchCollection(points_spikegroups, **kwargs)
            self.ax.add_collection(points_spikegroups)
            return points_spikegroups

        # include other units
        from matplotlib.cm import get_cmap
        from matplotlib.patches import Polygon
        import numpy as np

        points_spikegroups = []
        colors = get_cmap("Set2").colors
        if self.includeAllUnitsCheckBox.isChecked():
            for i, x in enumerate(self.state.spike_groups):
                if i == self.state.cur_spike_group:
                    continue
                artists = plot(i, color=colors[i % len(colors)])
                points_spikegroups.append(artists)

        # bg = self.
        if self.includeUnitCheckbox.isChecked():
            artists = plot(
                self.state.cur_spike_group,
                color="red",
                alpha=0.6 if self.mode == "heatmap" else 1,
            )
            points_spikegroups.append(artists)
        self.points_spikegroups = points_spikegroups

        return self.points_spikegroups

    # @qsignal_throttle_wrapper(interval=33)
    def render(self):

        self.view.draw_idle()

        # self.fig.canvas.restore_region(self.blit_data)
        # o = self.plot_curstim_line(self.state.stimno)

        o2 = self.plot_spikegroups()

        if self.pg_selector.active:
            self.pg_selector.set_visible(True)

            # self.pg_selector._draw_polygon()

            # self.pg_selector.draw()

        self.view.update()
        # return  o2
        # try:
        #     self.ax.redraw_in_frame()
        # except:
        #     self.fig.canvas.draw()
        # self.view.update()

    def plot_curstim_line(self, stimNo=None):
        self.plotter.highlight_stim(self.ax, stimNo, partial=False)
        self.view.draw_idle()

    @Slot()
    def updateAll(self):
        if self.mode != "heatmap" or self.plotter.ax_track_cmap is None:
            pass
        else:
            vmin = (
                self.std * self.lowerSpinBox.value()
            )  # self.percentiles[self.lowerSpinBox.value()]
            vmax = (
                self.std * self.upperSpinBox.value()
            )  # self.percentiles[self.upperSpinBox.value()]
            self.plotter.ax_track_cmap.set_clim(
                vmin,
                max(vmin + 0.001, vmax),
            )
            # self.ax_track_cmap.axes.draw_artist(self.ax_track_cmap)
            self.view.draw_idle()

    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1:
            return

        # if e.inaxes == self.ax:
        self.state.setStimNo(round(e.ydata))

    def toggleMaxMean(self):
        if self.toggleMaxMeanButton.isChecked():
            self.toggleMaxMeanButton.setText("Show Mean")
        else:
            self.toggleMaxMeanButton.setText("Show Max")
        self.reset_right_axes_data()

    def toggleShowMessages(self):
        i = 0
        i = next(
            i for i, e in enumerate(self.state.segment.events) if "log.other" == e.name
        )

        if self.showMessages.isChecked():
            self.showMessages.setText("Hide messages")

        else:
            self.showMessages.setText("Show messages")
            # self.plotter.highlight_stim(self.ax, stimNo, partial=False)

        self.toggle_plot_messages(self.state.segment.events[i])
        self.view.update()
        self.view.draw_idle()
        self.view.draw()

    def toggle_plot_messages(self, events):
        print("toggle_plot_messages")
        if self.messages is not None:
            for txt in self.messages:
                txt.remove()
            self.messages = None
        else:
            messages = []
            for i, t in enumerate(events):
                yy = self.state.event_signal.times.searchsorted(t, "right") - 1
                xx = 0
                ss = events.array_annotations["message"][i]
                messages.append(self.ax.text(xx, yy, ss))

            self.messages = messages

    def __del__(self):
        self.remove_references()
        self.close_conn.disconnect()
        super().__del__()


import matplotlib.patches


class LineSelector(PolygonSelector):
    # self.outerlines = None
    # def calculate_hits
    def __init__(self, *args, w=1000, **kwargs):
        self._selection_window = matplotlib.patches.Polygon(
            [(0, 0)], closed=False, edgecolor="purple", facecolor="none"
        )
        self.w = w

        super().__init__(*args, **kwargs)
        self.ax.add_patch(self._selection_window)

    def set_w(self, w):
        self.w = w
        self._draw_polygon()

    @property
    def artists(self):
        others = super().artists
        return others + (self._selection_window,)

    def set_visible(self, visible):
        return super().set_visible(visible)

    def clear(self):
        super().clear()
        self._selection_window.set_visible(False)
        self._selection_window.remove()
        self._selection_window = matplotlib.patches.Polygon(
            [(0, 0)], closed=False, edgecolor="purple", facecolor="none"
        )
        self.ax.add_patch(self._selection_window)

    def _draw_polygon_without_update(self):
        super()._draw_polygon_without_update()
        if len(self._xys) < 2:
            return

        # Create a polygon with width "w" around the line described by self._xys
        w = self.w  # You can adjust this width as needed
        xys = np.array(self._xys)
        if len(xys) < 2:
            return

        # #Calculate the perpendicular offsets for the polygon
        # dx = np.diff(xys[:, 0])
        # dy = np.diff(xys[:, 1])
        # length = np.sqrt(dx**2 + dy**2)
        # nx = -dy / length  # Normalized perpendicular vector (x-component)
        # ny = dx / length   # Normalized perpendicular vector (y-component)

        # Create the upper and lower edges of the polygon
        upper = xys + np.array([w / 2, 0])
        lower = xys - np.array([w / 2, 0])
        lower = lower[::-1]  # Reverse the lower edge to close the polygon

        # Combine the edges to form the polygon
        poly_points = np.vstack([upper, lower, upper[0]])
        if self._selection_window is None:
            self._selection_window = matplotlib.patches.Polygon(
                poly_points, closed=True, edgecolor="purple", alpha=0.5
            )

        self._selection_window.set_xy(poly_points)

    def _release(self, event):
        """Button release event handler."""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Place new vertex.
        elif (
            not self._selection_completed
            and "move_all" not in self._state
            and "move_vertex" not in self._state
        ):
            self._xys.insert(-1, (event.xdata, event.ydata))

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if not self._selection_completed and (
            event.key == self._state_modifier_keys.get("move_vertex")
            or event.key == self._state_modifier_keys.get("move_all")
        ):
            self._xys.append((event.xdata, event.ydata))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)


class DialogSignalSelect(QDialog):

    changeSelection = Signal(str, bool)

    def __init__(self, parent=None, options={}):
        super().__init__(parent, QtCore.Qt.Tool)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QVBoxLayout()
        self.cboxes = []
        for k, v in options.items():
            op = QCheckBox(text=k)
            op.setChecked(v)
            op.stateChanged.connect(lambda x, k=k: self.changeSelection.emit(k, x > 0))

            self.vbox.addWidget(op)
            self.cboxes.append(op)

        self.setLayout(self.vbox)


class DialogPolySelect(QDialog):
    changeSelection = Signal(float, float)
    onSubmit = Signal(float, float)

    def __init__(self, parent=None, options={}):
        super().__init__(parent, QtCore.Qt.Tool)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QFormLayout()
        self.cboxes = []

        self.window_size_input = QSpinBox(self)
        self.window_size_input.setMinimum(0)
        self.window_size_input.setMaximum(200)
        self.window_size_input.setValue(10)
        self.window_size_input.setSuffix("ms")
        self.window_size_input.setBaseSize(100, 10)
        self.window_size_input.valueChanged.connect(lambda *args: self.change())
        self.vbox.addRow(self.tr("window size"), self.window_size_input)

        self.minimumThreshold = QDoubleSpinBox(self)
        self.minimumThreshold.valueChanged.connect(lambda *args: self.change())
        self.minimumThreshold.setBaseSize(100, 10)
        self.minimumThreshold.setDecimals(2)
        self.minimumThreshold.setMinimum(-999999)
        self.minimumThreshold.setMaximum(999999)
        self.vbox.addRow(self.tr("minimumThreshold"), self.minimumThreshold)

        self.goButton = QPushButton("track")
        self.goButton.clicked.connect(lambda: self.onSubmit.emit(*self.getValues()))
        self.vbox.addRow(self.goButton)

        self.setLayout(self.vbox)

    def change(self):
        self.changeSelection.emit(
            self.window_size_input.value(), self.minimumThreshold.value()
        )

    def getValues(self):
        return (self.window_size_input.value(), self.minimumThreshold.value())


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    # view = DialogSignalSelect()
    state.loadFile(r"data/test2.h5")

    view = MultiTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()

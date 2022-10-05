import itertools
import sys
from dataclasses import dataclass, field

import matplotlib
import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_bases import MouseEvent
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import (FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from neo import Event
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QApplication, QCheckBox, QHBoxLayout,
                               QMainWindow, QSpinBox, QVBoxLayout, QWidget, QGroupBox, QRadioButton, QButtonGroup)

from .NeoSettingsView import NeoSettingsView
from .ViewerState import ViewerState, tracked_neuron_unit

mplstyle.use("fast")


class MultiTraceView(QMainWindow):

    right_ax_data = {}
    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):
        super().__init__(parent)
        self.state: ViewerState = None


        xsize = 1024
        ysize = 480
        dpi = 100

        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)

        self.mode = "heatmap"


        self.gs = self.fig.add_gridspec(1,2, width_ratios=[3,1])
        self.ax = self.fig.add_subplot(self.gs[0,0])
        self.ax_right = []
        #self.ax_right_fig = self.fig.add_subfigure(self.gs[0,1])
    
        # self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # create widgets
        self.view = FigureCanvas(self.fig)
        #self.view.update()
        # self.ps = PolygonSelectorTool(self.fig)
        # self.ps.enable()
        self.toolbar = NavigationToolbar2QT(self.view, self)
        # act = self.toolbar.addAction("selector")

        self.lock_to_stim = False

        self.addToolBar(self.toolbar)
        layout = QVBoxLayout()
        self.lowerSpinBox = QSpinBox(self)
        self.lowerSpinBox.setRange(1, 99)
        self.lowerSpinBox.valueChanged.connect(lambda x:self.updateAll())

        self.upperSpinBox = QSpinBox(self)
        self.upperSpinBox.setRange(1, 99)
        self.upperSpinBox.setValue(95)
        self.lowerSpinBox.valueChanged.connect(self.upperSpinBox.setMinimum)
        self.upperSpinBox.valueChanged.connect(self.lowerSpinBox.setMaximum)
        self.lowerSpinBox.setValue(45)
        self.upperSpinBox.valueChanged.connect(lambda x: self.updateAll())

        self.lock_to_stimCheckBox = QCheckBox("lock")

        def set_lock(*args):
            self.lock_to_stim = self.lock_to_stimCheckBox.isChecked()
            self.update_ylim(self.state.stimno)

        self.lock_to_stimCheckBox.stateChanged.connect(set_lock)

        self.includeAllUnitsCheckBox = QCheckBox("All units")
        self.includeAllUnitsCheckBox.stateChanged.connect(lambda x:self.plot_spikegroups())

        butgrp = QButtonGroup()

        linesRadio = QRadioButton("lines")
        heatmapRadio = QRadioButton("heatmap")
        unitOnlyRadio = QRadioButton("unit only")

        heatmapRadio.setChecked(True)
        butgrp.addButton(linesRadio)
        butgrp.addButton(heatmapRadio)
        butgrp.addButton(unitOnlyRadio)
        self.butgrp = butgrp
        
        def buttonToggled(id,checked):
            if heatmapRadio.isChecked():
                mode = "heatmap"
            elif linesRadio.isChecked():
                mode = "lines"
            elif unitOnlyRadio.isChecked():
                mode = "unitonly"
            if checked:
                self.mode = mode                
                self.setup_figure()

        butgrp.idToggled.connect(buttonToggled)
        
       
        layout2 = QHBoxLayout()
        layout2.addWidget(self.lowerSpinBox)
        layout2.addWidget(self.upperSpinBox)
        layout2.addWidget(self.includeAllUnitsCheckBox)
        layout2.addWidget(self.lock_to_stimCheckBox)
        layout2.addWidget(linesRadio)
        layout2.addWidget(heatmapRadio)
        layout2.addWidget(unitOnlyRadio)

        layout.addLayout(layout2)
        layout.addWidget(self.view)

        w = QWidget(self)
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.percentiles = []
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.points_spikegroup = None
        self.hline = None
        self.figcache=None
        if state is not None:
            self.set_state(state)
        self.setup_figure()
        self.update_axis()

    def set_state(self, state):
        self.state = state
        self.state.onLoadNewFile.connect(self.reset_right_axes_data)
        self.state.onLoadNewFile.connect(self.setup_figure)
        self.state.onUnitGroupChange.connect(self.plot_spikegroups)
        self.state.onUnitChange.connect(lambda x: self.plot_spikegroups())
        self.state.onStimNoChange.connect(self.plot_curstim_line)
        self.state.onLoadNewFile.connect(self.update_axis)
        self.state.onStimNoChange.connect(self.update_ylim)
        self.state.onUnitGroupChange.connect(lambda *args:self.reset_right_axes_data())
        self.state.onUnitChange.connect(lambda x:self.reset_right_axes_data())
        self.reset_right_axes_data()

    def reset_right_axes_data(self):
        if self.state is None:
            return
        if self.state.event_signal is None:
            return

        stimFreq_data = (1/np.diff(self.state.event_signal.times).rescale(pq.second),np.arange(1, len(self.state.event_signal.times)),)
        ug = self.state.getUnitGroup()
        
        latencies = ug.get_latencies(self.state.event_signal)
        idx_na = np.isnan(latencies)

        current_spike_diffs = np.diff(latencies[~idx_na])
        out = np.ones(latencies.shape) * np.nan *pq.ms
        out[np.where(~idx_na)[0][1:]] = current_spike_diffs
        #out[np.isnan(out)] = 0 * pq.ms
        self.right_ax_data = {'Stimulation Frequency':stimFreq_data, 'latency_diff':(out,np.arange(len(latencies)))}
        for x in self.state.segment.analogsignals:
            x.__hash__ = lambda x: hash(x.name)
            self.right_ax_data[x.name] = (np.mean(self.state.get_erp(x, self.state.event_signal),axis=1),np.arange(0, len(self.state.event_signal.times)))
        self.plot_right_axis()

    def setup_figure(self):
        mode = self.mode
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return

        #self.ax_right_fig.clear()
        self.percentiles = np.percentile(
            np.abs(self.state.get_erp()), np.arange(100)
        )
        if self.ax_track_leaf is not None:
            [x.remove() for x in self.ax_track_leaf]
            self.ax_track_leaf = None
        if self.ax_track_cmap is not None:
            self.ax_track_cmap.remove()
            self.ax_track_cmap = None
        
        if mode=="heatmap":

            self.ax_track_cmap = self.ax.imshow(
                np.abs(self.state.get_erp()),
                aspect="auto",
                cmap="gray_r",
                clim=(self.percentiles[40], self.percentiles[95]),
                interpolation="nearest",
            )
        elif(mode=="lines"):

            p90 = self.percentiles[95] * 4
            analog_signal_erp_norm = np.clip(
                self.state.get_erp(), -p90, p90
            ) / (p90 * 2)
            self.ax_track_leaf = self.ax.plot(
                (
                    (analog_signal_erp_norm * -1)
                    + np.arange(analog_signal_erp_norm.shape[0])[:, np.newaxis]
                ).T,
                color="gray",
                zorder=10,
            )
        else:
            pass
        if self.points_spikegroup is not None:
            self.points_spikegroup.remove()
            self.points_spikegroup = None
        #self.view.update()
        self.plot_spikegroups()
        self.plot_curstim_line(self.state.stimno)

        self.plot_right_axis()
        
        self.fig.tight_layout()

       
        self.view.draw_idle() 
        self.figcache = self.fig.canvas.copy_from_bbox(self.fig.bbox)
    
    def plot_right_axis(self):
        for ax in self.ax_right:
            ax.remove()
        self.ax_right = []
        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, len(self.right_ax_data.keys()), subplot_spec=self.gs[0,1])
        # plot right axes
        colorwheel = itertools.cycle(iter(["r","g","b","orange","purple","green"]))
        #self.fig.subfigures()
        for i, (label,data) in enumerate(self.right_ax_data.items()):
            self.ax_right.append(self.fig.add_subplot(gs00[0,i],sharey=self.ax))
            self.ax_right[i].set_yticks([])
            c = next(colorwheel)    
            self.ax_right[i].plot(*data, label=label, color=c)
            self.ax_right[i].set_xlabel(label)
            self.ax_right[i].xaxis.label.set_color(c)
            self.ax_right[i].tick_params(axis='y', colors=c)
        for ax in self.ax_right:
            try:
                ax.redraw_in_frame()
            except:
                pass
        self.view.update()

    def update_ylim(self, curStim):
        if self.lock_to_stim:
            cur_lims = self.ax.get_ylim()
            w = max(abs(cur_lims[1] - cur_lims[0]) // 2, 2)
            self.ax.set_ylim(curStim + w, curStim - w)
            self.view.draw_idle()

    def update_axis(self):
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return
        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x / self.state.sampling_rate)
        )
      
        self.ax.xaxis.set_major_formatter(func_formatter)
        loc = matplotlib.ticker.MultipleLocator(base=self.state.sampling_rate / 100) # this locator puts ticks at regular intervals
        self.ax.xaxis.set_major_locator(loc)
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
            for x in self.points_spikegroups:
                try:
                    x.remove()
                except:
                    pass
        self.points_spikegroups = []
        def plot(sgidx, **kwargs):
            sg = self.state.getUnitGroup(sgidx)
            
            points = np.array(
                [(x[0], i) for i, x in enumerate(sg.idx_arr) if x is not None]
            )
            if len(points) == 0:
                self.view.draw_idle()
                return
            return self.ax.scatter(
                points[:, 0], points[:, 1], s=4, **kwargs
            )
        # include other units
        if (self.includeAllUnitsCheckBox.isChecked()):
            for i,x in enumerate(self.state.spike_groups):
                if i == self.state.cur_spike_group:
                    continue
                artists = plot(i)
                self.points_spikegroups.append(artists)
                

        artists = plot(self.state.cur_spike_group, color="red")
        self.points_spikegroups.append(artists)
        try:
            self.ax.redraw_in_frame()
        except:
            pass
        self.view.draw_idle()

    def plot_curstim_line(self, stimNo=None):
        if stimNo is None:
            return

        if self.hline is None:
            self.hline = self.ax.axhline(stimNo)
        else:
            self.hline.set_ydata(stimNo)
         
        #self.ax.redraw_in_frame()
        self.view.draw_idle()
        #self.view.update()

    def updateAll(self):
        if self.mode!="heatmap":
            pass
        else:
            self.ax_track_cmap.set_clim(
                self.percentiles[self.lowerSpinBox.value()],
                self.percentiles[self.upperSpinBox.value()],
            )
            self.ax_track_cmap.axes.draw_artist(self.ax_track_cmap)
            #self.view.update()

    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1:
            return

        #if e.inaxes == self.ax:
        self.state.setStimNo(round(e.ydata))


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


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")

    view = MultiTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()

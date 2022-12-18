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
    QPushButton
)

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
        self.setFocusPolicy(Qt.ClickFocus)
        self.state: ViewerState = None
        self.ax_track_cmap = None
        self.ax_track_leaf = None
        self.points_spikegroup = None
        self.hline = None
        self.figcache = None

        xsize = 1024
        ysize = 480
        dpi = 80

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
        self.lowerSpinBox = QSpinBox(self)
        self.lowerSpinBox.setRange(1, 99)
        self.lowerSpinBox.valueChanged.connect(lambda x: self.updateAll())

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
        self.includeAllUnitsCheckBox.stateChanged.connect(
            lambda x: self.render()
        )

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

        butgrp.idToggled.connect(buttonToggled)

        self.settingsButton = QPushButton("settings")
        self.settingsDialog = DialogSignalSelect()
        self.settingsButton.clicked.connect(lambda : self.settingsDialog.show())
        self.rightPlots = {}




        layout2 = QHBoxLayout()
        layout2.addWidget(self.lowerSpinBox)
        layout2.addWidget(self.upperSpinBox)
        layout2.addWidget(self.includeAllUnitsCheckBox)
        layout2.addWidget(self.lock_to_stimCheckBox)
        layout2.addWidget(linesRadio)
        layout2.addWidget(heatmapRadio)
        layout2.addWidget(unitOnlyRadio)
        layout2.addWidget(self.settingsButton)

        layout.addLayout(layout2)
        layout.addWidget(self.view)

        w = QWidget(self)
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.percentiles = []

        if state is not None:
            self.set_state(state)
        self.setup_figure()
        # self.pg_selector =  PolygonSelector(self.ax, self.poly_selected)
        self.update_axis()
        self.blit()
        def draw_evt(evt):
            self.blit()
            self.render()
        self.fig.canvas.mpl_connect('draw_event',draw_evt)
        

    selected_poly = None

    def poly_selected(self, poly):
        self.selected_poly = poly
        p = matplotlib.path.Path(poly)
        extents = p.get_extents()
        # # create selection area for the erp
        # sample_rate = 3000 # possibly in ms

        # filt = np.zeros([(int(np.floor(extents.ymax))-int(np.floor(extents.ymin)))
        #     ,(int(np.floor(extents.xmax))-int(np.floor(extents.xmin)))*sample_rate] ) # TODO: smaple rate
        # i = np.indices(filt.shape).transpose((1,2,0)).reshape([-1,2]) + np.array((int(np.floor(extents.xmin)),int(np.floor(extents.ymin))))

        # p.contains_points(i)
        # a better view would be to get the path intersects for a horizontal line on each possible position
        for y in range(int(np.floor(extents.ymin), int(np.floor(extents.ymax)))):
            p2 = matplotlib.path.Path([(extents.xmin, y), (extents.xmax, y)])
            p.intersects_path(p2)  # only returns true.. not the coordinates

        print(poly)

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_P:
            self.pg_selector.set_active(~self.pg_selector.active)
            if self.pg_selector.active == 1:
                self.pg_selector.connect_default_events()
                self.pg_selector.set_visible(True)
            else:
                self.pg_selector.disconnect_events()
                self.pg_selector.set_visible(False)
        if e.key() == Qt.Key_Return and self.pg_selector.active:
            pass
            # add the new units
        self.update()

    def set_state(self, state):
        self.state = state
        self.state.onLoadNewFile.connect(self.reset_right_axes_data)
        self.state.onLoadNewFile.connect(self.setup_figure)
        self.state.onUnitGroupChange.connect(lambda *args: self.render())
        self.state.onUnitChange.connect(lambda *args: self.render())

        self.state.onStimNoChange.connect(self.update_ylim)
        self.state.onStimNoChange.connect(lambda *args: self.render())
        
        self.state.onLoadNewFile.connect(self.update_axis)
       
        self.state.onUnitGroupChange.connect(lambda *args: self.reset_right_axes_data())
        # self.state.onUnitChange.connect(lambda x:self.reset_right_axes_data())
        self.reset_right_axes_data()

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
            "Stimulation Frequency": stimFreq_data,
            "latency_diff": (out, np.arange(len(latencies))),
        }
        for x in self.state.segment.analogsignals:
            self.right_ax_data[x.name] = (
                np.mean(self.state.get_erp(x, self.state.event_signal), axis=1),
                np.arange(0, len(self.state.event_signal.times)),
            )
        
        self.rightPlots = {k:True for k,v in self.right_ax_data.items()}
        self.settingsDialog = DialogSignalSelect(options=self.rightPlots)
        def updateView(k,v):
            self.rightPlots[k] = v
            self.plot_right_axis()
        self.settingsDialog.changeSelection.connect(updateView)
        self.plot_right_axis()

    def setup_figure(self):
        mode = self.mode
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return

        # self.ax_right_fig.clear()
        self.percentiles = np.percentile(np.abs(self.state.get_erp()), np.arange(100))
        if self.ax_track_leaf is not None:
            [x.remove() for x in self.ax_track_leaf]
            self.ax_track_leaf = None
        if self.ax_track_cmap is not None:
            self.ax_track_cmap.remove()
            self.ax_track_cmap = None

        if mode == "heatmap":
            self.ax_track_cmap = self.ax.imshow(
                np.abs(self.state.get_erp()),
                aspect="auto",
                cmap="gray_r",
                clim=(self.percentiles[40], self.percentiles[95]),
                interpolation="antialiased",
            )
        elif mode == "lines":

            p90 = self.percentiles[95] * 4
            analog_signal_erp_norm = np.clip(self.state.get_erp(), -p90, p90) / (
                p90 * 2
            )
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
        self.view.update()
        self.blit()
        # self.plot_spikegroups()
        # self.plot_curstim_line(self.state.stimno)

        self.plot_right_axis()

        self.fig.tight_layout()
        return [x for x in [self.ax_track_cmap, self.ax_track_leaf] if x is not None]
        # self.view.draw_idle()

    def plot_right_axis(self):
        for ax in self.ax_right:
            ax.remove()
        self.ax_right = []
        count_axes = len([x for x,v in self.rightPlots.items() if v])
        # if count_axes == 0:
        #     self.ax.set_position([0,0,1,1])
        # TODO: when there are no plots, increase the width of the main plot to fill.

        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, max(count_axes,1) , subplot_spec=self.gs[0, 1]
        )
        # plot right axes
        colorwheel = itertools.cycle(iter(["r", "g", "b", "orange", "purple", "green"]))
        # self.fig.subfigures()
        for i, (label, data) in enumerate([(k,v) for k,v in self.right_ax_data.items() if self.rightPlots[k]]):
            
            self.ax_right.append(self.fig.add_subplot(gs00[0, i], sharey=self.ax))
            self.ax_right[i].set_yticks([])
            c = next(colorwheel)
            self.ax_right[i].plot(*data, label=label, color=c)
            self.ax_right[i].set_xlabel(label)
            self.ax_right[i].xaxis.label.set_color(c)
            self.ax_right[i].tick_params(axis="y", colors=c)
        # for ax in self.ax_right:
        #     try:
        #         ax.draw_idle()
        #     except:
        #         pass
        self.view.draw_idle() #TODO - this slows things down as it re-renders the image plot also.

    def update_ylim(self, curStim):
        if self.lock_to_stim:
            cur_lims = self.ax.get_ylim()
            w = max(abs(cur_lims[1] - cur_lims[0]) // 2, 2)
            self.ax.set_ylim(curStim + w, curStim - w)
            #self.view.update()
            self.view.draw()

    def update_axis(self): # TODO: plot these data using x as the time in millis.
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return
        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x / self.state.sampling_rate)
        )

        self.ax.xaxis.set_major_formatter(func_formatter)
        loc = matplotlib.ticker.MultipleLocator(
            base=self.state.sampling_rate / 100
        )  # this locator puts ticks at regular intervals
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

    @Slot()
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
                #self.view.draw_idle()
                return
            scat = self.ax.scatter(points[:, 0], points[:, 1], s=4, **kwargs)
            scat.set_animated(True)
            self.ax.draw_artist(scat)
            return scat

        # include other units
        from matplotlib.cm import get_cmap
        colors = get_cmap("Set2").colors
        if self.includeAllUnitsCheckBox.isChecked():
            for i, x in enumerate(self.state.spike_groups):
                if i == self.state.cur_spike_group:
                    continue
                artists = plot(i, color=colors[i%len(colors)])
                self.points_spikegroups.append(artists)
        #bg = self.
        artists = plot(self.state.cur_spike_group, color="red")
        self.points_spikegroups.append(artists)
        return self.points_spikegroups


    def blit(self):
        #self.fig.canvas.draw()
        self.blit_data = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    
    def render(self):
       
        #self.view.draw_idle()
        self.fig.canvas.restore_region(self.blit_data)
        o = self.plot_curstim_line(self.state.stimno)
        o2 = self.plot_spikegroups()
        self.view.update()
        return o + o2
        # try:
        #     self.ax.redraw_in_frame()
        # except:
        #     self.fig.canvas.draw()
        # self.view.update()

    def plot_curstim_line(self, stimNo=None):
        if stimNo is None:
            return

        if self.hline is None:
            self.hline = self.ax.axhline(stimNo)
            self.hline.set_animated(True)
        else:
            self.hline.set_ydata(stimNo)

        self.ax.draw_artist(self.hline)
        
        #self.view.draw_idle()
        #self.ax_track_cmap.draw()
        #self.view.draw_idle()
        #self.view.update()
        return [self.hline]


    @Slot()
    def updateAll(self):
        if self.mode != "heatmap" or self.ax_track_cmap is None:
            pass
        else:

            self.ax_track_cmap.set_clim(
                self.percentiles[self.lowerSpinBox.value()],
                self.percentiles[self.upperSpinBox.value()],
            )
            # self.ax_track_cmap.axes.draw_artist(self.ax_track_cmap)
            self.view.draw_idle()

    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1:
            return

        # if e.inaxes == self.ax:
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


class DialogSignalSelect(QDialog):

    changeSelection = Signal([str,bool])
    def __init__(self, parent=None, options={}):
        super().__init__(parent)
        self.initUI(options)

    def initUI(self, options):
        self.vbox = QVBoxLayout()
        self.cboxes=[]
        for k,v in options.items():
            op = QCheckBox(text=k)
            op.setChecked(v)
            op.stateChanged.connect(lambda x,k=k: self.changeSelection.emit(k, x>0))

            self.vbox.addWidget(op)
            self.cboxes.append(op)

        self.setLayout(self.vbox)

    pass


if __name__ == "__main__":
    
    app = QApplication([])
    state = ViewerState()
    #view = DialogSignalSelect()
    state.loadFile(r"data/test2.h5")

    view = MultiTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()

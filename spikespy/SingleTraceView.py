import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from scipy.signal import find_peaks
import matplotlib
import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal, Slot
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
)

from .ViewerState import ViewerState

mplstyle.use("fast")


class SingleTraceView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.state = state

        xsize = 1024
        ysize = 480
        dpi = 100
        self.closest_pos = 0  # the closest inflection to the current spike
        self.fig = Figure(figsize=(xsize / dpi, ysize / dpi), dpi=dpi)
        #  create widgets
        self.view = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar2QT(self.view, self)
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 20], hspace=0)

        self.ax = self.fig.add_subplot(self.gs[1, 0])
        self.topax = self.fig.add_subplot(self.gs[0, 0], sharex=self.ax)
        self.topax.yaxis.set_visible(False)
        self.topax.xaxis.set_visible(False)
        self.topax.spines["top"].set_visible(False)
        self.topax.spines["left"].set_visible(False)
        self.topax.spines["bottom"].set_visible(False)
        self.topax.spines["right"].set_visible(False)

        self.addToolBar(self.toolbar)

        self.identified_spike_line = self.ax.axvline(
            6000, zorder=0, visible=False, animated=True, alpha=0.3, color="blue"
        )
        self.trace_line_cache = None
        self.scatter_peaks = None

        self.setCentralWidget(self.view)

        self.state.onLoadNewFile.connect(self.setupFigure)
        self.state.onUnitChange.connect(self.updateFigure)
        self.state.onUnitGroupChange.connect(self.updateFigure)
        self.state.onStimNoChange.connect(self.updateFigure)

        self.fig.canvas.mpl_connect("button_press_event", self.view_clicked)
        # self.fig.canvas.mpl_connect('key_press_event', self.keyPressEvent)
        self.select_local_maxima_width = 1
        self.closest_pos = 0

        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

        self.blit_data = None

        def draw_evt(evt):
            self.blit()
            self.updateFigure()

        self.fig.canvas.mpl_connect("draw_event", draw_evt)
        self.setupFigure()

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
        self.ax.grid(True, which="both")

        if self.trace_line_cache is not None:
            self.trace_line_cache.remove()
            self.trace_line_cache = None

        self.fig.tight_layout()
        self.view.draw_idle()
        self.updateFigure()

    @Slot()
    def updateFigure(self):
        sg = self.state.getUnitGroup()
        dpts = self.state.get_erp()[self.state.stimno]  # this should only happen once.
        pts, _ = find_peaks(dpts)
        pts_down, _ = find_peaks(-1 * dpts)
        pts = np.sort(np.hstack([pts, pts_down]).flatten())
        cur_point = sg.idx_arr[self.state.stimno]
        if self.trace_line_cache is None:
            self.trace_line_cache = self.ax.plot(dpts, color="purple")[0]
            self.trace_line_cache.set_animated(True)
        else:
            self.trace_line_cache.set_data(np.arange(len(dpts)), dpts)

        self.topax.clear()
        values = [x[0] for x in sg.idx_arr if x is not None]
        idxs = [i for i, x in enumerate(sg.idx_arr) if x is not None]
        step = self.state.sampling_rate * 0.0005
        bins = np.arange(0, len(dpts), step)
        values_binned = np.histogram(values, bins=bins)

        self.topax.step(values_binned[1][1:], values_binned[0], color="gray")
        self.topax.set_ylim([1, max(values_binned[0]) + 1])
        if cur_point is not None:
            self.identified_spike_line.set_data(([cur_point[0], cur_point[0]], [0, 1]))
            self.identified_spike_line.set_visible(True)
            i = pts.searchsorted(cur_point[0])
            i2 = pts[i - 1 : i + 1]
            self.closest_pos = i2[np.argmin(np.abs(cur_point[0] - i2))]
            self.topax.axvline(
                cur_point[0],
                color="blue",
            )

        else:
            self.identified_spike_line.set_visible(False)
        cur_idx = np.searchsorted(
            idxs, self.state.stimno
        )  # plot the previous and next identified spike in this group
        if cur_idx > 0 and (idxs[cur_idx] - idxs[cur_idx - 1]) < 10:
            self.topax.axvline(values[cur_idx - 1], color="red", alpha=0.5)
        if cur_idx < len(idxs) and (idxs[cur_idx + 1] - idxs[cur_idx] - cur_idx) < 10:
            self.topax.axvline(values[cur_idx + 1], color="green", alpha=0.5)
        # if self.scatter_peaks is not None:
        #     self.scatter_peaks.remove()

        # self.scatter_peaks = self.ax.scatter(pts, dpts[pts], color="black", marker="x")

        # self.scatter_peaks2 = self.ax.scatter(pts_down, dpts[pts_down], color="black", marker="x")
        try:
            self.fig.canvas.restore_region(self.blit_data)
            self.ax.draw_artist(self.identified_spike_line)
            self.ax.draw_artist(self.trace_line_cache)

            self.topax.redraw_in_frame()
            # for x in self.topax.lines:
            #     self.topax.draw_artist(x)
            self.view.update()

        except:
            pass

    def blit(self):
        self.blit_data = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def set_cur_pos(self, x):
        x = round(x)
        dpts = self.state.get_erp()[self.state.stimno]
        if self.select_local_maxima_width > 1:
            w = self.select_local_maxima_width
            if x < 0 + w or x > self.state.analog_signal_erp.shape[1] - w:
                return

            x += np.argmax(np.abs(dpts[x - w : x + w])) - w

        self.state.setUnit(x)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_N:
            self.state.setUnit(self.closest_pos)
        elif e.key() == Qt.Key_Z:
            pass  # TODO: zoom into current spike


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")

    view = SingleTraceView(state=state)
    view.show()
    app.exec()
    sys.exit()

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
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    Signal,
    Slot,
    QThread,
    QThreadPool,
    QRunnable,
)
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

from .processing import create_erp_signals
from .ViewerState import ViewerState

mplstyle.use("fast")


class WorkerSignals(QObject):
    result = Signal(np.ndarray)
    cancel = Signal()


class updateConv(QRunnable):
    def __init__(self, analog_signal, sg, erp):
        self.signals = WorkerSignals()

        self.erp = erp
        self.sg = sg
        self.analog_signal = analog_signal
        self.cancelled = False

        self.signals.cancel.connect(self.cancel)

        super().__init__()

    def run(self):
        self.mean_erp = np.mean(
            create_erp_signals(
                self.analog_signal, self.sg.event, -0.01 * pq.s, +0.02 * pq.s
            ),
            axis=0,
        )
        conv = np.convolve(
            self.mean_erp / np.std(self.mean_erp),
            (self.erp / np.std(self.mean_erp)).flat[:],
            mode="same",
        ).reshape(self.erp.shape)
        conv[:, 0 : len(self.mean_erp)] = 0
        conv[:, -len(self.mean_erp) :] = 0

        # if not self.thread().isInterruptionRequested():
        self.signals.result.emit(conv)

    def cancel(self):
        self.cancelled = True


class updateHist(QRunnable):
    def __init__(self, analog_signal, sg, erp):
        self.signals = WorkerSignals()

        self.erp = erp
        self.sg = sg
        self.analog_signal = analog_signal
        self.cancelled = False

        self.signals.cancel.connect(self.cancel)

        super().__init__()

    def run(self):
        self.mean_erp = np.mean(
            create_erp_signals(
                self.analog_signal, self.sg.event, -0.01 * pq.s, +0.02 * pq.s
            ),
            axis=0,
        )
        conv = np.convolve(
            self.mean_erp,
            self.erp.flat[:],
            mode="same",
        ).reshape(self.erp.shape)
        conv[:, 0 : len(self.mean_erp)] = 0
        conv[:, -len(self.mean_erp) :] = 0

        # if not self.thread().isInterruptionRequested():
        self.signals.result.emit(conv)

    def cancel(self):
        self.cancelled = True


from matplotlib.ticker import Formatter

# Custom formatter class


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
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0)

        self.ax = self.fig.add_subplot(self.gs[1, 0])
        self.topax = self.fig.add_subplot(self.gs[0, 0], sharex=self.ax)
        self.topax.yaxis.set_visible(False)
        self.topax.xaxis.set_visible(False)
        self.topax.spines["top"].set_visible(False)
        self.topax.spines["left"].set_visible(False)
        self.topax.spines["bottom"].set_visible(False)
        self.topax.spines["right"].set_visible(False)
        self.topax.set_ylim(0, 1)

        self.fig.tight_layout()
        self.addToolBar(self.toolbar)

        self.identified_spike_line = self.ax.axvline(
            6000, zorder=0, visible=False, animated=True, alpha=0.3, color="blue"
        )
        self.trace_line_cache = None
        self.scatter_peaks = None

        self.setCentralWidget(self.view)

        self.state.onLoadNewFile.connect(self.setupFigure)

        self.state.onUnitChange.connect(self.updateHistogram)
        self.state.onUnitChange.connect(self.updateFigure)

        self.state.onUnitGroupChange.connect(self.updateHistogram)
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
        self.topax_lines = []

        self.task = None
        self.conv = None
        self.step =None
        self.setupFigure()

    @Slot()
    def view_clicked(self, e: MouseEvent):
        if self.toolbar.mode != "" or e.button != 1:
            return

        if e.inaxes == self.ax:
            self.set_cur_pos(e.xdata * self.state.sampling_rate)

    @Slot()
    def setupFigure(self):
        if self.state is None:
            return
        if self.state.analog_signal is None:
            return

        self.ax.grid(True, which="both")
        if self.trace_line_cache is not None:
            self.trace_line_cache.remove()
            self.trace_line_cache = None
        self.stimno_label = self.ax.text(0,1.01, "{stimno}", transform=self.ax.transAxes, animated=True)

        erp = self.state.get_erp()
        self.ax.set_xlim(0, erp.shape[-1] / self.state.sampling_rate)
        ylim = np.std(np.abs(erp)) * 8
        self.ax.set_ylim(-ylim, ylim)

        class CustomFormatter(Formatter):
            def __init__(self, ax: Any):
                super().__init__()
                self.set_axis(ax)

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
                return "{0:g}".format(1000 * x)
                # tl = [
                #     x
                #     for x in self.axis.get_minor_locator().tick_values(vmin, vmax)
                #     if vmin <= x and x <= vmax
                # ]

                # if x == tl[0] or x == tl[-1]:
                #     return "{0:g}".format(1000 * x)

                # return ""

        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x)
        )
        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax))
        # loc = matplotlib.ticker.MaxNLocator(
        #     steps=[1, 5, 10],
        # )  # this locator puts ticks at regular intervals
        loc = matplotlib.ticker.MultipleLocator(0.05)
        self.ax.xaxis.set_major_locator(loc)
        loc = matplotlib.ticker.MultipleLocator(0.01)
        self.ax.xaxis.set_minor_locator(loc)

        self.view.draw()
        self.view.update()
        self.toolbar.set_history_buttons()
        self.step = None
        self.updateHistogram()

    def get_settings(self):
        return {"xlim": self.ax.get_xlim()}

    def set_settings(self, values):
        if "xlim" in values:
            self.ax.set_xlim(values["xlim"])

    @Slot(np.ndarray)
    def updateThreadDone(self, x):
        self.conv = x
        self.updateFigure()

    def updateHistogram(self):
        sg = self.state.getUnitGroup()
        if self.step is not None:
            try:
                self.step[0].remove()
                self.step = None
            except:
                pass

        values = [x[0] for x in sg.idx_arr if x is not None]

        step = self.state.sampling_rate * 0.0005
        if len(values) == 0:
            return
        else:
            bins = np.arange(
                np.floor((min(values) // step)) * step,
                np.ceil(max(values) + step),
                step,
            )
        values_binned = np.histogram(values, bins=bins)

        if self.task is not None:
            self.task.signals.cancel.emit()
        # if self.updateThread.isRunning():
        #   self.updateThread.requestInterruption()

        self.task = updateConv(self.state.analog_signal, sg, self.state.get_erp())
        self.task.signals.result.connect(self.updateThreadDone)

        QThreadPool.globalInstance().start(self.task)

        self.step = self.topax.step(
            values_binned[1][1:] / self.state.sampling_rate,
            values_binned[0] / max(values_binned[0]),
            color="gray",
        )

        self.topax.redraw_in_frame()

        self.blit_data_topax = self.fig.canvas.copy_from_bbox(self.topax.bbox)

    @Slot()
    def updateFigure(self):
        sg = self.state.getUnitGroup()
        dpts = self.state.get_erp()[self.state.stimno]  # this should only happen once.
        cur_event_time = self.state.event_signal.times[self.state.stimno]
        hours, remainder = divmod(cur_event_time.magnitude, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.stimno_label.set_text(f"{self.state.stimno:04} [{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}]")

        pts, _ = find_peaks(dpts)
        pts_down, _ = find_peaks(-1 * dpts)
        pts = np.sort(np.hstack([pts, pts_down]).flatten())
        cur_point = sg.idx_arr[self.state.stimno]
        if self.trace_line_cache is None:
            self.trace_line_cache = self.ax.plot(
                np.arange(len(dpts)) / self.state.sampling_rate, dpts, color="purple"
            )[0]
            self.trace_line_cache.set_animated(True)
        else:
            self.trace_line_cache.set_data(
                np.arange(len(dpts)) / self.state.sampling_rate, dpts
            )

        idxs = [i for i, x in enumerate(sg.idx_arr) if x is not None]
        values = [x[0] / self.state.sampling_rate for x in sg.idx_arr if x is not None]
        for x in self.topax_lines:
            x.remove()
            del x
        self.topax_lines = []
        if cur_point is not None:
            self.identified_spike_line.set_data(
                (
                    [
                        cur_point[0] / self.state.sampling_rate,
                        cur_point[0] / self.state.sampling_rate,
                    ],
                    [0, 1],
                )
            )
            self.identified_spike_line.set_visible(True)
            i = pts.searchsorted(cur_point[0])
            i2 = pts[i - 1 : i + 1]
            self.closest_pos = i2[np.argmin(np.abs(cur_point[0] - i2))]
            self.topax_lines.append(
                self.topax.axvline(
                    cur_point[0] / self.state.sampling_rate, color="blue", animated=True
                )
            )

        else:
            self.identified_spike_line.set_visible(False)
        cur_idx = np.searchsorted(
            idxs, self.state.stimno
        )  # plot the previous and next identified spike in this group
        if cur_idx > 0 and (self.state.stimno - idxs[cur_idx - 1]) < 10:
            self.topax_lines.append(
                self.topax.axvline(
                    values[cur_idx - 1], color="green", alpha=0.5, animated=True
                )
            )

        if cur_point is None:
            cur_idx -= 1  # edge case where there is no unit
        if (
            (cur_idx + 1) < len(idxs)
            and (idxs[cur_idx + 1] - self.state.stimno) < 10
            and cur_idx >= 0
        ):
            self.topax_lines.append(
                self.topax.axvline(
                    values[cur_idx + 1], color="red", alpha=0.5, animated=True
                )
            )

        if self.conv is not None:
            conv = self.conv[self.state.stimno]
            conv_high = np.percentile(conv[1000:-1000], 99)
            highlight = find_peaks(conv, conv_high)
            hl = highlight[0][np.argsort(conv[highlight[0]])[::-1]]
            for x in hl:
                self.topax_lines.append(
                    self.topax.axvline(
                        x / self.state.sampling_rate,
                        color="gold",
                        ls="--",
                        animated=True,
                        alpha=0.3,
                    )
                )

        # if self.scatter_peaks is not None:
        #     self.scatter_peaks.remove()

        # self.scatter_peaks = self.ax.scatter(pts, dpts[pts], color="black", marker="x")

        # self.scatter_peaks2 = self.ax.scatter(pts_down, dpts[pts_down], color="black", marker="x")
        try:

            self.fig.canvas.restore_region(self.blit_data)
            self.fig.canvas.restore_region(self.blit_data_topax)

            self.ax.draw_artist(self.identified_spike_line)
            self.ax.draw_artist(self.trace_line_cache)
            self.ax.draw_artist(self.stimno_label)

            for x in self.topax_lines:
                self.topax.draw_artist(x)
            # for x in self.topax.lines:
            #     self.topax.draw_artist(x)
            self.view.update()

        except:
            pass

    def blit(self):
        self.blit_data = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.blit_data_topax = self.fig.canvas.copy_from_bbox(self.topax.bbox)

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

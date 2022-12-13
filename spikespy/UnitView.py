import sys
from typing import Any, List, Optional, Union

import matplotlib.style as mplstyle
import numpy as np
import PySide6
import quantities as pq
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
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
        # todo: move this to setup, update the current line. at the moment this causes slowdown of ui
        for i, idx, d in zip(range(len(sig_erp)), idx_arr, sig_erp):

            if idx is None:
                ydata = np.zeros(self.w * 2)
            else:
                ydata = d[max(idx[0] - self.w, 0) : min(idx[0] + self.w, len(d))]

            self.lines[i] = ax.plot(
                np.arange(-self.w, self.w), ydata, alpha=0.3, color="gray"
            )[
                0
            ]  # TODO: convert to lineCollection or blit it.
            if idx is None:
                self.lines[i].set_visible(False)

        self.update_curstim_line(self.state.stimno)
        self.view.draw_idle()

    @Slot()
    def update_curstim_line(self, stimno):

        idx = self.state.getUnitGroup().idx_arr[stimno]
        grayline = self.lines[stimno]
        if idx is None:
            grayline.set_visible(False)
            if stimno == self.state.stimno:
                self.selected_line.set_visible(False)
        else:

            d = self.state.analog_signal_erp[
                stimno
            ]  # TODO: this should not be required.
            yarr = d[idx[0] - self.w : idx[0] + self.w]

            grayline.set_visible(True)
            grayline.set_ydata(yarr)
            if stimno == self.state.stimno:
                self.selected_line.set_visible(True)
                self.selected_line.set_ydata(yarr)

            # self.axes["main"].relim()
            # self.axes["main"].autoscale_view(True, True, True)
        self.view.draw_idle()

    @Slot()
    def view_clicked(self, event):
        pass

    def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
        return self.parentWidget().keyPressEvent(event)

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
from .QMatplotlib import QMatplotlib

mplstyle.use("fast")


class UnitView(QMatplotlib):
    def __init__(
        self,
        parent: PySide6.QtWidgets.QWidget = None,
        state: ViewerState = None,
    ):
        
        self.axes = None
        self.w = None
        self.selected_line = None
        self.lines = None
        self.blit_data = None
        self.mean_line = None
        super().__init__(state=state, parent=parent, include_matplotlib_toolbar=True)

        def draw_evt(evt):
            self.blit()
            self.update_curstim_line(self.state.stimno)
            self.canvas.update()

        self.canvas.mpl_connect("draw_event", draw_evt)

    def blit(self):
        self.blit_data = self.canvas.copy_from_bbox(self.axes["main"].bbox)

    def setState(self, state):
        super().setState(state)
        self.state.onStimNoChange.connect(self.update_curstim_line)
        self.state.onUnitChange.connect(self.updateAll)
        self.state.onUnitGroupChange.connect(self.updateAll)
        self.state.onLoadNewFile.connect(self.updateAll)


    @Slot()
    def updateAll(self):
        self.figure.clear()
        self.setup_figure()

    def setup_figure(self):
        if self.state.analog_signal is None:
            return
        self.w = int(self.state.sampling_rate * 0.002)
        self.axes = {"main": self.figure.add_subplot(111)}
        ax = self.axes["main"]
        xarr = np.arange(-self.w, +self.w)
        self.selected_line = ax.plot(
            xarr, np.zeros(len(xarr)), color="purple", zorder=10, animated=True
        )[0]


        ax.axvline(0, ls="--", color="black")

        sg = self.state.getUnitGroup()
        sig_erp = self.state.analog_signal_erp
        idx_arr = sg.idx_arr
        self.lines = {}
        ylims = [-0.1, 0.1]
        mean_arr = np.zeros((len(xarr)))
        for i, idx, d in zip(range(len(sig_erp)), idx_arr, sig_erp):
            if idx is None:
                continue
            ydata = d[max(idx[0] - self.w, 0) : min(idx[0] + self.w, len(d))]
            mean_arr += ydata
            self.lines[i] = np.vstack((np.arange(-self.w, self.w), ydata)).T
            ylims[0] = min(ylims[0], np.min(ydata))
            ylims[1] = max(ylims[1], np.max(ydata))

        from matplotlib.collections import LineCollection

        lc = LineCollection(self.lines.values(), alpha=0.3, color="gray")
        self.mean_line = ax.plot(np.arange(-self.w, self.w), mean_arr/len(self.lines), ls="--", color="black", zorder=10)[0]
        ax.add_artist(lc)
        ax.set_ylim(ylims)

        self.canvas.draw_idle()

    @Slot()
    def update_curstim_line(self, stimno):
        idx = self.state.getUnitGroup().idx_arr[stimno]
        self.canvas.restore_region(self.blit_data)
        if stimno in self.lines:
            grayline = self.lines[stimno]
            if stimno == self.state.stimno:
                self.selected_line.set_visible(True)
                self.selected_line.set_ydata(grayline[:, 1])
        else:
            self.selected_line.set_visible(False)
        if idx is not None:
            self.selected_line.set_visible(True)
            self.axes["main"].draw_artist(self.selected_line)
        self.canvas.update()
    def draw_figure(self):
        pass#return super().draw_figure()

    def on_click(self, event):
        if event.inaxes != self.axes["main"]:
            return False
        if self.matplotlib_toolbar.mode != "":
            return False
        # find the closest line
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return False
        xdata = int(xdata)
        ydata = int(ydata)
        # find the closest line
        closest_line = None
        closest_dist = 1000000
        for i, line in self.lines.items():
            # Calculate the distance from the point to the line segments
            for j in range(len(line) - 1):
                p1, p2 = line[j], line[j + 1]
                # Vector math to calculate the distance from a point to a line segment
                line_vec = p2 - p1
                point_vec = np.array([xdata, ydata]) - p1
                line_len = np.dot(line_vec, line_vec)
                if line_len == 0:  # Avoid division by zero for degenerate line segments
                    dist = np.linalg.norm(point_vec)
                else:
                    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
                    projection = p1 + t * line_vec
                    dist = np.linalg.norm(np.array([xdata, ydata]) - projection)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_line = i
        if closest_line is not None:
            self.state.setStimNo(closest_line)

        return super().on_click(event)
    def keyPressEvent(self, event: PySide6.QtGui.QKeyEvent) -> None:
        return self.parentWidget().keyPressEvent(event)

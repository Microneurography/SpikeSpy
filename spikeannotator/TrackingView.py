import sys
from dataclasses import dataclass, field
from re import I
from typing import Any, List, Optional, Union

import matplotlib.style as mplstyle
import PySide6
import quantities as pq
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from PySide6.QtWidgets import (QApplication, QFormLayout, QHBoxLayout,
                               QMainWindow, QPushButton, QSpinBox, QStyle, QDoubleSpinBox,
                               QWidget)

from .basic_tracking import track_basic
from .ViewerState import ViewerState

mplstyle.use("fast")


class TrackingView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = None,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        self.state = state
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)


        qsb_window_size = QSpinBox(self)
        qsb_window_size.setMinimumWidth(100)
        qsb_window_size.setMaximum(100)
        qsb_window_size.setMinimum(10)
        self.qsb_window_size = qsb_window_size

        layout.addRow(self.tr("window size (ms)"), qsb_window_size)

        qsb_threshold = QDoubleSpinBox(self)
        qsb_threshold.setMinimumWidth(100)
        qsb_threshold.setMaximum(1000000)
        qsb_threshold.setMinimum(-1000000)

        self.qsb_threshold = qsb_threshold
        try:
            self.updateThresholdFromUnit()
        except:
            pass
        ql = QHBoxLayout()
        ql.addWidget(qsb_threshold)
        qbut_threshold_update = QPushButton(icon=self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        qbut_threshold_update.clicked.connect(self.updateThresholdFromUnit)
        ql.addWidget(qbut_threshold_update)

        layout.addRow(self.tr("Threshold (mV)"),ql)

        qsb_max_skip = QSpinBox(self)
        qsb_max_skip.setMaximum(10)
        qsb_max_skip.setMinimum(0)
        qsb_max_skip.setValue(1)
        self.qsb_max_skip = qsb_max_skip
        layout.addRow(self.tr("Maximum skips"), qsb_max_skip)
        
        #qbut_threshold_update.setFixedHeight(20)
        #layout.addRow(qbut_threshold_update)
        qbut_go = QPushButton("Track")
        layout.addRow(qbut_go)

        qbut_go.clicked.connect(self.trackUnit)

    def updateThresholdFromUnit(self):
        if self.state is None:
            return
        unit_events = self.state.getUnitGroup().event
        last_event = unit_events.searchsorted(
            self.state.event_signal[self.state.stimno] + (0.5 * pq.s)
        )  # find the most recent event

        starting_time = unit_events[max(last_event - 1, 0)]
        threshold = (
            self.state.analog_signal[
                self.state.analog_signal.time_index(starting_time)
            ][0]
            * 0.8
        ).rescale("mV")
        self.qsb_threshold.setValue(int(threshold))

    def trackUnit(self):
        unit_events = self.state.getUnitGroup().event
        last_event = unit_events.searchsorted(
            self.state.event_signal[self.state.stimno] + (0.5 * pq.s)
        )  # find the most recent event

        starting_time = unit_events[max(last_event - 1, 0)]
        window = self.qsb_window_size.value() * pq.ms
        threshold = self.qsb_threshold.value() * pq.mV

        evt2 = track_basic(
            self.state.analog_signal,
            self.state.event_signal,
            starting_time=starting_time,
            window=window,
            threshold=threshold,
            max_skip=self.qsb_max_skip.value()
        )

        self.state.updateUnit(
            event=unit_events.merge(evt2)
        )  # Perhaps create new units to merge later?


if __name__ == "__main__":
    app = QApplication([])
    view = TrackingView()
    view.show()
    app.exec()
    sys.exit()

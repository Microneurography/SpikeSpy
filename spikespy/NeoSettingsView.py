from typing import Any, List, Optional, Union

import matplotlib.style as mplstyle
import neo
import PySide6
from PySide6.QtWidgets import QComboBox, QFormLayout, QMainWindow, QWidget, QSpinBox
import quantities as pq
from .ViewerState import ViewerState

mplstyle.use("fast")


class NeoSettingsView(QMainWindow):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = ...,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        self.state = state
        self.state.onLoadNewFile.connect(
            lambda: self.populate_comboboxes(self.state.segment)
        )
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.analogSignalCombo = QComboBox(self)
        layout.addRow("Analog signal", self.analogSignalCombo)
        self.analogSignalCombo.currentIndexChanged.connect(
            lambda _: self.state.set_data(self.analogSignalCombo.currentData())
        )

        self.eventSignalCombo = QComboBox(self)

        layout.addRow("event channel", self.eventSignalCombo)
        self.eventSignalCombo.currentIndexChanged.connect(
            lambda _: self.state.set_data(
                event_signal=self.eventSignalCombo.currentData()
            )
        )
        self.populate_comboboxes(seg=self.state.segment)
        self.window_size = QSpinBox()

        self.window_size.setMinimum(500)
        self.window_size.setMaximum(2000)
        self.window_size.setValue(int(self.state.window_size.rescale(pq.ms)))
        self.window_size.setSuffix("ms")
        self.window_size.setSingleStep(100)
        self.window_size.valueChanged.connect(self.state.set_window_size)

        layout.addRow("window_size", self.window_size)

    def populate_comboboxes(self, seg: neo.Block):
        self.analogSignalCombo.blockSignals(True)
        self.eventSignalCombo.blockSignals(True)
        self.analogSignalCombo.clear()
        self.eventSignalCombo.clear()

        if seg is None:
            return
        for a in seg.analogsignals:
            self.analogSignalCombo.addItem(a.name, userData=a)
        for a in seg.events:
            self.eventSignalCombo.addItem(a.name, userData=a)
        self.analogSignalCombo.setCurrentIndex(
            self.analogSignalCombo.findData(self.state.analog_signal)
        )

        self.analogSignalCombo.blockSignals(False)
        self.eventSignalCombo.setCurrentIndex(
            self.eventSignalCombo.findData(self.state.event_signal)
        )
        self.eventSignalCombo.blockSignals(False)

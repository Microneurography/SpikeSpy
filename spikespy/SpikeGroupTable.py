from typing import Any, List, Optional, Union

import PySide6


from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor
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

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
    QStyledItemDelegate
)

from .ViewerState import ViewerState
from PySide6.QtWidgets import QComboBox


class SpikeGroupTableView(QWidget):
    def __init__(
        self,
        parent: Optional[PySide6.QtWidgets.QWidget] = ...,
        state: ViewerState = None,
    ) -> None:
        super().__init__(parent)
        self.spike_tablemodel = SpikeGroupTableModel(lambda: state.spike_groups)

        self.tbl = SpikeGroupTableView2(self)
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
        self.tbl.setItemDelegateForColumn(4, ComboBoxDelegate(self))

    @Slot()
    def set_selection(self, x):
        y = x.toList()
        if len(y) > 0:
            sg = x.first().top()
        else:
            sg = 0
        self.state.setUnitGroup(sg)


class SpikeGroupTableView2(QTableView):
    pass


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super(ComboBoxDelegate, self).__init__(parent)
        self.items = ["CMh", "CMi", "unknown"]  # Example items for the dropdown

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.setEditable(True)
        combo.addItems(self.items)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

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
            elif column == 3:  # notes
                return sg.event.annotations.get("notes")
            elif column == 4:  # fibre_type
                return sg.event.annotations.get("fibre_type")
            else:
                return ""
        elif role == Qt.BackgroundRole:
            return QColor(Qt.white)

    def setData(
        self,
        index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex],
        value: Any,
        role: int = ...,
    ) -> bool:
        if role == Qt.EditRole:
            column = index.column()
            row = index.row()
            sg = self.spikegroups_func()[row]
            if column == 3: # notes
                sg.event.annotations["notes"] = value
                self.dataChanged.emit(index, index)
                return True

            if column == 4:  # fibre class
                sg.event.annotations["fibre_type"] = value
                #sg.fibre_type = value
                self.dataChanged.emit(index, index)
                return True
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if index.column() in  [3,4]:  # fibre class
            return Qt.ItemIsEditable | Qt.ItemIsEnabled
        return Qt.ItemIsEnabled
    

if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"/Users/xs19785/Documents/Open Ephys/06-dec.h5")
    # evt = neo.Event(np.array([100]) * pq.s)
    # evt.array_annotate(info=["something"], number=[5])
    view = SpikeGroupTableView(None,state=state)
    # view = EventEdit(event=evt[0], annotations=evt.array_annotations_at_index(0))
    view.show()
    app.exec()

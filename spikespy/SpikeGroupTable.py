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
    QStyledItemDelegate,
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
        state.onUnitChange.connect(self.update)
        state.onUnitGroupChange.connect(self.update)
        state.onLoadNewFile.connect(self.update)
        
        self.tbl.setSelectionBehavior(QTableView.SelectRows)
        #self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setStyleSheet("QTableView::item:selected { background: #a0c4ff; }")

        self.addButton = QPushButton("+")
        self.addButton.clicked.connect(self.state.addUnitGroup)
        self.removeButton = QPushButton("-")
        self.removeButton.clicked.connect(self.remove_selected_row)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.tbl)
        vlayout.addWidget(self.addButton)
        vlayout.addWidget(self.removeButton)
        self.setLayout(vlayout)

        self.tbl.selectionModel().selectionChanged.connect(self.set_selection)
        self.tbl.setItemDelegateForColumn(4, ComboBoxDelegate(self, ["cm","ch","cmh","cmihi","cmih"]))

        self.addButton.clicked.disconnect()
        self.addButton.clicked.connect(self.add_row_and_select)
        self.spike_tablemodel.dataChanged.connect(self.state.onUnitGroupChange)
        self.update()

    def update(self):
        self.spike_tablemodel.update()
        # Ensure the correct row is selected as per state
        current_group = self.state.getUnitGroup() 
        if current_group is not None:
            # Find the row corresponding to the current unit group
            for row in range(self.tbl.model().rowCount()):
                sg = self.tbl.model().data(self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE")
                if sg == current_group:
                    self.tbl.selectionModel().blockSignals(True)
                    self.tbl.selectRow(row)
                    self.tbl.selectionModel().blockSignals(False)
                    break
        # self.tbl.selectionModel().disconnect(self.set_selection)
        # self.tbl.selectionModel().setCurrentIndex(
        #     ,
        #     QItemSelectionModel.SelectCurrent,
        # )

    @Slot()
    def set_selection(self, x):
        y = x.toList()
        if len(y) > 0:
            row = y[0].top()
            self.state.setUnitGroup(row)
        else:
            self.state.setUnitGroup(0)

    def remove_selected_row(self):
        selection = self.tbl.selectionModel().selectedRows()
        if selection:
            row = selection[0].row()
            sg = self.tbl.model().data(self.tbl.model().index(row, 0), role="SPIKEGROUP_ROLE")

            self.state.removeUnitGroup(sg)
     

    def add_row_and_select(self):
        self.state.addUnitGroup()
        # Select the last row after adding
        row_count = self.tbl.model().rowCount()
        if row_count > 0:
            self.tbl.selectRow(row_count - 1)


class SpikeGroupTableView2(QTableView):
    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and index.column() == 0:
            self.selectRow(index.row())
            # Optionally emit selectionChanged if you want to trigger logic
        else:
            # Ignore selection for other columns
            event.ignore()
        super().mousePressEvent(event)


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, items: List[str] = None):
        super(ComboBoxDelegate, self).__init__(parent)
        if items is None:
            items = ["Cm", "Cmh", "Cmih", "Cmihi", "Ch"]
        self.items = items

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        # Set the default text to match the current data
        combo.setEditable(True)
        combo.addItems(self.items)

        return combo


    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


class SpikeGroupTableModel(QAbstractTableModel):
    def __init__(self, spikegroups_func=None):
        """
        spikegroups_func is a functino which returns the spike groups
        """
        QAbstractTableModel.__init__(self,)
        self.spikegroups_func = spikegroups_func
        self.spikegroups = self.spikegroups_func()
        self.headers = ["SpikeID", "start", "end"]
        self.annotations_to_show = {'notes': "notes",  "fibre class":"fibre_type", "thermal":"thermal_response","mechanical":'mechanical_response'}
        self.headers += list(self.annotations_to_show.keys())

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
        if role == Qt.DisplayRole or role == Qt.EditRole:
            if column == 0:  # SpikeID
                return sg.event.name or row
            elif column in (1, 2):  # start/end
                if sg.get_window() is None:
                    return ""
                return sg.get_window()[column - 1]
            elif column >=3:
                return sg.event.annotations.get(self.annotations_to_show[self.headers[column]])

            # elif column == 3:  # notes
            #     return sg.event.annotations.get("notes")
            # elif column == 4:  # fibre_type
            #     return sg.event.annotations.get("fibre_type")
            else:
                return ""
        elif role == "SPIKEGROUP_ROLE":
            return sg
        #elif role == Qt.BackgroundRole:
            #return QColor(Qt.white)

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
            if column>=3:
                sg.event.annotations[self.annotations_to_show[self.headers[column]]] = value
                self.dataChanged.emit(index, index)
                
                return True

        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() >= 3:  # fibre class
            flags |= Qt.ItemIsEditable
        return flags




if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")
    # evt = neo.Event(np.array([100]) * pq.s)
    # evt.array_annotate(info=["something"], number=[5])
    view = SpikeGroupTableView(None, state=state)
    # view = EventEdit(event=evt[0], annotations=evt.array_annotations_at_index(0))
    view.show()
    app.exec()

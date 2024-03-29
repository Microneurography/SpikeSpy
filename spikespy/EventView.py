from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from PySide6.QtCore import QAbstractListModel, Qt, QAbstractTableModel

from neo import Event
from .ui.EventView import Ui_EventView
from .ViewerState import ViewerState
from typing import List
import sys
import numpy as np
import neo
import quantities as pq


class ListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.listData = []

    def setData(self, data):
        self.listData = data
        self.endResetModel()

    def rowCount(self, parent):
        return len(self.listData)


class EventViewModel(ListModel):
    def data(self, index, role):
        if role == Qt.DisplayRole:
            return f"{self.listData[index.row()]:.2f}"


class EventSelectorModel(ListModel):
    def data(self, index, role):
        if role == Qt.DisplayRole:
            ld: Event = self.listData[index.row()]
            return f"{ld.name}"


class EventViewTableModel(QAbstractTableModel):
    def __init__(self, parent=None, event: Event = None) -> None:
        super().__init__(parent)
        self.event = event or []
        self.keys = [] if event is None else event.array_annotations.keys

    def rowCount(self, parent):
        return len(self.event)

    def columnCount(self, parent):
        return len(self.keys) + 1

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section == 0:
                return "timestamp"
            return self.keys[section - 1]
        else:
            return f"{section}"

    def data(self, index, role):
        if role == Qt.DisplayRole:
            ld: Event = self.event[index.row()]
            if index.column() == 0:  # if the first index return the timestamp
                return f"{ld:.2f}"

            val = self.event.array_annotations[self.keys[index.column() - 1]][
                index.row()
            ]
            try:
                return f"{val:.2f}"
            except:
                return val

    def updateEvent(self, e: Event):

        self.event = e
        self.keys = [] if e is None else list(e.array_annotations.keys())
        self.endResetModel()

    def sort(self, Ncol, order):
        # TODO: implement sorting
        pass


class EventView(QtWidgets.QWidget):
    def __init__(self, parent=None, state: ViewerState = None):
        super().__init__(parent)
        self.ui = Ui_EventView()
        self.ui.setupUi(self)

        self.eventSelectorModel = EventSelectorModel()
        self.ui.comboBox.setModel(self.eventSelectorModel)
        self.ui.comboBox.currentIndexChanged.connect(self.onEventChange)
        # self.model = EventViewModel()
        # self.ui.eventListView.setModel(self.model)
        self.model = EventViewTableModel(self)
        self.ui.eventTableView.setModel(self.model)
        # self.ui.eventListView.selectionModel().selectionChanged.connect(self.on_listBox_change)
        self.ui.eventTableView.setSelectionBehavior(QtWidgets.QTableView.SelectRows)

        self.ui.goButton.clicked.connect(self.go_clicked)
        self.ui.deleteButton.clicked.connect(self.del_clicked)
        self.ui.addButton.clicked.connect(self.add_clicked)

        self.state: ViewerState = None
        self.setState(state)

        self.show()

    def setState(self, state: ViewerState):
        if state is None:
            return
        self.state = state

        self.updateModel()

        self.evtChangeConnection = self.state.onStimNoChange.connect(
            self.onStimNoChange
        )
        self.state.onLoadNewFile.connect(self.updateModel)
        self.state.onUnitChange.connect(lambda x: self.model.dataChanged())

    def updateModel(self):
        evts = self.state.segment.events
        if evts is not None:
            self.load_events(evts)

    def onStimNoChange(self):
        t = self.state.event_signal[self.state.stimno]
        i = self.model.event.searchsorted(t)
        self.ui.eventTableView.setCurrentIndex(self.model.index(i, 0))

    def onEventChange(self, index):
        e = self.eventSelectorModel.listData[index]
        self.model.updateEvent(e)

    def add_event(self, info):
        # TODO: update the event and annotations of selected event.
        pass

    def add_clicked(self):

        dialog = EventEdit(
            self,
            self.state.event_signal[self.state.stimno],
            {k: "" for k in self.model.event.array_annotations.keys()},
        )

        dialog.buttonBox.accepted.connect(lambda: self.add_event(dialog.annotations))
        dialog.setModal(True)
        dialog.show()

    def del_clicked(self):
        idxs = self.ui.eventTableView.selectedIndexes()
        self.state.updateUnit()
        pass

    def go_clicked(self):

        self.disconnect(self.evtChangeConnection)
        i = self.ui.eventTableView.selectedIndexes()[0].row()
        evt_idx = self.state.event_signal.searchsorted(self.model.event[i])
        self.state.setStimNo(evt_idx)
        self.evtChangeConnection = self.state.onStimNoChange.connect(
            self.onStimNoChange
        )

    def load_events(self, events: List[Event]):
        if len(events) > 0:
            self.model.updateEvent(events[0])
            self.eventSelectorModel.setData(events)

    def unit_selected(self):
        pass

    def on_listBox_change(self, val):
        row = val.indexes()[0].row()
        info = ""
        for k, v in self.model.listData.array_annotations.items():
            info += f"{k}: {v[row]}\n"
        nom = (
            self.model.listData.labels[row]
            if len(self.model.listData.labels) > 0
            else ""
        )
        self.ui.eventDetailView.setText(f"{row} - {nom}\n{info}")


class EventEdit(QtWidgets.QDialog):
    def __init__(self, parent=None, event: neo.Event = None, annotations: dict = None):
        super().__init__(parent)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.annotations = {**annotations, "timestamp": np.array(event.rescale(pq.s))}

        def on_change(s, k):
            self.annotations[k] = s

        layout = QtWidgets.QFormLayout()
        self.timestamp = QtWidgets.QDoubleSpinBox(self)
        self.timestamp.setMaximum(60 * 60 * 60)  # a 60 hour recording
        self.timestamp.setValue(np.array(event.rescale(pq.s)))
        self.timestamp.valueChanged.connect(lambda s: on_change(s, "timestamp"))
        layout.addRow("timestamp", self.timestamp)
        self.editValues = {}

        for k, v in annotations.items():
            self.editValues[k] = QtWidgets.QLineEdit(self)
            self.editValues[k].setText(str(v))
            self.editValues[k].textChanged.connect(lambda s, k=k: on_change(s, k))
            layout.addRow(k, self.editValues[k])

        self.saveButton = QtWidgets.QPushButton(text="save")

        widget2 = QtWidgets.QWidget()
        widget2.setLayout(layout)

        layout2 = QtWidgets.QVBoxLayout(self)
        layout2.addWidget(widget2)
        layout2.addWidget(self.buttonBox)
        self.setLayout(layout2)


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"/Users/xs19785/Documents/Open Ephys/06-dec.h5")
    # evt = neo.Event(np.array([100]) * pq.s)
    # evt.array_annotate(info=["something"], number=[5])
    view = EventView(state=state)
    # view = EventEdit(event=evt[0], annotations=evt.array_annotations_at_index(0))
    view.show()
    app.exec()
    sys.exit()

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication

from PySide6.QtCore import QAbstractListModel, Qt
from neo import Event
from .ui.EventView import Ui_EventView
from .ViewerState import ViewerState
from typing import List
import sys

class ListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.listData = []
        
    def setData(self, data):
        self.listData= data 
        self.endResetModel()

    def rowCount(self,parent):
        return len(self.listData)

class EventViewModel(ListModel):
    def data(self, index, role):
        if role == Qt.DisplayRole:
            return f"{self.listData[index.row()]:.2f}"

class EventSelectorModel(ListModel):
    def data(self, index, role):
        if role == Qt.DisplayRole:
            ld:Event = self.listData[index.row()]
            return f"{ld.name}"

class EventView(QtWidgets.QWidget):
    def __init__(self, parent=None, state:ViewerState=None):
        super().__init__(parent) 
        self.ui = Ui_EventView()
        self.ui.setupUi(self)

        self.model = EventViewModel()
        self.ui.eventListView.setModel(self.model)
        self.ui.eventListView.selectionModel().selectionChanged.connect(self.on_listBox_change)

        self.eventSelectorModel = EventSelectorModel()
        self.ui.comboBox.setModel(self.eventSelectorModel)
        self.ui.comboBox.currentIndexChanged.connect(self.onEventChange)

        self.ui.goButton.clicked.connect(self.go_clicked)
        
        self.state:ViewerState = None
        self.setState(state)


        self.show() 
    
    def setState(self, state:ViewerState):
        if state is None:
            return
        self.state = state

        evts = state.segment.events
        if evts is not None:
            self.load_events(evts)
        
        self.evtChangeConnection = self.state.onStimNoChange.connect(self.onStimNoChange)
        

    def onStimNoChange(self):
        t = self.state.event_signal[self.state.stimno]
        i = self.model.listData.searchsorted(t)
        self.ui.eventListView.setCurrentIndex(self.model.index(i,0))

    def onEventChange(self, index):
        e = self.eventSelectorModel.listData[index]
        self.model.setData(e)

    def add_clicked(self):
        pass

    def del_clicked(self):
        pass

    def go_clicked(self):
        
        self.disconnect(self.evtChangeConnection)
        i = self.ui.eventListView.selectedIndexes()[0].row()
        evt_idx = self.state.event_signal.searchsorted(self.model.listData[i])
        self.state.setStimNo(evt_idx)
        self.evtChangeConnection = self.state.onStimNoChange.connect(self.onStimNoChange)

    def load_events(self,events:List[Event]):
        if len(events) > 0:
            self.model.setData(events[0])
            self.eventSelectorModel.setData(events)


    def unit_selected(self):
        pass
        

    def on_listBox_change(self, val):
        row = val.indexes()[0].row()
        info = ""
        for k,v in self.model.listData.array_annotations.items():
            info += f"{k}: {v[row]}\n"
        nom = self.model.listData.labels[row] if len(self.model.listData.labels)>0 else ""
        self.ui.eventDetailView.setText(f"{row} - {nom}\n{info}")
        


if __name__ == "__main__":

    app = QApplication([])
    state = ViewerState()
    state.loadFile(r"data/test2.h5")

    view = EventView(state=state)
    view.show()
    app.exec()
    sys.exit()
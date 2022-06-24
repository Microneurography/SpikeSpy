import sys
from typing import Union

import neo
import PySide6
from neo.io import NixIO
from PySide6.QtCore import (QAbstractItemModel, QAbstractTableModel, QDir,
                            QModelIndex, Qt, Slot)
from PySide6.QtGui import QAction, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QApplication, QComboBox, QStyledItemDelegate,
                               QTreeView, QVBoxLayout, QWidget, QHBoxLayout, QPushButton)
import numpy as np

from spikeannotator.APTrack_experiment_import import TypeID, find_square_pulse_numpy


class ComboboxDelegate(QStyledItemDelegate):
    def __init__(self, parent=None,options=[]):
        super().__init__(parent)
        self.options = options

    def createEditor(self, parent: PySide6.QtWidgets.QWidget, option: PySide6.QtWidgets.QStyleOptionViewItem, index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex]) -> PySide6.QtWidgets.QWidget:
        editor = QComboBox(parent)
        editor.setFrame(False)
        for o in self.options:
            editor.addItem(o)
        return editor

    def setEditorData(self, editor: QComboBox, index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex]) -> None:
        val = index.model().data(index, Qt.EditRole)
        for x in range(editor.count()):
            if val == editor.itemData(x, Qt.EditRole):
                break
        editor.setCurrentIndex(x)

    
    def setModelData(self, editor: QComboBox, model: QAbstractItemModel, index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex]) -> None:
        model.setData(index, editor.itemData(editor.currentIndex(),Qt.EditRole), Qt.EditRole)
    
    def updateEditorGeometry(self, editor: PySide6.QtWidgets.QWidget, option: PySide6.QtWidgets.QStyleOptionViewItem, index: Union[PySide6.QtCore.QModelIndex, PySide6.QtCore.QPersistentModelIndex]) -> None:
        editor.setGeometry(option.rect)

class QNeoSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.treeView = QTreeView(self)
        self.model = QStandardItemModel(self)
        self.treeView.setModel(self.model)
        # self.treeView.setEditTriggers(QAbstractItemView.AllEditTriggers)
        #self.treeView.clicked.connect(self.treeView.edit)

        t = QWidget()
        hboxlayout = QHBoxLayout(t)
        hboxlayout.addWidget(self.treeView)

        self.verticalLayout = QVBoxLayout()

        self.setLayout(self.verticalLayout)
        t2 = QWidget()
        central_vlayout = QVBoxLayout(t2)
        add_button = QPushButton("+")
        self.output = neo.Segment()

        def add_button_clicked(process=None):
            sel = self.get_selection()[0]
            if process == "TTL":
                    d = sel.as_array()
                    idxs = find_square_pulse_numpy(
                            d,
                            sel.sampling_rate * 0.0004,
                            (2 * np.std(d)) + np.mean(d)
                    )  
                    idxs_rising = idxs[
                        0
                    ] 
                    sel = neo.Event(
                    (idxs_rising / sel.sampling_rate).rescale("s"),
                    type_id=TypeID.TTL.value,
                    name=sel.name,
                    array_annotations={
                        "duration": (
                            np.array(idxs[1] - idxs[0] - 1) / sel.sampling_rate
                        ).rescale("s"),
                        "maximum": (idxs[2] * sel.units).rescale("mV"),
                    },
                    )   
            if isinstance(sel, neo.AnalogSignal):
                
                self.output.analogsignals.append(sel)
            elif isinstance(sel, neo.Event):
                self.output.events.append(sel)
            
        add_button.clicked.connect(add_button_clicked)

        add_ttl_button = QPushButton("->TTL")
        add_ttl_button.clicked.connect(lambda: add_button_clicked("TTL"))

        minus_button = QPushButton("-")
        central_vlayout.addWidget(add_button)
        central_vlayout.addWidget(add_ttl_button)
        central_vlayout.addWidget(minus_button)
        
        hboxlayout.addWidget(t2)
        
        self.selectedTreeView = QTreeView(self)
        self.selectedTreeModel = QStandardItemModel(self)
        self.selectedTreeView.setModel(self.selectedTreeModel)
        hboxlayout.addWidget(self.selectedTreeView)

        
        self.verticalLayout.addWidget(t)
        but = QPushButton("done")
        self.verticalLayout.addWidget(but)

        self.map = {}
        # __qtreewidgetitem = QTreeWidgetItem(self.treeWidget)
       
        # __qtreewidgetitem1 = QTreeWidgetItem(__qtreewidgetitem)

        # treeHeader.setText(0,"type")
        # treeHeader.setText(1,"item")
        # __qtreewidgetitem.setText(1, "item") 
        # __qtreewidgetitem.setText(0, "type")

        # __qtreewidgetitem1.setText(0, "test") 
        # __qtreewidgetitem1.setText(1, "test2")
        
        #c.setText(1, "surprise!")
    
    def load_neo(self, data:neo.Block, model=None):
        if model is None:
            model = self.model
        rt = model.invisibleRootItem()
        rt.setColumnCount(1)
        for i,s in enumerate(data.segments):

            si = QStandardItem(f"{i}-{s.name}")
            si_analogs = QStandardItem("analogs")
            si.appendRow(si_analogs)
            for i, a in enumerate(s.analogsignals):
                si2 = QStandardItem(f"{i}-{a.name}")
                
                uid = f"analog-{i}"
                si2.setData(uid)
                self.map[uid] = a
                si_analogs.appendRow([si2])
            
            si_events = QStandardItem("events")
            for i, a in enumerate(s.events):
                si2 = QStandardItem(f"{i}-{a.name}")
                
                uid = f"events-{i}"
                si2.setData(uid)
                self.map[uid] = a
                si_events.appendRow([si2]) 
            si.appendRow(si_events)

            rt.appendRow(si)
    
    def get_selection(self):
        rownos = [self.model.itemFromIndex(x).data() for x in self.treeView.selectedIndexes()]
        return [self.map[r] for r in rownos if r is not None]


    
if __name__ == "__main__":

    app = QApplication([])

    # model = QFileSystemModel()
    # model.setRootPath(QDir.currentPath())
    # tree = QTreeView()

    # tree.setModel(model)

    # tree.setRootIndex(model.index(QDir.currentPath()))
    # tree.show()
    data = neo.NixIO("data/test2.h5",mode="ro").read_block()
    view = QNeoSelector()
    view.load_neo(data)
    view.show()
    app.exec()
    sys.exit()

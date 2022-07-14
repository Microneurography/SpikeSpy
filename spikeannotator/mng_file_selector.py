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
from spikeannotator.ViewerState import ViewerState


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
    def __init__(self, parent=None, state:ViewerState=None):
        super().__init__(parent)
        self.state = state
        if self.state is not None:
            self.state.onLoadNewFile.connect(self.reset)
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
        self.outblock = neo.Block()

        self.output = neo.Segment()
        self.outblock.segments.append(self.output)
        def add_button_clicked(process=None):
            sel = self.get_selection()[0]
            if process == "TTL":
                    d = sel.as_array()
                    idxs = find_square_pulse_numpy(
                            d,
                            sel.sampling_rate * 0.0004,
                            (2 * np.std(d)) + np.mean(d) # TODO: parameterise this
                    )  
                    idxs_rising = idxs[
                        0
                    ] 
                    sel = neo.Event(
                    (idxs_rising / sel.sampling_rate).rescale("s") + sel.t_start,
                    type_id=TypeID.TTL.value,
                    name=sel.name,
                    array_annotations={
                        "duration": (
                            np.array(idxs[1] - idxs[0] - 1) / sel.sampling_rate
                        ).rescale("s"),
                        "maximum": (idxs[2] * sel.units).rescale("mV"),
                    },
                    )   
            if process == "wavelet":
                from .processing import wavelet_denoise
                import copy
                d = sel.as_array() 
                wd = wavelet_denoise(d[:,0])
                sel = sel.copy()
                sel[:,0] = (wd * sel.units)[:,np.newaxis]

                sel.name = (sel.name or "") + ".wavelet"
                del d
                del wd

            if isinstance(sel, neo.AnalogSignal):
                
                self.output.analogsignals.append(sel)
            elif isinstance(sel, neo.Event):
                self.output.events.append(sel)
            
            self.load_neo(self.outblock,self.selectedTreeModel)
        add_button.clicked.connect(add_button_clicked)

        add_ttl_button = QPushButton("->TTL")
        add_ttl_button.clicked.connect(lambda: add_button_clicked("TTL"))
        add_wlet_button = QPushButton("+ wavelet")
        add_wlet_button.clicked.connect(lambda: add_button_clicked("wavelet"))


        minus_button = QPushButton("-")

        def minus_button_clicked():
            sel = [self.selectedTreeModel.itemFromIndex(x).data() for x in self.selectedTreeView.selectedIndexes()]
            for s in sel:
                self.output.analogsignals = [x for x in self.output.analogsignals if id(x) != id(s)]
                self.output.events = [x for x in self.output.events if id(x)!=id(s)]
            
            self.load_neo(self.outblock,self.selectedTreeModel)
        minus_button.clicked.connect(minus_button_clicked)
        central_vlayout.addWidget(add_button)
        central_vlayout.addWidget(add_ttl_button)
        central_vlayout.addWidget(add_wlet_button)
        central_vlayout.addWidget(minus_button)
        
        hboxlayout.addWidget(t2)
        
        self.selectedTreeView = QTreeView(self)
        self.selectedTreeModel = QStandardItemModel(self)
        self.selectedTreeView.setModel(self.selectedTreeModel)
        hboxlayout.addWidget(self.selectedTreeView)

        
        self.verticalLayout.addWidget(t)
        but = QPushButton("done")

        def done_clicked():
            self.state.set_segment(self.output)

        but.clicked.connect(done_clicked)
        self.verticalLayout.addWidget(but)

        self.map = {}
        self.reset()
        # __qtreewidgetitem = QTreeWidgetItem(self.treeWidget)
       
        # __qtreewidgetitem1 = QTreeWidgetItem(__qtreewidgetitem)

        # treeHeader.setText(0,"type")
        # treeHeader.setText(1,"item")
        # __qtreewidgetitem.setText(1, "item") 
        # __qtreewidgetitem.setText(0, "type")

        # __qtreewidgetitem1.setText(0, "test") 
        # __qtreewidgetitem1.setText(1, "test2")
        
        #c.setText(1, "surprise!")
    def reset(self):
        self.output
        blk = neo.Block()
        from copy import deepcopy
        seg = deepcopy(self.state.segment)
        for i,sg in enumerate(self.state.spike_groups):
            sg.event.name=f"unit_{i}"
            seg.events.append(sg.event)
        blk.segments.append(seg)
        self.load_neo(blk)

    def load_neo(self, data:neo.Block, model=None, clear=True):
        if model is None:
            model = self.model
        if clear:
            model.clear()
        rt = model.invisibleRootItem()
        rt.setColumnCount(1)
        for i,s in enumerate(data.segments):

            si = QStandardItem(f"{i}-{s.name}")
            si_analogs = QStandardItem("analogs")
            si.appendRow(si_analogs)
            for i2, a in enumerate(s.analogsignals):
                si2 = QStandardItem(f"{i2}-{a.name}")
                si2.setData(a)
                si_analogs.appendRow([si2])
            
            si_events = QStandardItem("events")
            si_units = QStandardItem("units")
            for i2, a in enumerate(s.events):
                si2 = QStandardItem(f"{i}-{a.name}")
                
                si2.setData(a)
                if a.name.startswith("unit"):
                    si_units.appendRow([si2])
                else:
                    si_events.appendRow([si2]) 
                
            si.appendRow(si_events)
            si.appendRow(si_units)
            rt.appendRow(si)
    
    def get_selection(self):
        rownos = [self.model.itemFromIndex(x).data() for x in self.treeView.selectedIndexes()]
        return [r for r in rownos if r is not None]


    
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

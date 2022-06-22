import sys
from typing import Union

import neo
import PySide6
from neo.io import NixIO
from PySide6.QtCore import (QAbstractItemModel, QAbstractTableModel, QDir,
                            QModelIndex, Qt, Slot)
from PySide6.QtGui import QAction, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QApplication, QComboBox, QStyledItemDelegate,
                               QTreeView, QVBoxLayout, QWidget)


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

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.addWidget(self.treeView)
        self.setLayout(self.verticalLayout)
        delegate = ComboboxDelegate(options=['signal','events'])
        self.treeView.setItemDelegateForColumn(1,delegate)
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
    
    def load_neo(self, data:neo.Block):
        rt = self.model.invisibleRootItem()
        rt.setColumnCount(1)
        for i,s in enumerate(data.segments):

            si = QStandardItem(f"{i}-{s.name}")
            si_analogs = QStandardItem("analogs")
            si.appendRow(si_analogs)
            for i, a in enumerate(s.analogsignals):
                si2 = QStandardItem(f"{i}-{a.name}")
                si_analogs.appendRow([si2,QStandardItem()])
                
                self.map[si2.index().row()] = a
            
            si_events = QStandardItem("events")
            for i, a in enumerate(s.events):
                si2 = QStandardItem(f"{i}-{a.name}")
                si_events.appendRow([si2,QStandardItem()]) 
                self.map[si2.index().row()] = a
            si.appendRow(si_events)

            rt.appendRow(si)
    
    def get_selection(self):
        rownos = [x.row() for x in self.treeView.selectedIndexes()]
        return [self.map[r] for r in rownos]

class NeoTreeModel(QAbstractItemModel):
    pass

    
if __name__ == "__main__":

    app = QApplication([])

    # model = QFileSystemModel()
    # model.setRootPath(QDir.currentPath())
    # tree = QTreeView()

    # tree.setModel(model)

    # tree.setRootIndex(model.index(QDir.currentPath()))
    # tree.show()
    view = QNeoSelector()
    view.load_neo(data)
    view.show()
    app.exec()
    print(view.get_selection())
    sys.exit()

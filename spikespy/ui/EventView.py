# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'EventViewxlTlhU.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class Ui_EventView(object):
    def setupUi(self, EventView):
        if not EventView.objectName():
            EventView.setObjectName("EventView")
        EventView.resize(331, 260)
        self.verticalLayout = QVBoxLayout(EventView)
        self.verticalLayout.setObjectName("verticalLayout")
        self.comboBox = QComboBox(EventView)
        self.comboBox.setObjectName("comboBox")

        self.verticalLayout.addWidget(self.comboBox)

        self.splitter = QSplitter(EventView)
        self.splitter.setObjectName("splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.splitter.setHandleWidth(7)
        self.eventTableView = QTableView(self.splitter)
        self.eventTableView.setObjectName("eventTableView")
        self.splitter.addWidget(self.eventTableView)

        self.verticalLayout.addWidget(self.splitter)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.addButton = QPushButton(EventView)
        self.addButton.setObjectName("addButton")

        self.horizontalLayout.addWidget(self.addButton)

        self.deleteButton = QPushButton(EventView)
        self.deleteButton.setObjectName("deleteButton")

        self.horizontalLayout.addWidget(self.deleteButton)

        self.horizontalSpacer = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum
        )

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.goButton = QPushButton(EventView)
        self.goButton.setObjectName("goButton")

        self.horizontalLayout.addWidget(self.goButton)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(EventView)

        QMetaObject.connectSlotsByName(EventView)

    # setupUi

    def retranslateUi(self, EventView):
        EventView.setWindowTitle(
            QCoreApplication.translate("EventView", "Events", None)
        )
        self.addButton.setText(QCoreApplication.translate("EventView", "Add", None))
        self.deleteButton.setText(
            QCoreApplication.translate("EventView", "Delete", None)
        )
        self.goButton.setText(QCoreApplication.translate("EventView", "Go", None))

    # retranslateUi

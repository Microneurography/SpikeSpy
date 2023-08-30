from spikespy.ViewerState import APTrackDialog
from PySide6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = APTrackDialog(None)
    dialog.exec()
    if dialog.accepted:
        print(dialog.get_config())

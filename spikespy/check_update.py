import importlib.metadata
import requests
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel,QScrollArea, QFrame
from PySide6.QtCore import Qt

def check_github_releases(repo= "Microneurography/SpikeSpy", current_version=None):
    """
    Check the GitHub repository for new releases and get the release notes.

    Parameters:
    repo (str): The GitHub repository in the format 'owner/repo'.
    current_version (str): The current version of the software.

    Returns:
    tuple: A tuple containing the latest version and release notes if there is a new release, otherwise None.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url)
    if current_version is None:
        import importlib

        current_version = importlib.metadata.version("spikespy")
    
    if response.status_code == 200:
        latest_release = response.json()
        latest_version = latest_release['tag_name']
        release_notes = latest_release['body']
        
        if latest_version != current_version and latest_version!=("v"+current_version):
            return latest_version, release_notes
        else:
            return None
    else:
        response.raise_for_status()

class ReleaseNotesDialog(QDialog):
    def __init__(self, version, release_notes):
        super().__init__()
        self.setWindowTitle("New release available")
        layout = QVBoxLayout()
        version_label = QLabel(f"Version: {version}")
        release_notes_label = QLabel(release_notes)


        scroll_area = QScrollArea()
        scroll_area.setWidget(release_notes_label)
        layout.addWidget(version_label)
        layout.addWidget(scroll_area)
        
        
        self.setLayout(layout)

class VersionCheckWorker(QObject):
    finished = Signal()
    result = Signal(object)

    def __init__(self, repo="Microneurography/SpikeSpy", current_version=None):
        super().__init__()
        self.repo = repo
        self.current_version = current_version

    @Slot()
    def run(self):
        info = check_github_releases(self.repo, self.current_version)
        self.result.emit(info)


class VersionCheckThread(QThread):
    def __init__(self, repo="Microneurography/SpikeSpy", current_version=None):
        super().__init__()
        self.worker = VersionCheckWorker(repo, current_version)
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)
        self.worker.finished.connect(self.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.finished.connect(self.deleteLater)
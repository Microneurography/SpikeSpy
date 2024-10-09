import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from spikespy.QMatplotlib import QMatplotlib
import matplotlib
from matplotlib.ticker import Formatter
from matplotlib.collections import LineCollection
from spikespy.ViewerState import ViewerState
import matplotlib.style as mplstyle
mplstyle.use("fast")
class MultiTraceFixedView(QMatplotlib):
    settings = {
            "n_lines_pre": 5,
            "n_lines_post": 5,
            #"scale": None,
        }
    def __init__(self, parent=None,state: ViewerState=None):

        super().__init__(parent=parent,state=state,include_matplotlib_toolbar=True)
        
    def setState(self, state: ViewerState):
        super().setState(state)
        self.state.onStimNoChange.connect(lambda x:self.update_figure())
        self.set_scale_setting()
    
    def set_scale_setting(self):
        if self.get_settings().get("scale") is None:
            erp = self.state.get_erp()
            ylim = np.std(np.abs(erp)) * 8
            self.settings["scale"] = 1/ylim  
    def onLoadNewFile(self):
        super().onLoadNewFile()
        
        

    def get_settings(self):
        return self.settings 
    
    def setup_figure(self):
        n_lines = self.get_settings().get("n_lines_pre", 10)
        n_lines_post = self.get_settings().get("n_lines_post", 10)

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        # Create a LineCollection
        self.lines = LineCollection([], colors="grey", zorder=1)
        self.ax.add_collection(self.lines)
        self.points = self.ax.scatter([], [], color="red")


        func_formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(1000 * x)
        )
        self.ax.xaxis.set_major_formatter(func_formatter)
        self.ax.xaxis.set_minor_formatter(CustomFormatter(self.ax))
        self.ax.set_ylim(n_lines + n_lines_post + 1 ,-1)
        self.ax.set_xlim(0, 0.5)

    def draw_figure(self):
        if self.state is None:
            return False
        n_lines_pre = self.get_settings().get("n_lines_pre", 10)
        n_lines_post = self.get_settings().get("n_lines_post", 10)
        scale = self.get_settings().get("scale", 1)
        stimno = self.state.stimno
        erp = self.state.get_erp()
        start = max(stimno-n_lines_pre,0)
        stop =min(start+n_lines_pre+n_lines_post+1, len(erp))
        dpts= erp[start:stop]
        
        segments = []
        colors = []
        for i in range(len(dpts)):
            x = np.arange(erp.shape[-1]) / self.state.sampling_rate
            y = (dpts[i] * scale) + i
            segments.append(np.column_stack([x, y]))
            if i == stimno - start:
                colors.append("purple")
            else:
                colors.append("grey")
        x_ys = []
        for i,l in enumerate(self.state.spike_groups[self.state.cur_spike_group].get_latencies(self.state.event_signal[start:stop])):
            if np.isnan(l):
                continue
            l= l.rescale("s")
            sample_no = int(l*self.state.sampling_rate)
            if sample_no>=len(dpts[i]) or sample_no<0:
                continue
            r = 20
            x = np.arange(sample_no-r, sample_no+r)
            y = (dpts[i][x]*scale)+i
            segments.append(np.column_stack([x/self.state.sampling_rate, y]))
            colors.append("red")
            x_ys.append((float(l.magnitude),(dpts[i][sample_no]*scale)+i))
        if self.points is not None:
            self.points.remove()
            self.points = None
        
        #print(x_ys)
       

        self.lines.set_segments(segments)
        self.lines.set_color(colors)
        if len(x_ys)>0:
           self.points=self.ax.scatter(*zip(*x_ys),s=10,color="black",zorder=10)
     
        self.ax.set_yticks(np.arange(0,len(dpts)),np.arange(start,stop))


class CustomFormatter(Formatter):
    def __init__(self, ax):
        super().__init__()
        self.set_axis(ax)

    def __call__(self, x, pos=None):
        # Find the axis range
        vmin, vmax = self.axis.get_view_interval()
        major_locs = [
            x
            for x in self.axis.get_major_locator().tick_values(vmin, vmax)
            if vmin <= x and x <= vmax
        ]
        if len(major_locs) >= 2:
            return ""
        return "{0:g}".format(1000 * x)




import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer

# Assuming MultiLinePlot and ViewerState are defined in the same module


class TestMultiLinePlot(QMainWindow):
    def __init__(self, state:ViewerState, rate=5000):
        super().__init__()
        self.viewer_state = state
        self.plot = MultiLinePlot(self.viewer_state)
        self.setCentralWidget(self.plot)
        self.stimno = 200

        # Timer to update stimno every 5 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stimno)
        self.timer.start(rate)  # 5000 ms = 5 seconds
        self.direction = 1

    def update_stimno(self):
        print("next stimno")
        self.stimno += self.direction
        if self.stimno==0:
            self.direction=1
        if self.stimno == len(self.viewer_state.event_signal):
            self.direction=-1

        self.viewer_state.setStimNo(self.stimno)

if __name__ == "__main__":
    state = ViewerState()
    # view = DialogSignalSelect()
    state.loadFile(r"data/test2.h5")
    app = QApplication(sys.argv)
    test_window = TestMultiLinePlot(state, rate=10)
    test_window.show()
    sys.exit(app.exec())
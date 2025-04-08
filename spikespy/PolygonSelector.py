from matplotlib.widgets import PolygonSelector
from matplotlib.lines import Line2D


class PolygonSelectorTool(PolygonSelector):
    def __init__(self, ax, onselect, **kwargs):
        super().__init__(ax, onselect, **kwargs)
        self._xys = []

    def _release(self, event):
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1
        elif not self._selection_completed:
            self._xys.insert(-1, (event.xdata, event.ydata))

    def _on_key_release(self, event):
        if event.key == self._state_modifier_keys.get("clear"):
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)


class LineSelector(PolygonSelector):
    def _draw_polygon(self):
        super()._draw_polygon()

    def _release(self, event):
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1
        elif (
            not self._selection_completed
            and "move_all" not in self._state
            and "move_vertex" not in self._state
        ):
            self._xys.insert(-1, (event.xdata, event.ydata))

    def _on_key_release(self, event):
        if not self._selection_completed and (
            event.key == self._state_modifier_keys.get("move_vertex")
            or event.key == self._state_modifier_keys.get("move_all")
        ):
            self._xys.append((event.xdata, event.ydata))
            self._draw_polygon()
        elif event.key == self._state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)

from .ViewerState import ViewerState
import numpy as np
from neo import Event

class UnitSuggestor():
    update_on_unit_change = True
    update_on_unit_group_change = True
    def __init__(self, state=None,settings=None):
        self._suggestions = None
        self.settings = None
        self.state:ViewerState = None

        self.setState(state)
        self.set_settings(settings)
    
    def setState(self, state):
        self.state:ViewerState= state
        if self.update_on_unit_change:
            self.state.onUnitChange.connect(self.updateSuggestions)
        self.state.onUnitGroupChange.connect(self.updateSuggestions)
        self.state.onLoadNewFile.connect(self.updateSuggestions)

    # TODO: run in a thread
    def updateSuggestions(self):
        self._suggestions = self.suggest()

    def get_predictions(self):
        return self._suggestions

    def settings_dialog(self):
        pass
    
    def set_settings(self,settings):
        self.settings=settings
    def get_settings(self):
        return {**self.settings}
    
    def suggest(self):
        pass

class threshold_suggestor(UnitSuggestor):
    update_on_unit_change = False
    update_on_unit_group_change = False
    def get_settings(self):
        settings = super().get_settings()
        settings['threshold'] = None

    def suggest(self):
        settings = self.get_settings()
        asig = self.state.analog_signal
        
        if settings['threshold'] is None:
            threshold = None
            threshold = np.std(asig) * settings.get('zscore_threhsold',5)
        
        threshold_crossings = asig>threshold
        threshold_crossings = np.where(threshold_crossings)[0]
        threshold_corssings = asig.t_start + (threshold_crossings/asig.sampling_rate)
        suggestions = Event(threshold_corssings)
        return suggestions
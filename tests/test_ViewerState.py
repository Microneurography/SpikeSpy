import pytest
from spikespy.ViewerState import ViewerState


@pytest.fixture()
def state():
    state = ViewerState()
    state.loadFile(r"data/test2.h5")
    return state


def test_undo(state: ViewerState):
    state.setStimNo(1)
    state.setUnit(300)
    unit_group = state.getUnitGroup()
    cur_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency == 300
    state.setUnit(100)
    unit_group = state.getUnitGroup()
    new_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency != new_latency
    assert new_latency == 100

    state.undo()
    unit_group = state.getUnitGroup()
    undo_latency = unit_group.idx_arr[state.stimno][0]
    assert cur_latency == undo_latency

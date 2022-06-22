
import logging

import neo
import numpy as np
import quantities as pq

logger = logging.getLogger("Tracking")

def track_basic(raw_data:neo.AnalogSignal, stimulus_events:neo.Event, starting_time:pq.s, threshold=0.1 * pq.mV, window=0.01*pq.s, max_skip=1):
    """
    basic tracking as per APTrack plugin

    given an identified time & threshold, look at track below within a given window for a threshold crossing, repeat until end
    """

    start_idx = stimulus_events.as_array().searchsorted(starting_time)
    
    offset = starting_time - stimulus_events[start_idx-1]

    traced_times = []
    skipped = 0

    def time_to_index(ts):
        return int(((ts - raw_data.t_start) * raw_data.sampling_rate).base)

    
    for x in stimulus_events[start_idx:]:
        t_start = time_to_index(x + offset - (window/2))
        t_end = time_to_index(x + offset + (window/2))

        max_idx = np.argmax(
            raw_data[t_start:t_end] * (-1 if threshold<0 else 1)
        )
        max_val = raw_data[t_start + max_idx]
        if (max_val >= threshold and threshold >= 0) or (max_val<=threshold and threshold <0):
            skipped = 0
            max_ts = ((max_idx + t_start) / raw_data.sampling_rate) + raw_data.t_start
            traced_times.append(max_ts.rescale("s"))
            offset = max_ts-x
        else:
            skipped+=1
            logger.info("max < threshold")
        if skipped > max_skip:
            logger.info("Stopping tracking as max skips reached")
            break
    
    return neo.Event(np.array(traced_times)*pq.s)

# TODO: use peak finding first then windowing. this will (hopefully) prevent runaways


def test_track_basic():
    fname = r"test_data/test_annotated.h5"
    data = neo.NixIO(fname, "ro").read_block().segments[0]
    stimulus_events = data.events[0]

    raw_data = data.analogsignals[0]
    starting_time = data.events[1][10] # a tracked unit
    window = 0.02 * pq.s
    threshold =  raw_data[raw_data.time_index(starting_time)][0] * 0.8 #0.1 * pq.mV

    evt2 = track_basic(raw_data, stimulus_events, starting_time=starting_time, window=window,threshold=threshold)

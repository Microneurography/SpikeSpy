from neo import Segment, Block, AnalogSignal
from pathlib import Path
import re
import quantities as pq

from .APTrack_experiment_import import readContinous, readHeader


def open_ephys_to_neo(foldername):
    """
    neo's openephys io is very slow and loads ALL channels into memory.. as floats.
    This will create a block from an openephys folder

    """
    all_files = list(Path(foldername).glob("*_2.continuous"))
    all_files.sort()

    seg = Segment()

    for f in all_files:

        header, ch_mmap = readContinous(str(f))
        sampling_rate = float(header["sampleRate"]) * pq.Hz
        ch_units_probe = pq.UnitQuantity(
            "kmicroVolts", pq.uV / float(header["bitVolts"]), symbol="kuV"
        )
        asig = AnalogSignal(
            ch_mmap["data"].flat,
            sampling_rate=sampling_rate,
            units=ch_units_probe,
            name=f.stem,
        )
        seg.analogsignals.append(asig)
    return seg

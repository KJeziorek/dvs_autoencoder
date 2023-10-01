import numpy as np
import math

# from Dsec-Det: https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/io.py

class EventReader:
    def _extract_from_h5_by_index(self, filehandle, ev_start_idx: int, ev_end_idx: int):
        events = filehandle['events']
        x, y, t, p = events['x'], events['y'], events['t'], events['p']

        x_new = x[ev_start_idx:ev_end_idx]
        y_new = y[ev_start_idx:ev_end_idx]
        p_new = p[ev_start_idx:ev_end_idx]
        t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

        output = {'p': p_new, 't': t_new, 'x': x_new, 'y': y_new,}
        return output

    def extract_index(self, h5file, ev_start_idx: int, ev_end_idx: int):
        return self._extract_from_h5_by_index(h5file, ev_start_idx, ev_end_idx)
        
    def extract_timewindow(self, h5file, t_min_us: int, t_max_us: int):
        ms2idx = np.asarray(h5file['ms_to_idx'], dtype='int64')
        t_offset = h5file['t_offset'][()]

        events = h5file['events']
        t = events['t']

        t_ev_start_us = t_min_us - t_offset
        assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
        t_ev_start_ms = t_ev_start_us // 1000
        ms2idx_start_idx = int(t_ev_start_ms)
        ev_start_idx = ms2idx[ms2idx_start_idx]

        t_ev_end_us = t_max_us - t_offset
        assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
        t_ev_end_ms = math.floor(t_ev_end_us / 1000)
        ms2idx_end_idx = t_ev_end_ms
        ev_end_idx = ms2idx[ms2idx_end_idx]

        return self._extract_from_h5_by_index(h5file, ev_start_idx, ev_end_idx)
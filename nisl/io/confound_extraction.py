import numpy as np
from scipy import signal, stats, linalg

from .. import utils

def high_variance_confounds(niimgs):
    """ Return confounds time series extracted from voxels with high
        variance.

        Notes
        ======
        This method is related to what has been published in the
        literature as 'CompCorr' (Behzadi NeuroImage 2007).
    """
    # XXX: I should not use apply_mask to load series
    niimgs = utils.check_niimg(niimgs)
    series = niimgs.get_data().T
    del niimgs
    series = np.reshape(series, (len(series), -1))
    for serie in series.T:
        serie[:] = signal.detrend(serie)
    # Retrieve the 1% high variance voxels
    var = np.mean(series ** 2, axis=0)
    var_thr = stats.scoreatpercentile(var, 99)
    series = series[var > var_thr]
    # XXX: should I use a randomized SVD?
    u, s, confounds = linalg.svd(series, full_matrices=False)
    confounds = confounds[:10]
    return confounds



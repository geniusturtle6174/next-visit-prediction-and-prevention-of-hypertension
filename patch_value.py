import numpy as np


def height_weight_bmi(height, weight, bmi, verbose=False):
    if height <= 0 and weight > 0 and bmi > 0:
        if verbose:
            print('height_weight_bmi: height patched')
        return 100 * (weight / bmi) ** 0.5, weight, bmi
    if height > 0 and weight <= 0 and bmi > 0:
        if verbose:
            print('height_weight_bmi: weight patched')
        return height, bmi * ((height / 100) ** 2), bmi
    if height > 0 and weight > 0 and bmi <= 0:
        if verbose:
            print('height_weight_bmi: bmi patched')
        return height, weight, weight / ((height / 100) ** 2)
    return height, weight, bmi


def wc_hc_whr(wc, hc, whr, verbose=False):
    if wc <= 0 and hc > 0 and whr > 0:
        if verbose:
            print('wc_hc_whr: wc patched')
        return hc * whr, hc, whr
    if wc > 0 and hc <= 0 and whr > 0:
        if verbose:
            print('wc_hc_whr: hc patched')
        return wc, wc / whr, whr
    if wc > 0 and hc > 0 and whr <= 0:
        if verbose:
            print('wc_hc_whr: whr patched')
        return wc, hc, wc / hc
    return wc, hc, whr


def patch_for_one_person(raw_fea, global_mean=None, return_stats=False):
    assert raw_fea.ndim == 2 # (time, fea)
    stat = {
        'total_idx': 0,
        'need_patch': 0, # Num of all neg values
        'num_patched': 0, # Num of patched values
        'num_pos_when_patch': [],
    }
    for f in range(raw_fea.shape[1]):
        fea = raw_fea[:, f]
        if np.all(fea >= 0):
            continue
        neg_idx = np.where(fea < 0)[0]
        pos_idx = np.where(fea > 0)[0]
        stat['total_idx'] += fea.size
        stat['need_patch'] += neg_idx.size
        if global_mean is None:
            # pos_mean = np.mean(fea > 0)
            if pos_idx.size == 1:
                raw_fea[neg_idx, f] = fea[pos_idx]
            elif pos_idx.size > 0:
                raw_fea[neg_idx, f] = np.interp(neg_idx, pos_idx, fea[pos_idx])
            stat['num_patched'] += neg_idx.size
            stat['num_pos_when_patch'].append(pos_idx.size)
        else:
            raise NotImplementedError('NotImplementedError')
    if return_stats:
        return raw_fea, stat
    return raw_fea

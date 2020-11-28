from easydict import EasyDict as edict
import numpy as np
import torch.nn as nn

__C = edict()
cfg = __C

# 0. Basic config

__C.TAG = 'default'
__C.CUDA_COST = False  # for acceleration
__C.CLAMP_INPUT = True
__C.MIN_SCALE = 128  # padding min scale
__C.PAD_BY_CONS = False # padding by constant values
__C.PAD_CONS = -1
__C.SPARSE = False 


# 1. Module Structure

__C.DAP_LAYER = False
__C.DAP_INIT_BY_ID = False
__C.DAP_BY_TEMPERATURE = False

__C.CTF = False   # coarse to fine
__C.CTF_CONTEXT = False

__C.REMOVE_WARP_HOLE = True
__C.FLOW_REG_BY_MAX = True



__C.SCALE_CONTEXT6 = 1.0
__C.SCALE_CONTEXT5 = 1.0
__C.SCALE_CONTEXT4 = 1.0
__C.SCALE_CONTEXT3 = 1.0
__C.SCALE_CONTEXT2 = 1.0

__C.SEARCH_RANGE = [3,3,3,3,3]


# 2. Training

__C.WEIGHT_DECAY = 0.0
__C.MOMENTUM = 0.9
__C.BETA = 0.999

__C.LOSS_TYPE = 'L1'
__C.MultiScale_W = [1.,0.5,0.25]
__C.CROP_SIZE = [256,256]

__C.USE_VALID_RANGE = True
__C.VALID_RANGE = [[8,8],[32,32],[64,64],[128,128]]


__C.HALF_THINGS = False # using half of the flyingthings dataset


# Aug
__C.SIMPLE_AUG = False
__C.RANDOM_TRANS = 10
__C.SPARSE_RESIZE = True # to deal with kitti sparse gt

__C.AUG_brightness = 0.4
__C.AUG_contrast = 0.4
__C.AUG_saturation = 0.4
__C.AUG_hue = 0.159235
__C.ASY_COLOR_AUG = True

__C.SPATIAL_AUG_PROB = 0.8

__C.DROP_LAST = True

__C.SUP_RAW_FLOW = False # supervise both raw flow (by soft argmin) and refined flow (by context net) 




#####
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], edict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))

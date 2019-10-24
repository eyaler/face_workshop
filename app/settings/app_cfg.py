import os
from os.path import join
from pathlib import Path
import logging
import time


# -----------------------------------------------------------------------------
# Filepaths
# -----------------------------------------------------------------------------

DIR_SELF = os.path.dirname(os.path.realpath(__file__))
DIR_ROOT = Path(DIR_SELF).parent.parent

DATA_STORE = join(DIR_ROOT, 'data_store')
DIR_IMAGES = join(DATA_STORE, 'images')
DIR_MODELS = join(DATA_STORE, 'models')
DIR_HAARCASCADES = join(DIR_MODELS, 'opencv')

FP_FRONTALFACE = join(DIR_HAARCASCADES, 'haarcascade_frontalface_default.xml')
FP_FRONTALFACE_ALT = join(DIR_HAARCASCADES, 'haarcascade_frontalface_alt.xml')
FP_FRONTALFACE_ALT2 = join(DIR_HAARCASCADES, 'haarcascade_frontalface_alt2.xml')
FP_FRONTALFACE_ALT_TREE = join(DIR_HAARCASCADES, 'haarcascade_frontalface_alt_tree.xml')
FP_PROFILEFACE = join(DIR_HAARCASCADES, 'haarcascade_profileface.xml')

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------

LOG = logging.getLogger('face_workshop')

ZERO_PADDING = 6
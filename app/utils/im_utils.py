import logging
import math

import cv2 as cv
from PIL import Image, ImageOps
import imutils
import numpy as np

from app.models.bbox import BBoxDim, BBoxNorm

log = logging.getLogger('face_workshop')

def is_pil(im):
  '''Ensures image is Pillow format
  '''
  try:
    im.verify()
    return True
  except:
    return False

def is_np(im):
  '''Checks if image if numpy
  '''
  return type(im) == np.ndarray

def np2pil(im, swap=True):
  '''Convert Numpy ndarray image to Pillow Image
    :param im: image in numpy or PIL.Image format
    :returns: image in Pillow RGB format
  '''
  try:
    im.verify()
    log.warn('Expected Numpy received PIL')
    return im
  except:
    if swap:
      im = bgr2rgb(im)
    return Image.fromarray(im.astype('uint8'), 'RGB')

def ensure_pil(im):
  '''Ensures image is in Pillow Image format
  '''
  return np2pil(im) if is_np(im) else im

def ensure_np(im):
  '''Ensures image is in Numpy ndarray format
  '''
  return pil2np(im) if is_pil(im) else im

def pil2np(im, swap=True):
  '''Ensure image is Numpy.ndarry format
    :param im: image in numpy or PIL.Image format
    :returns: image in Numpy uint8 format
  '''
  if type(im) == np.ndarray:
    log.warn('Expected PIL received Numpy')
    return im
  im = np.asarray(im, np.uint8)
  if swap:
    im = rgb2bgr(im)
  return im

def num_channels(im):
  '''Returns number of channels in numpy.ndarray image'''
  if len(im.shape) > 2:
    return im.shape[2]
  else:
    return 1

def is_grayscale(im, threshold=5):
  '''Returns True if image is grayscale
  :param im: (numpy.array) image
  :return (bool) of if image is grayscale'''
  b = im[:,:,0]
  g = im[:,:,1]
  mean = np.mean(np.abs(g - b))
  return mean < threshold

def fit_image(im, size, method=Image.ANTIALIAS, bleed=0.0, centering=(0.5, 0.5)):
  '''Fits image to size with forced crop
  :param im: (np.ndarry or PIL) image 
  :param size: (int, int) output size
  :param method: (int) interpolation. Options: ANTIALIAS, BICUBIC, BILINEAR, NEAREST
  '''
  if is_np(im):
    im = np2pil(im)
    was_np = True

  if centering[1] * im.size[1] < size[1]/2:
    cy = 0
  elif (1 - centering[1]) * size[1] < size[1]/2:
    cy = 1
  else:
    cy = centering[1]
    
  if centering[0] * im.size[0] < size[0]/2:
    cx = 0
  elif (1 - centering[0]) * size[0] < size[0]/2:
    cx = 1
  else:
    cx = centering[0]

  im = ImageOps.fit(im, size, method=method, bleed=0.0, centering=(cx, cy))
  if was_np:
    im = pil2np(im)
  return im

def crop_from_bbox(im, bbox_norm):
  '''Returns cropped region of image
  '''
  assert is_np(im), 'Must be numpy image'
  dim = im.shape[:2][::-1]
  x1, y1, x2, y2 = bbox_norm.to_bbox_dim(dim).xyxy
  return im[y1:y2, x1:x2]

def set_crop(im_bg, im_new, xyxy=None):
  w, h = im_new.shape[:2][::-1]
  if xyxy is None:
    x1,y1,x2,y2 = (0,0, w,h)
  else:
    x1,y1,x2,y2 = xyxy
  #x2 = min(x2, x1+w)
  #y2 = min(y2, y1+h)
  im_bg[y1:y2, x1:x2] = im_new
  return im_bg

def get_crop(im, xyxy):
  x1,y1,x2,y2 = xyxy
  return im[y1:y2, x1:x2]

def resize_crop_bbox(im, bbox_crop, bbox_roi, new_dim):
  '''Force resize/crop image centered on bbox
  :param im: (numpy) image
  :param bbox_crop: (BBoxNorm) of the crop zone
  :param bbox_roi: (BBoxNorm) of the object roi
  :param new_dim: (int, int) new image size
  '''
  was_pil = False
  if is_pil(im):
    im = pil2np(im)
    was_pil = True
  
  # force image to dimension
  im_dim = im.shape[:2][::-1]
  im_dim_orig = im.shape[:2][::-1]
  bbox_dim = bbox_crop.to_bbox_dim(im_dim)
  # find max scale
  sw = new_dim[0]/bbox_dim.w
  sh = new_dim[1]/bbox_dim.h
  scale_magnitude_w = (1/sw) if sw < 1 else sw
  scale_magnitude_h = (1/sh) if sh < 1 else sh
  offset = [0, 0, 0, 0]
  
  # BUG: add 5% margin to account for float imprecision 
  # bbox_crop * image dimensions can yeild slightly smaller cropped images
  bug_padding_scale = 0.00

  if scale_magnitude_w >= scale_magnitude_h:
    s = sw
    sm = scale_magnitude_w
    tw = math.ceil(s*im_dim[0]) + int((bug_padding_scale*im_dim[0]))
    im = resize(im, width=tw)
  else:
    s = sh
    sm = scale_magnitude_h
    # BUG: add 5% margin to account for float imprecision 
    # bbox_crop * image dimensions can yeild slightly smaller cropped images
    th = math.ceil(s*im_dim[1]) + int((bug_padding_scale*im_dim[1]))
    im = resize(im, height=th)

  im_dim_scaled = im.shape[:2][::-1]
  bbox_crop_scaled_dim = bbox_crop.to_bbox_dim(im_dim_scaled)
  x1, y1, x2, y2 = bbox_crop_scaled_dim.xyxy
  #bbox_crop_scaled = bbox_crop_scaled_dim.to_bbox_norm()
  #im = crop_from_bbox(im, bbox_crop_scaled)
  im = im[y1:y2, x1:x2]
  im_dim_cropped = im.shape[:2][::-1]
  
  # crop excess width
  if im_dim_cropped[0] > new_dim[0]:
    # force crop width
    delta_w = (im_dim_cropped[0] - new_dim[0])
    delta_w_left = delta_w // 2
    offset[0] = delta_w_left
    offset[2] = 0-(delta_w - delta_w_left)
  if im_dim_cropped[1] > new_dim[1]:
    delta_h = (im_dim_cropped[1] - new_dim[1])
    delta_h_top = delta_h // 2
    offset[1] = delta_h_top
    offset[3] = 0-(delta_h - delta_h_top)

  #im_dim_cropped = im.shape[:2][::-1]

  x1, y1, x2, y2 = (offset[0], offset[1], im_dim_cropped[0] + offset[2], im_dim_cropped[1] + offset[3])
  im = im[y1:y2, x1:x2]

  if im_dim_cropped[0] < new_dim[0] or im_dim_cropped[1] < new_dim[1]:
    # TODO: bug that sometimes images are +/- a few pixels less than actual size
    im_tmp_bg = make_np_im(new_dim)
    im = set_crop(im_tmp_bg, im)
  
  # adjust the face roi
  im_dim = im.shape[:2][::-1]
  tx,ty = (bbox_crop_scaled_dim.x1, bbox_crop_scaled_dim.y1)
  translate_xyxy = (-tx, -ty, -tx, -ty)
  translate_offset = (offset[0], offset[1], 0, 0)
  
  bbox_roi_dim = bbox_roi.to_bbox_dim(im_dim_scaled).translate(translate_xyxy).translate(translate_offset)
  bbox_roi = BBoxDim.from_xyxy_dim(bbox_roi_dim.xyxy, im_dim).to_bbox_norm()

  if was_pil:
    im = np2pil(im)
  return im, bbox_roi
  
def make_np_im(wh, color=(0,0,0)):
  '''Creates Numpy image
  :param wh: (int, int) width height
  :param color: (int, int, int) in RGB
  '''
  w,h = wh
  im = np.ones([h,w,3], dtype=np.uint8)
  im[:] = color[::-1]
  return im

############################################
# imutils (external)
# pip install imutils
############################################

def resize(im, width=0, height=0, interpolation=cv.INTER_LINEAR):
  '''resize image using imutils. Use w/h=[0 || None] to prioritize other edge size
    :param im: a Numpy.ndarray image
    :param wh: a tuple of (width, height)
  '''
  # TODO change to cv.resize and add algorithm choices
  w = width
  h = height
  if w is 0 and h is 0:
    return im
  elif w > 0 and h > 0:
    ws = im.shape[1] / w
    hs = im.shape[0] / h
    if ws > hs:
      return imutils.resize(im, width=w, inter=interpolation)
    else:
      return imutils.resize(im, height=h, inter=interpolation)
  elif w > 0 and h is 0:
    return imutils.resize(im, width=w, inter=interpolation)
  elif w is 0 and h > 0:
    return imutils.resize(im, height=h, inter=interpolation)
  else:
    return im



############################################
# OpenCV 
############################################

def bgr2gray(im):
  '''Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns: Numpy.ndarray (Gray)
  '''
  return cv.cvtColor(im, cv.COLOR_BGR2GRAY)

def gray2bgr(im):
  '''Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (Gray)
    :returns: Numpy.ndarray (BGR)
  '''
  return cv.cvtColor(im, cv.COLOR_GRAY2BGR)

def bgr2rgb(im):
  '''Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns: Numpy.ndarray (RGB)
  '''
  return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def rgb2bgr(im):
  '''Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (RGB)
    :returns: Numpy.ndarray (RGB)
  '''
  return cv.cvtColor(im, cv.COLOR_RGB2BGR)
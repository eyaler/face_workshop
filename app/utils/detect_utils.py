import logging

import cv2 as cv
import numpy as np
import dlib

from app.models.bbox import BBoxDim, BBoxNorm
from app.settings import app_cfg

log = logging.getLogger('face_workshop')

class FaceDetectorHAAR:

  # Define a function to detect faces using OpenCV's haarcascades
  def __init__(self, fp_cascade=app_cfg.FP_FRONTALFACE):
    self.classifier = cv.CascadeClassifier(fp_cascade)

  def detect(self, im, scale_factor=1.1,overlaps=3,min_size=70,flags=0):
      
    min_size = (min_size, min_size) # minimum face size
    #max_size = (max_size, max_size) # maximum face size
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Convert to grayscale first
    #matches = classifier.detectMultiScale(im_gray, scale_factor, overlaps, flags, min_size, max_size)
    cv_rects = self.classifier.detectMultiScale(im_gray, 
                                          scaleFactor=scale_factor, 
                                          minNeighbors=overlaps, 
                                          minSize=min_size,
                                          flags=cv.CASCADE_SCALE_IMAGE)
    # By default, OpenCV returns x,y,w,w
    dim = im.shape[:2][::-1]
    cv_rects = [BBoxDim.from_xywh_dim((r[0],r[1],r[2],r[3]), dim).to_bbox_norm() for r in cv_rects]
    return cv_rects


class FaceDetectorDLIB:

  def __init__(self):
    # init dlib
    self.predictor = dlib.shape_predictor(app_cfg.FP_DLIB_PREDICTOR)
    self.detector = dlib.get_frontal_face_detector()

  def detect(self, im, pyramids=0):
    dlib_rects = self.detector(im, pyramids)
    dim = im.shape[:2][::-1]
    bboxes = [ BBoxDim.from_xyxy_dim((r.left() ,r.top(), r.right(), r.bottom()), dim).to_bbox_norm() for r in dlib_rects]
    return bboxes

  def landmarks(self, im, bbox):
    dim = im.shape[:2][::-1]
    x1,y1,x2,y2 = bbox.to_bbox_dim(dim).xyxy
    roi_dlib = dlib.rectangle(x1,y1,x2,y2)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Convert to grayscale first
    landmarks = [[p.x, p.y] for p in self.predictor(im_gray, roi_dlib).parts()]
    return landmarks


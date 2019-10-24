import sys
from math import sqrt

import numpy as np
import cv2 as cv
import PIL
from PIL import ImageDraw

from app.utils import im_utils
from app.settings import app_cfg

log = app_cfg.LOG

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

# ---------------------------------------------------------------------------
#
# 3D landmark drawing utilities
#
# ---------------------------------------------------------------------------

def plot_keypoints(im, kpts):
  '''Draw 68 key points
  :param im: the input im
  :param kpts: (68, 3). flattened list
  '''
  im = im.copy()
  kpts = np.round(kpts).astype(np.int32)
  for i in range(kpts.shape[0]):
    st = kpts[i, :2]
    im = cv.circle(im, (st[0], st[1]), 1, (0, 0, 255), 2)
    if i in end_list:
      continue
    ed = kpts[i + 1, :2]
    im = cv.line(im, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
  return im


def calc_hypotenuse(pts):
  bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
  center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
  radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
  bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
  llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
  return llength / 3

def build_camera_box(rear_size=90):
  point_3d = []
  rear_depth = 0
  point_3d.append((-rear_size, -rear_size, rear_depth))
  point_3d.append((-rear_size, rear_size, rear_depth))
  point_3d.append((rear_size, rear_size, rear_depth))
  point_3d.append((rear_size, -rear_size, rear_depth))
  point_3d.append((-rear_size, -rear_size, rear_depth))

  front_size = int(4 / 3 * rear_size)
  front_depth = int(4 / 3 * rear_size)
  point_3d.append((-front_size, -front_size, front_depth))
  point_3d.append((-front_size, front_size, front_depth))
  point_3d.append((front_size, front_size, front_depth))
  point_3d.append((front_size, -front_size, front_depth))
  point_3d.append((-front_size, -front_size, front_depth))
  point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

  return point_3d


def plot_pose_box(im, Ps, pts68s, color=(40, 255, 0), line_width=2):
  '''Draw a 3D box as annotation of pose. 
    ref: https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
  :param image: the input image
  :param P: (3, 4). Affine Camera Matrix.
  :param kpts: (2, 68) or (3, 68)
  '''
  im_draw = im.copy()
  if not isinstance(pts68s, list):
    pts68s = [pts68s]
  
  if not isinstance(Ps, list):
    Ps = [Ps]
  
  for i in range(len(pts68s)):
    pts68 = pts68s[i]
    llength = calc_hypotenuse(pts68)
    point_3d = build_camera_box(llength)
    P = Ps[i]

    # Map to 2d im points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(pts68[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv.polylines(im_draw, [point_2d], True, color, line_width, cv.LINE_AA)
    cv.line(im_draw, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv.LINE_AA)
    cv.line(im_draw, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv.LINE_AA)
    cv.line(im_draw, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv.LINE_AA)

    return im_draw



# ---------------------------------------------------------------------------
#
# OpenCV drawing functions
#
# ---------------------------------------------------------------------------

pose_types = {'pitch': (0,0,255), 'roll': (255,0,0), 'yaw': (0,255,0)}

def draw_landmarks2D_cv(im, points_norm, radius=3, color=(0,255,0)):
  '''Draws facial landmarks, either 5pt or 68pt
  '''
  im_dst = im.copy()
  dim = im.shape[:2][::-1]
  for x,y in points_norm:
    pt = (int(x*dim[0]), int(y*dim[1]))
    cv.circle(im_dst, pt, radius, color, -1, cv.LINE_AA)
  return im_dst

def draw_landmarks2D_pil(im_pil, points_norm, radius=3, color=(0,255,0)):
  '''Draws facial landmarks, either 5pt or 68pt
  '''
  assert im_utils.is_pil(im_pil)
  draw = ImageDraw.Draw(im_pil)
  dim = im.shape[:2][::-1]
  for x,y in points_norm:
    x1, y1 = (int(x*dim[0]), int(y*dim[1]))
    xyxy = (x1, y1, x1+radius, y1+radius)
    draw.ellipse(xyxy, fill='white')
  del draw
  im_dst = im_utils.ensure_np(im_pil)
  im_dst = im_utils.rgb2bgr(im_dst)
  return im_dst


def draw_landmarks3D_cv(im, points, radius=3, color=(0,255,0)):
  '''Draws 3D facial landmarks
  '''
  im_dst = im.copy()
  for x,y,z in points:
    cv.circle(im_dst, (x,y), radius, color, -1, cv.LINE_AA)
  return im_dst


def draw_bbox_cv(im_np, bbox_norm, color=(0,255,0), stroke_weight=2):
  '''Draws BBox onto cv image
  '''
  bbox_dim = bbox_norm.to_bbox_dim(im_np.shape[:2][::-1])
  return cv.rectangle(im_np, bbox_dim.p1.xy, bbox_dim.p2.xy, color, stroke_weight, cv.LINE_AA)
  

def draw_bbox_pil(im, bboxes_norm, color=(0,255,0), stroke_weight=2):
  '''Draws BBox onto cv image
  :param color: RGB value
  '''
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    was_np = True
  else:
    was_np = False

  if not type(bboxes_norm) == list:
    bboxes_norm = [bboxes_norm]
  

  im_draw = ImageDraw.ImageDraw(im)

  for bbox_norm in bboxes_norm:
    bbox_dim = bbox_norm.to_bbox_dim(im.size)
    xyxy = (bbox_dim.p1.xy, bbox_dim.p2.xy)  
    im_draw.rectangle(xyxy, outline=color, width=stroke_weight)
  del im_draw

  if was_np:
    im = im_utils.pil2np(im)
  return im

  def draw_fsr_pil(im, fsr, font_lg, font_md, color=(0,255,0), stroke_weight=4):
  
    # ensure image is PIL
    if im_utils.is_np(im):
      im = im_utils.np2pil(im)
      was_np = True
    else:
      was_np = False
    
    # init vars
    stroke_outer = 10
    stroke_face = 4
    
    # init draw canvas
    canvas = ImageDraw.ImageDraw(im)
    
    # draw face roi
    bbox_norm = BBoxNorm.from_xyxy(fsr.rect)
    bbox_dim = bbox_norm.to_bbox_dim(im.size)
    xyxy = (bbox_dim.p1.xy, bbox_dim.p2.xy)  
    canvas.rectangle(xyxy, outline=(255,255,255), width=stroke_face)
    
    # draw confidence border
    bbox_norm_outer = BBoxNorm(0.0, 0.0, 1.0, 1.0)
    bbox_dim_outer = bbox_norm_outer.to_bbox_dim(im.size)
    xyxy = (bbox_dim_outer.p1.xy, bbox_dim_outer.p2.xy)  
    canvas.rectangle(xyxy, outline=(255,125,0), width=stroke_outer)
    
    # draw Dataset and score
    
    # draw dataset text bg
    t = fsr.dataset_key
    tw, th = font_lg.getsize(t)
    xyxy = (stroke_outer,stroke_outer,tw*1.2, th*1.6)
    x1,y1,x2,y2 = xyxy
    canvas.rectangle(xyxy, fill=(255,125,0))
    canvas.text((x1+2, y1-4), t, (0,0,0), font_lg)
    
    # draw per
    t = f'{int(100*fsr.confidence)}%'
    tw, th = font_md.getsize(t)
    x1, y1, x2, y2 =  xyxy
    xyxy = (x1, y2, x1+tw*1.2, y2+th*1.6)
    x1, y1, x2, y2 =  xyxy
    canvas.rectangle(xyxy, fill=(255,125,0))
    canvas.text((x1+2, y1), t, (0,0,0), font_md)
    
    # draw subdir/caption under face roi
    t = fsr.subdir
    tw, th = font_md.getsize(t)
    xyxy = (bbox_dim.x1, bbox_dim.y2, bbox_dim.x1+bbox_dim.w, bbox_dim.y2+th+6)
    x1, y1, x2, y2 =  xyxy
    canvas.rectangle(xyxy, fill=(255,255,255))
    canvas.text((x1+2+stroke_face, y1+2), t, (0,0,0), font_md)

    del canvas

    if was_np:
      im = im_utils.pil2np(im)
    return im



def draw_pose(im, pt_nose, image_pts):
  '''Draws 3-axis pose over image
  TODO: normalize point data
  '''
  im_dst = im.copy()
  log.debug(f'pt_nose: {pt_nose}')
  log.debug(f'image_pts pitch: {image_pts["pitch"]}')
  cv.line(im_dst, pt_nose, tuple(image_pts['pitch']), pose_types['pitch'], 3)
  cv.line(im_dst, pt_nose, tuple(image_pts['yaw']), pose_types['yaw'], 3)
  cv.line(im_dst, pt_nose, tuple(image_pts['roll']), pose_types['roll'], 3)
  return im_dst

def draw_text_cv(im, pt_norm, text, size=1.0, color=(0,255,0)):
  '''Draws degrees as text over image
  '''
  im_dst = im.copy()
  dim = im.shape[:2][::-1]
  pt = tuple(map(int, (pt_norm[0]*dim[0], pt_norm[1]*dim[1])))
  cv.putText(im_dst, text, pt, cv.FONT_HERSHEY_SIMPLEX, size, color, thickness=1, lineType=cv.LINE_AA)
  return im_dst


def draw_degrees(im, pose_data, color=(0,255,0)):
  '''Draws degrees as text over image
  '''
  im_dst = im.copy()
  for i, pose_type in enumerate(pose_types.items()):
    k, clr = pose_type
    v = pose_data[k]
    t = '{}: {:.2f}'.format(k, v)
    origin = (10, 30 + (25 * i))
    cv.putText(im_dst, t, origin, cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, thickness=2, lineType=2)
  return im_dst
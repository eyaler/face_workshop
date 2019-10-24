import json
from dataclasses import dataclass, asdict


@dataclass
class RectNorm:
  x1: float
  y1: float
  x2: float
  y2: float

  @property
  def w(self):
    return (self.x2 - self.x1)
  
  @property
  def width(self):
    return self.w

  @property
  def h(self):
    return (self.y2 - self.y1)
  
  @property
  def height(self):
    return self.h
  
  @property
  def cx(self):
    return self.x1 + (self.width / 2)

  @property
  def cy(self):
    return self.y1 + (self.height / 2)

  @property
  def cxcy(self):
    return (self.cx, self.cy)

  @property
  def area(self):
    return self.w * self.h
  
  @property
  def p1(self):
    return (self.x1, self.y1)

  @property
  def p2(self):
    return (self.x2, self.y2)

  @property
  def xyxy(self):
    return (self.x1, self.y1, self.x2, self.y2)

  @property
  def xywh(self):
    return (self.x1, self.y1, self.w, self.h)
  
  def as_xyxy(self):
    """Deprecate"""
    return (self.x1, self.y1, self.x2, self.y2)

  def as_xywh(self):
    """Deprecate"""
    return (self.x1, self.y1, self.w, self.h)
  
  @classmethod
  def from_rect_dim(cls, rd):
    w,h = rd.dim
    x1,y1,x2,y2 = list(map(int, (rd.x1 / w, rd.y1 / h, rd.x2 / w, rd.y2 / h)))
    return cls(x1, y1, x2, y2, dim)
  
  @classmethod
  def from_xyxy(cls, xyxy):
    x1, y1, x2, y2 = xyxy
    return cls(x1, y1, x2, y2)

  @classmethod
  def from_xywh(cls, x, y, w, h):
    x1, y1, x2, y2 = (x, y, x + w, y + h)
    return cls(x1, y1, x2, y2)

  @classmethod
  def from_cxcywh(cls, cx, cy, w, h):
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return cls(x1, y1, x2, y2)

  def to_rect_dim(self, dim):
    w,h = dim
    x1,y1,x2,y2 = list(map(int, (self.x1 * w, self.y1 * h, self.x2 * w, self.y2 * h)))
    return RectDim(x1, y1, x2, y2, dim)

  def to_json(self):
    return {
      'x1': float(self.x1),
      'x2': float(self.x2),
      'y1': float(self.y1),
      'y2': float(self.y2),
    }

  def __str__(self):
    #return json.dumps(asdict(self))
    return str(self.to_json())

  def asdict(self):
    '''Overrides issue with being initialized with float32 which are nonseriazable'''
    return str(self.to_json())



@dataclass
class RectIn(RectNorm):
  
  x1: int
  y1: int
  x2: int
  y2: int  

  @property
  def cx(self):
    return int(self.x1 + (self.width / 2))

  @property
  def cy(self):
    return int(self.y1 + (self.height / 2))
  

@dataclass
class RectDim(RectNorm):
  
  x1: int
  y1: int
  x2: int
  y2: int  
  dim: (int, int)

  @property
  def cx(self):
    return int(self.x1 + (self.width / 2))

  @property
  def cy(self):
    return int(self.y1 + (self.height / 2))
  
  def as_rect_norm(self):
    w,h = self.dim
    x1,y1,x2,y2 = (self.x1 / w, self.y1 / h, self.x2 / w, self.y2 / h)
    return RectNorm(x1, y1, x2, y2)
  
  @classmethod
  def from_rect_norm(cls, rn, dim):
    w,h = dim
    x1,y1,x2,y2 = list(map(int, (rn.x1 * w, rn.y1 * h, rn.x2 * w, rn.y2 * h)))
    return cls(x1, y1, x2, y2, dim)
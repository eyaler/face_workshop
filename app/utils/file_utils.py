"""
File utilities for reading, writing, and formatting data
"""
import sys
import os
from os.path import join
import stat
from pathlib import Path
import json
import csv
from glob import glob, iglob
from datetime import datetime
import time
import pickle
import shutil
import collections
import pathlib

import yaml
import pandas as pd
import numpy as np

from app.settings import app_cfg
import dataclasses, json



log = app_cfg.LOG

# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def mkdirs(fp):
  """Ensure parent directories exist for a filepath
  :param fp: string, Path
  """
  fpp = ensure_posixpath(fp)
  fpp = fpp.parent if fpp.suffix else fpp
  fpp.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def load_yaml(fp_yml, loader=yaml.Loader):
  '''Loads YAML file (Use .yaml, not .yml)'''
  with open(fp_yml, 'r') as fp:
    cfg = yaml.load(fp, Loader=loader)
  return cfg


def load_csv(fp_in, as_list=True):
  """Loads CSV and retuns list of items
  :param fp_in: string filepath to CSV
  :returns: list of all CSV data
  """ 
  if not Path(fp_in).exists():
    log.info('not found: {}'.format(fp_in))
  log.info('loading: {}'.format(fp_in))
  with open(fp_in, 'r') as fp:
    items = csv.DictReader(fp)
    if as_list:
      items = [x for x in items]
    log.info('returning {:,} items'.format(len(items)))
    return items


def load_txt(fp_in):
  with open(fp_in, 'rt') as fp:
    lines = fp.read().rstrip('\n').split('\n')
  return lines


class EnhancedJSONEncoder(json.JSONEncoder):
  def default(self, o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return super().default(o)


def load_json(fp_in):
  """Loads JSON and returns items
  :param fp_in: (str) filepath
  :returns: (dict) data from JSON
  """
  if not Path(fp_in).exists():
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'r') as fp:
    data = json.load(fp)
  return data


def load_pkl(fp_in):
  """Loads Pickle and returns items
  :param fp_in: (str) filepath
  :returns: (dict) data from JSON
  """
  if not Path(fp_in).exists():
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'rb') as fp:
    data = pickle.load(fp)
  return data

def load_file(fp_in):
  ext = get_ext(fp_in)
  if ext == 'json':
    return load_json(fp_in)
  elif ext == 'pkl':
    return load_pkl(fp_in)
  elif ext == 'csv':
    return load_csv(fp_in)
  elif ext == 'txt':
    return load_txt(fp_in)
  elif ext == 'yaml' or ext == 'yml':
    return load_yaml(fp_in)
  else:
    log.error(f'Invalid extension: {ext}')
    return None

# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_txt(data, fp_out, ensure_path=True):
  """Writes text file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data:
    log.error('no data')
    return
    
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if type(data) == list:
      fp.write('\n'.join(data))
    else:
      fp.write(data)


def write_pkl(data, fp_out, ensure_path=True):
  """Writes Pickle file
  :param fp_out: (str)filepath
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp_out) # mkdir
  with open(fp_out, 'wb') as fp:
    pickle.dump(data, fp)


def write_json(data, fp_out, minify=True, ensure_path=True, sort_keys=True, verbose=False):
  """Writes JSON file
  :param fp_out: (str)filepath 
  :param minify: (bool) minify JSON 
  :param verbose: (bool) print status
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if minify:
      json.dump(data, fp, separators=(',',':'), sort_keys=sort_keys, cls=EnhancedJSONEncoder)
      # json.dump(data, fp, separators=(',',':'), sort_keys=sort_keys)
    else:
      json.dump(data, fp, indent=2, sort_keys=sort_keys, cls=EnhancedJSONEncoder)
      # json.dump(data, fp, indent=2, sort_keys=sort_keys)
  if verbose:
    log.info('Wrote JSON: {}'.format(fp_out))


def write_csv(data, fp_out, header=None):
  """ """
  with open(fp_out, 'w') as fp:
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()
    if type(data) is dict:
      for k, v in data.items():
        fp.writerow('{},{}'.format(k, v))


def write_file(data, fp_in, **kwargs):
  ext = get_ext(fp_in)
  if ext == 'json':
    return write_json(data, fp_in, **kwargs)
  elif ext == 'pkl':
    return write_pkl(data, fp_in)
  elif ext == 'csv':
    return write_csv(data, fp_in)
  elif ext == 'txt':
    return write_txt(data, fp_in)
  else:
    log.error(f'Invalid extension: {ext}')
    return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def timestamp_to_str():
  return datetime.now().strftime("%Y%m%d%H%M%S")
  
def zpad(x, zeros=app_cfg.ZERO_PADDING):
  return str(x).zfill(zeros)

def add_suffix(fp, suffix):
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}{suffix}{fpp.suffix}')

def swap_ext(fp, ext):
  """Swaps file extension
  """
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}.{ext}')

def get_ext(fpp, lower=True):
  """Retuns the file extension w/o dot
  :param fpp: (Pathlib.path) filepath
  :param lower: (bool) force lowercase
  :returns: (str) file extension (ie 'jpg')
  """
  fpp = ensure_posixpath(fpp)
  ext = fpp.suffix.replace('.', '')
  return ext.lower() if lower else ext

def ensure_posixpath(fp):
  """Ensures filepath is pathlib.Path
  :param fp: a (str, LazyFile, PosixPath)
  :returns: a PosixPath filepath object
  """
  if type(fp) == str:
    fpp = Path(fp)
  elif type(fp) == pathlib.PosixPath:
    fpp = fp
  else:
    raise TypeError('{} is not a valid filepath type'.format(type(fp)))
  return fpp

def glob_multi(dir_in, exts=['jpg', 'png'], recursive=True):
  files = []
  for ext in exts:
    if recursive:
      fp_glob = join(dir_in, '**/*.{}'.format(ext))
      files +=  glob(fp_glob, recursive=True)
    else:
      fp_glob = join(dir_in, '*.{}'.format(ext))
      files += glob(fp_glob)
  return files

def glob_subdirs_limit(fp_dir_in, ext='jpg', limit=3, random=False):
  '''Globs one level subdirectories and limits files returned
  '''
  files = []
  for subdir in iglob(join(fp_dir_in, '*')):
    glob_files = glob(join(subdir, f'*.{ext}'))
    if glob_files:
      files.extend(glob_files[:limit])
  return files
  
def order_items(records):
  """Orders records by ASC SHA256"""
  return collections.OrderedDict(sorted(records.items(), key=lambda t: t[0]))


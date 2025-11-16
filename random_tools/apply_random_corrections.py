#!/usr/bin/env python3
"""
Program to apply the corrections for random coincidences
"""

import struct
import random
import numpy as np


def save_header(old_head_name, new_head_name, new_events_number):
  with open(old_head_name, 'r', encoding="utf-8") as old_header, open(
      new_head_name, 'w', encoding="utf-8") as new_header:
    line = old_header.readline()
    new_header.write(f'{line}')
    line = old_header.readline()
    s = line.split(': ')
    new_header.write(f'{s[0]}: {new_events_number}\n')
    lines = old_header.readlines()
    for x in range(7):
      new_header.write(f'{lines[x]}')


def calculate_angle(x, y):
  if y >= 0:
    if x >= 0:
      angle = np.arcsin(y / np.sqrt(x**2 + y**2)) * 180. / np.pi
    else:
      angle = 180. - np.arcsin(y / np.sqrt(x**2 + y**2)) * 180. / np.pi
  else:
    if x < 0:
      angle = 180. + np.arcsin(-y / np.sqrt(x**2 + y**2)) * 180. / np.pi
    else:
      angle = 360. - np.arcsin(-y / np.sqrt(x**2 + y**2)) * 180. / np.pi
  return angle


def read_lut(lut_name):
  lut = np.fromfile(lut_name, dtype='float32').reshape((62400, 6))
  return lut


def read_prob_map(map_name):
  prob_map = np.fromfile(
      map_name, dtype='float32', sep=' '
  ).reshape((1200, 1200))
  return prob_map


def save_cdf_no_tof(
    old_file_name, old_head_name, lut_name, map_name
):  #old_file_name - path to .Cdf file; old_head_name - path to .Cdh file; lut_name - path to lut file; map_name - path to .txt file with probability matrix
  new_file_name = old_file_name.split('.Cdf')[0] + '_randomCorrected.Cdf'
  with open(old_file_name, 'rb') as old_file, open(new_file_name,
                                                   'wb') as new_file:
    lut = read_lut(lut_name)
    prob_map = read_prob_map(map_name)
    events = 0
    while True:
      line = old_file.read(12)
      if line == b'':
        break
      _, c1, c2 = struct.unpack('iii', line)
      x1, y1, z1, _, _, _ = lut[c1]
      x2, y2, z2, _, _, _ = lut[c2]
      angle1 = calculate_angle(x1, y1)
      angle2 = calculate_angle(x2, y2)
      module1 = angle1 // 15
      module2 = angle2 // 15
      pos1 = 50 * module1 + (z1 + 250) // 10
      pos2 = 50 * module2 + (z2 + 250) // 10
      if random.random() <= prob_map[int(pos1)][int(pos2)]:
        events += 1
        new_file.write(line)
    print(f'New .Cdf file has been created:\n{new_file_name}')
    new_head_name = old_head_name.split('.Cdh')[0] + '_randomCorrected.Cdh'
    save_header(old_head_name, new_head_name, events)
    print(f'New .Cdh file has been created:\n{new_head_name}')


def save_cdf_with_tof(old_file_name, old_head_name, lut_name, map_name):
  new_file_name = old_file_name.split('.Cdf')[0] + '_randomCorrected.Cdf'
  with open(old_file_name, 'rb') as old_file, open(new_file_name,
                                                   'wb') as new_file:
    lut = read_lut(lut_name)
    prob_map = read_prob_map(map_name)
    events = 0
    while True:
      line = old_file.read(16)
      if line == b'':
        break
      _, _, c1, c2 = struct.unpack('ifii', line)
      x1, y1, z1, _, _, _ = lut[c1]
      x2, y2, z2, _, _, _ = lut[c2]
      angle1 = calculate_angle(x1, y1)
      angle2 = calculate_angle(x2, y2)
      module1 = angle1 // 15
      module2 = angle2 // 15
      pos1 = 50 * module1 + (z1 + 250) // 10
      pos2 = 50 * module2 + (z2 + 250) // 10
      if random.random() <= prob_map[int(pos1)][int(pos2)]:
        events += 1
        new_file.write(line)
    print('New .Cdf file has been created:\n{new_file_name0}')
    new_head_name = old_head_name.split('.Cdh')[0] + '_randomCorrected.Cdh'
    save_header(old_head_name, new_head_name, events)
    print(f'New .Cdh file has been created:\n{new_head_name}')


# ########################
# MAIN
# ########################

if __name__ == '__main__':
  save_cdf_no_tof(
      '/media/sf_shared/J_Modular/ImageQuality/output/SetSeed/converted_df.Cdf',
      '/media/sf_shared/J_Modular/ImageQuality/output/SetSeed/converted_df.Cdh',
      '/home/szymon/Pulpit/GITHUB/total-body-tools/mac_to_lut_conversion/JModularv2',
      '/media/sf_shared/J_Modular/ImageQuality/output/SetSeed/probMatrix.txt'
  )

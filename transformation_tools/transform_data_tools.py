"""
Helper functions for data trasformation from Gate root files
with 'Coincidences' tree to csv format.
"""
import numpy as np
import pandas
try:
  import uproot
except ModuleNotFoundError:
  import uproot3 as uproot

BRANCHES_COINCIDENCES = [
    'eventID1', 'globalPosX1', 'globalPosY1', 'globalPosZ1', 'time1',
    'energy1', 'sourcePosX1', 'sourcePosY1', 'sourcePosZ1', 'comptonPhantom1',
    'comptonCrystal1', 'eventID2', 'globalPosX2', 'globalPosY2', 'globalPosZ2',
    'time2', 'energy2', 'sourcePosX2', 'sourcePosY2', 'sourcePosZ2',
    'comptonPhantom2', 'comptonCrystal2'
]
ALIASES_COINCIDENCES = [
    'eventID1', 'x1', 'y1', 'z1', 't1', 'e1', 'sX1', 'sY1', 'sZ1',
    'comptonPhantom1', 'comptonCrystal1', 'eventID2', 'x2', 'y2', 'z2', 't2',
    'e2', 'sX2', 'sY2', 'sZ2', 'comptonPhantom2', 'comptonCrystal2'
]
TREE_COINCIDENCES = 'Coincidences'

#Column names according to GOJA format
GOJA_FEATURES = [
    "x1",  # 1 gamma detected x position [cm]
    "y1",  # 1 gamma detected y position [cm]
    "z1",  # 1 gamma detected z position [cm]
    "t1",  # 1 gamma detection time [ps]
    "x2",  # 2 gamma detected x position [cm]
    "y2",  # 2 gamma detected y position [cm]
    "z2",  # 2 gamma detected z position [cm]
    "t2",  # 2 gamma detection time [ps]
    "vol1",  # 1 gamma volume ID
    "vol2",  # 2 gamma volume ID
    "e1",  # 1 gamma energy loss during detection [keV]
    "e2",  # 2 gamma energy loss during detection [keV]
    # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
    "class",
    "sX1",  # 1 gamma emission x position [cm]
    "sY1",  # 1 gamma emission y position [cm]
    "sZ1",  # 1 gamma emission z position [cm]
    "sX2",  # 2 gamma emission x position [cm]
    "sY2",  # 2 gamma emission y position [cm]
    "sZ2"  # 2 gamma emission z position [cm]
]


def get_class(row):
  if row['eventID1'] != row['eventID2']:
    return 4
  if row['comptonPhantom1'] == 0 and row['comptonPhantom2'] == 0 and row[
      'comptonCrystal1'] == 1 and row['comptonCrystal2'] == 1:
    return 1
  if row['comptonPhantom1'] != 0 or row['comptonPhantom2'] != 0:
    return 2
  if row['comptonCrystal1'] > 1 or row['comptonCrystal2'] > 1:
    return 3
  return -1


# pylint: disable-msg=C0103
def reconstruct_vertex_with_delta_time(x1, y1, z1, x2, y2, z2, dt):
  """ Reconstruct emission vertex based on the naive method (MLP)

  Args:
    x1 (float): in mm
    y1 (float): in mm
    z1 (float): in mm
    x2 (float): in mm
    y2 (float): in mm
    z2 (float): in mm
    dt (float): time difference in seconds

  Returns:
    float: emission vertex

  """

  sOfL = 3E11  #[mm/s]  0.03 cm/ps = 3 * 10^-2 * 10 mm * 1/s 10^12 = 3E11 mm/s
  halfX = (x1 - x2) / 2
  halfY = (y1 - y2) / 2
  halfZ = (z1 - z2) / 2
  LORHalfSize = (halfX**2 + halfY**2 + halfZ**2)**(1 / 2)
  versX = halfX / LORHalfSize
  versY = halfY / LORHalfSize
  versZ = halfZ / LORHalfSize
  dX = dt * sOfL * versX / 2
  dY = dt * sOfL * versY / 2
  dZ = dt * sOfL * versZ / 2
  rX1 = (x1 + x2) / 2 - dX
  rY1 = (y1 + y2) / 2 - dY
  rZ1 = (z1 + z2) / 2 - dZ
  return (rX1, rY1, rZ1)


def get_length(x1, y1, z1, x2, y2, z2):
  return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(1 / 2)


def get_angle_3D(x1, y1, z1, x2, y2, z2):
  cos3D = (x1 * x2 + y1 * y2 + z1 * z2) / \
      ((x1**2 + y1**2 + z1**2)**(1/2) * (x2**2 + y2**2+z2**2)**(1/2))
  return np.degrees(np.arccos(np.array(cos3D, dtype=np.float32)))


def get_angle_2D(x1, y1, x2, y2):
  cos2D = (x1 * x2 +
           y1 * y2) / ((x1**2 + y1**2)**(1 / 2) * (x2**2 + y2**2)**(1 / 2))
  return np.degrees(np.arccos(np.array(cos2D, dtype=np.float32)))


def get_vect(x1, y1, z1, x2, y2, z2):
  return x2 - x1, y2 - y1, z2 - z1


def get_orthogonal_projection(x1, y1, z1, x2, y2, z2, x0, y0, z0):
  """Compute the orthogonal projection of point (x0,y0,z0) onto the line that passes through point (x1,y1,z1) and (x2,y2,z2).

  Args:
    x1 (float): x coordinate of the first point on the line.
    y1 (float): y coordinate of the first point on the line.
    z1 (float): z coordinate of the first point on the line.
    x2 (float): x coordinate of the second point on the line.
    y2 (float): y coordinate of the second point on the line.
    z2 (float): z coordinate of the second point on the line.
    x0 (float): x coordinate of the point to project.
    y0 (float): y coordinate of the point to project.
    z0 (float): z coordinate of the point to project.

  Returns:
    x, y, z coordinates of the orthogonal projection.
  """
  vect_12 = get_vect(x1, y1, z1, x2, y2, z2)
  vect_10 = get_vect(x1, y1, z1, x0, y0, z0)
  dot = np.dot(vect_12, vect_10)
  norm = get_length(x1, y1, z1, x2, y2, z2)
  start = [x1, y1, z1]
  return [s + dot / (norm**2) * x for s, x in zip(start, vect_12)]


def get_scatter_point(x_E, y_E, z_E, x_1, y_1, z_1, x_2, y_2, z_2, deltaT):
  """Compute the coordinate of the scatter point.

  Args:
    x_E (float): x coordinate of the emission point.
    y_E (float): y coordinate of the emission point.
    z_E (float): z coordinate of the emission point.
    x_1 (float): x coordinate of the non-scattered hit.
    y_1 (float): y coordinate of the non-scattered hit.
    z_1 (float): z coordinate of the non-scattered hit.
    x_2 (float): x coordinate of the scattered hit.
    y_2 (float): y coordinate of the scattered hit.
    z_2 (float): z coordinate of the scattered hit.
    deltaT (float): time difference between hits.

  Returns:
    x, y, z coordinates of the scatter point.
  """
  speed_of_light = 3E11  # mm/s
  x_p, y_p, z_p = get_orthogonal_projection(
      x_E, y_E, z_E, x_1, y_1, z_1, x_2, y_2, z_2
  )
  c = get_length(x_E, y_E, z_E, x_2, y_2, z_2)
  K = deltaT * speed_of_light + get_length(x_E, y_E, z_E, x_1, y_1, z_1)
  r = get_length(x_2, y_2, z_2, x_p, y_p, z_p)

  ma = get_length(x_E, y_E, z_E, x_p, y_p, z_p)
  sqrt_arg = -(-c + r) * (c + r)
  assert sqrt_arg >= 0
  m = K * (-K + c) * (K + c) / (2 *
                                (K**2 - c**2 + r**2)) + np.sqrt(sqrt_arg) * (
                                    K**2 - c**2 + 2 * r**2
                                ) / (2 * (K**2 - c**2 + r**2))
  a = ma - m

  length_e1 = get_length(x_E, y_E, z_E, x_1, y_1, z_1)
  vect_unit_e1 = [
      v / length_e1 for v in get_vect(x_E, y_E, z_E, x_1, y_1, z_1)
  ]

  return [v - a * w for v, w in zip([x_E, y_E, z_E], vect_unit_e1)]


def add_features(df):
  df['dt'] = df['t1'] - df['t2']
  df['rX'], df['rY'], df['rZ'] = reconstruct_vertex_with_delta_time(
      df['x1'], df['y1'], df['z1'], df['x2'], df['y2'], df['z2'], df['dt']
  )
  df['rError'] = np.sqrt(
      (df['sX1'] - df['rX'])**2 + (df['sY1'] - df['rY'])**2 +
      (df['sZ1'] - df['rZ'])**2
  )
  df['lorL'] = get_length(
      df['x1'], df['y1'], df['z1'], df['x2'], df['y2'], df['z2']
  )
  df['deg3D'] = get_angle_3D(
      df['x1'], df['y1'], df['z1'], df['x2'], df['y2'], df['z2']
  )
  df['deg2D'] = get_angle_2D(df['x1'], df['y1'], df['x2'], df['y2'])
  df['rL'] = np.sqrt(
      np.square(df['rX']) + np.square(df['rY']) + np.square(df['rZ'])
  )
  df['eSum'] = df['e1'] + df['e2']
  df['eDiff'] = np.abs(df['e1'] - df['e2'])


def sigma_energy(energy, coeff=0.0444):
  return coeff / np.sqrt(energy) * energy


# CRT = sqrt(2) *FWHM =~ 3.33 sigma
def get_smear_time(crt):
  sigma = crt / 3.33

  def smear_time_(time):
    return np.random.normal(time, sigma)

  return smear_time_


# FWHM = 2sqrt(2ln2) sigma =~2.355 sigma
def get_smear_z(fwhm):
  sigma = fwhm / 2.355

  def smear_z_(z):
    return np.random.normal(z, sigma)

  return smear_z_


# Energy must be in MeV
def smear_energy(energy):
  sigma = 1000. * sigma_energy((energy) * 1. / 1000.)
  return np.random.normal(energy, sigma)


def get_dataframe_from_root_file(in_file, tree_name, branches, aliases):
  tfile = uproot.open(in_file)
  coincidences = tfile[tree_name]
  branches_and_aliases = dict(zip(branches, aliases))
  df = coincidences.arrays(branches, library='pd')
  df.rename(columns=branches_and_aliases, inplace=True)
  return df


def add_classes_and_features(df):
  df['class'] = df.apply(get_class, axis=1)
  add_features(df)
  return df


#From file in GOJA format
def get_dataframe_from_goja_file(in_file, column_names=None, **kwargs):
  if column_names is None:
    column_names = GOJA_FEATURES
  df = pandas.read_csv(
      in_file,
      sep="\t",
      names=column_names,
      header=None,
      index_col=False,
      **kwargs
  )
  return df

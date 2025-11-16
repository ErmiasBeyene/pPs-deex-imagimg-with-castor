"""
Tool for filtering scatter and random events
"""
import argparse
import os
import pathlib
from typing import Callable, Optional, Tuple, List

import h5py
import numba as nb
import numpy as np
import uproot

IndexCalculator = Callable[[int, int, int, int, float, float], Tuple[int, int]]
AngleFilter = Callable[
    [float, float, float, float, float, float, float, float], bool]


def get_index_calculator_random(
    z_len: float = 500., z_indexes: int = 50
) -> IndexCalculator:
  """
    Get function to calculate indexes in random probability matrix

    Parameters
    ----------
    z_len: float
        length of scintillator in mm
    z_indexes: int
        number of segments to divide scintillator along z axis

    Returns
    -------
    IndexCalculator
        Function to calculate indexes
    """

  # pylint: disable=unused-argument
  def index_calculator(rid1: int, rid2: int, cid1: int, cid2: int, z1: float, z2: float) \
          -> Tuple[int, int]:

    z1 = z1 / z_len + 0.5
    z2 = z2 / z_len + 0.5

    idx1 = int((rid1 + z1) * z_indexes)
    idx2 = int((rid2 + z2) * z_indexes)

    return idx1, idx2

  return index_calculator


def get_index_calculator_scatter(strips: int = 13) -> IndexCalculator:
  """
    Get function to calculate indexes in scatter probability matrix

    Parameters
    ----------
    strips: int
        number of scintillators per rsector

    Returns
    -------
    IndexCalculator
        Function to calculate indexes
    """

  # pylint: disable=unused-argument
  def index_calculator(rid1: int, rid2: int, cid1: int, cid2: int, z1: float, z2: float) \
          -> Tuple[int, int]:

    idx1 = rid1 * strips + cid1
    idx2 = rid2 * strips + cid2

    return idx1, idx2

  return index_calculator


def get_angle_filter(tof_max_ns: float = 1.5, preselection_cut: float = 2. * np.pi / 9.) \
        -> AngleFilter:
  """
    Get function to filter out events based on angle

    Parameters
    ----------
    tof_max_ns: float
        maximum time of flight value
    preselection_cut: float
        cut threshold

    Returns
    -------
    AngleFilter
        Function, which returns information whether given event should be filtered out
    """

  def angle_filter(x1, x2, y1, y2, z1, z2, t1, t2) -> bool:
    dt = abs(t1 - t2) * 1.e+9
    a = x1 * x2 + y1 * y2 + z1 * z2
    b = x1 * x1 + y1 * y1 + z1 * z1
    c = x2 * x2 + y2 * y2 + z2 * z2

    angle = np.arccos(a / np.sqrt(b * c)) * 180. / np.pi

    print(f"angle: {angle}")

    return angle < np.pi - preselection_cut * np.sqrt(
        1. - (dt / tof_max_ns) * (dt / tof_max_ns)
    )

  return angle_filter


@nb.njit(parallel=True)
def calculate_filter_mask_impl(
    df,
    mask,
    prob: Optional[np.ndarray] = None,
    index_calculator: Optional[IndexCalculator] = None,
    angle_filter: Optional[AngleFilter] = None
):
  """
    Function to filter events (random or scatter)

    Parameters
    ----------
    df: awkward.Array
        Root tree read as structured array
    mask: np.ndarray
        Filter mask
    prob: Optional[np.ndarray]
        Probability matrix
    index_calculator: Optional[IndexCalculator]
        Function to calculate indexes in probability matrix
    angle_filter: Optional[AngleFilter]
        Function for optional angle filtering

    Notes
    -----
    This function uses numba jit compilation functionality to speed up operation
    """
  time1 = df["time1"]
  time2 = df["time2"]
  event_id1 = df["eventID1"]
  event_id2 = df["eventID2"]
  compton_phantom1 = df["comptonPhantom1"]
  compton_phantom2 = df["comptonPhantom2"]
  rayleigh_phantom1 = df["RayleighPhantom1"]
  rayleigh_phantom2 = df["RayleighPhantom2"]
  global_pos_x1 = df["globalPosX1"]
  global_pos_x2 = df["globalPosX2"]
  global_pos_y1 = df["globalPosY1"]
  global_pos_y2 = df["globalPosY2"]
  global_pos_z1 = df["globalPosZ1"]
  global_pos_z2 = df["globalPosZ2"]
  rsector_id1 = df["rsectorID1"]
  rsector_id2 = df["rsectorID2"]
  crystal_id1 = df["crystalID1"]
  crystal_id2 = df["crystalID2"]

  def is_valid_event(idx: int) -> bool:

    if event_id1[idx] == event_id2[idx] and (compton_phantom1[idx] != 0
                                             or compton_phantom2[idx] != 0
                                             or rayleigh_phantom1[idx] != 0
                                             or rayleigh_phantom2[idx] != 0):
      return False

    if angle_filter is not None and angle_filter(
        global_pos_x1[idx], global_pos_x2[idx], global_pos_y1[idx],
        global_pos_y2[idx], global_pos_z1[idx], global_pos_z2[idx], time1[idx],
        time2[idx]):
      return False

    if index_calculator is not None:
      i1, i2 = index_calculator(
          rsector_id1[idx], rsector_id2[idx], crystal_id1[idx],
          crystal_id2[idx], global_pos_z1[idx], global_pos_z2[idx]
      )

      fraction = prob[i1, i2]

      return np.random.random() <= fraction

    return True

  # pylint: disable=not-an-iterable
  for i in nb.prange(len(df)):
    mask[i] = is_valid_event(i)


def read_probability_matrix(prob_matrix_file: str) -> np.ndarray:
  """
    Read file with probability matrix

    Supported file types are: .txt and .h5

    Parameters
    ----------
    prob_matrix_file: str
        Path to file with probability matrix

    Returns
    -------
    np.ndarray
        probability matrix
    """
  extension = pathlib.Path(prob_matrix_file).suffix
  if extension == ".txt":
    prob_matrix = np.loadtxt(prob_matrix_file)
  elif extension == ".h5":
    with h5py.File(prob_matrix_file, 'r') as f:
      prob_matrix = f['DS1'][:]
  else:
    raise Exception(
        f"Invalid file extension: {extension}. Supported extensions: .txt, .h5"
    )
  return prob_matrix


def filter_events(
    in_files: List[str], out_dir: str, matrix_files: List[str],
    cut_angles: bool
):
  """
    Filter random / scatter events using provided probability matrices. For each input file new file
    with 'filtered' suffix is created in output directory.

    Parameters
    ----------
    in_files: list[str]
        List of paths to input files
    out_dir: str
        Path to output directory
    matrix_files: list[str]
        List of paths to probability matrices
    cut_angles: bool
        Whether to enable angle filter

    """

  for in_file in in_files:
    f = uproot.open(in_file)
    tree = f["Coincidences"]
    columns = [
        name for name in tree.keys() if name not in [
            "comptVolName1", "comptVolName2", "RayleighVolName1",
            "RayleighVolName2"
        ]
    ]

    df = tree.arrays(columns)

    out = uproot.recreate(
        os.path.join(out_dir,
                     pathlib.Path(in_file).stem + ".filtered.root")
    )

    final_mask = np.ones(len(df), dtype=np.bool_)

    if cut_angles:
      mask = np.zeros(len(df), dtype=np.bool_)
      calculate_filter_mask_impl(
          df, mask, angle_filter=nb.njit(get_angle_filter())
      )
      final_mask &= mask

    for prob_matrix_file in matrix_files:
      prob_matrix = read_probability_matrix(prob_matrix_file)
      mask = np.zeros(len(df), dtype=np.bool_)

      calculate_filter_mask_impl(
          df,
          mask,
          prob=prob_matrix,
          index_calculator=nb.njit(get_index_calculator_random())
      )
      final_mask &= mask

    out_df = df[final_mask]
    out["Coincidences"] = out_df


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Tool to filter scatter and random events"
  )

  parser.add_argument(
      "-i",
      "--input",
      metavar="/path/to/input.root",
      type=str,
      dest="in_files",
      required=True,
      nargs='+',
      help="Path to input file(s)"
  )
  parser.add_argument(
      "-d",
      "--output-dir",
      metavar="/path/to/output/",
      type=str,
      dest="out_dir",
      required=True,
      help="Path to output directory"
  )
  parser.add_argument(
      "-p",
      "--prob-matrix",
      action='append',
      metavar="/path/to/probMatrix.txt",
      type=str,
      dest="matrix_files",
      required=False,
      help="Path to probability matrix, can be specified multiple times for "
      "multiple filtering"
  )
  parser.add_argument(
      "--angle_cut",
      metavar="angle in radians",
      type=bool,
      dest="cut_angles",
      required=False,
      default=False,
      help="Whether to enable angle filter"
  )

  args = parser.parse_args()
  filter_events(**vars(args))

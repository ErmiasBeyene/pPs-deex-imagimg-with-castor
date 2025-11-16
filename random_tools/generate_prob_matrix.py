"""
Tool to filter scatter and random event
"""
import argparse

import numba as nb
import numpy as np
import uproot


def load_he_singles_data(input_files):
  """
    Load HESingles tree from multiple files

    Parameters
    ----------
    input_files: list[str]
        List of input files

    Returns
    -------
    awkward.array
        Structured array
    """
  branches = [
      "time", "rsectorID", "crystalID", "globalPosX", "globalPosY",
      "globalPosZ"
  ]
  arr = uproot.concatenate(
      [f"{input_file}:HESingles" for input_file in input_files], branches
  )
  return arr


@nb.njit
def __calculate_valid_events_numbers(arr, is_in_coincidence, prob):
  shift = 200. * 1e-9
  window = 3. * 1e-9
  dtw_number = 0
  end_index = 0
  time = arr["time"]
  rsector_id = arr["rsectorID"]
  global_pos_z = arr["globalPosZ"]

  for begin_index in range(len(arr) - 1):
    t_min = time[begin_index]

    for k in range(end_index + 1, len(arr)):
      if time[k] > t_min + shift + window:
        end_index = k
        break

    if is_in_coincidence[begin_index]:
      continue

    two_photon_coincidence = False
    coincidence_index = -1

    for j in range(begin_index + 1, end_index + 1):
      if time[j] < time[begin_index] + shift:
        continue

      if time[j] > time[begin_index] + shift + window:
        break

      if is_in_coincidence[j] or rsector_id[j] == rsector_id[begin_index]:
        continue

      if not two_photon_coincidence:
        # Found 2nd photon in coincidence
        two_photon_coincidence = True
        coincidence_index = j
      else:
        # More than two photons in coincidence:
        two_photon_coincidence = False
        break

    if two_photon_coincidence:
      dtw_number += 1

      is_in_coincidence[coincidence_index] = True

      rsector_id1 = rsector_id[begin_index]
      rsector_id2 = rsector_id[coincidence_index]
      global_pos_z1 = global_pos_z[begin_index] + 250
      global_pos_z2 = global_pos_z[coincidence_index] + 250

      i1 = int(rsector_id1 * 50) + int(global_pos_z1 / 10)
      i2 = int(rsector_id2 * 50) + int(global_pos_z2 / 10)

      prob[i1][i2] += 1.
      prob[i2][i1] += 1.

  return dtw_number


def calculate_valid_events_numbers(arr, prob_matrix_size):
  """
    Calculate matrix with valid (two photon coincidence) events counts per matrix element

    Parameters
    ----------
    arr: awkward.array
        Structured array with input data
    prob_matrix_size: int
        Size of output array

    Returns
    -------
    tuple[np.ndarray, int]
        Matrix and number of valid events

    """
  prob = np.zeros((prob_matrix_size, prob_matrix_size), dtype=np.float32)
  is_in_coincidence = np.zeros(len(arr), dtype=np.bool_)
  dtw_number = __calculate_valid_events_numbers(arr, is_in_coincidence, prob)
  return prob, dtw_number


def load_coincidences_data(input_files):
  """
    Load Coincidences tree from multiple files

    Parameters
    ----------
    input_files: list[str]
        List of input files

    Returns
    -------
    awkward.array
        Structured array
    """
  branches = [
      "eventID1", "time1", "globalPosX1", "globalPosY1", "globalPosZ1",
      "rsectorID1", "comptonPhantom1", "comptonCrystal1", "RayleighPhantom1",
      "eventID2", "time2", "globalPosX2", "globalPosY2", "globalPosZ2",
      "rsectorID2", "comptonPhantom2", "comptonCrystal2", "RayleighPhantom2"
  ]
  arr = uproot.concatenate(
      [f"{input_file}:Coincidences" for input_file in input_files], branches
  )
  return arr


@nb.njit
def __calculate_mc(arr, mc):
  event_id1 = arr["eventID1"]
  global_pos_z1 = arr["globalPosZ1"]
  rsector_id1 = arr["rsectorID1"]
  compton_phantom1 = arr["comptonPhantom1"]
  compton_crystal1 = arr["comptonCrystal1"]
  rayleigh_phantom1 = arr["RayleighPhantom1"]
  event_id2 = arr["eventID2"]
  global_pos_z2 = arr["globalPosZ2"]
  rsector_id2 = arr["rsectorID2"]
  compton_phantom2 = arr["comptonPhantom2"]
  compton_crystal2 = arr["comptonCrystal2"]
  rayleigh_phantom2 = arr["RayleighPhantom2"]
  for i in range(len(arr)):

    is_compton_phantom = compton_phantom1[i] != 0 or compton_phantom2[i] != 0
    is_compton_crystal = compton_crystal1[i] != 1 or compton_crystal2[i] != 1
    rayleigh_phantom = rayleigh_phantom1[i] != 0 or rayleigh_phantom2[i] != 0

    if event_id1[i] == event_id2[i] and (is_compton_phantom
                                         or is_compton_crystal
                                         or rayleigh_phantom):
      continue

    i1 = int(rsector_id1[i] * 50) + int((global_pos_z1[i] + 250) / 10)
    i2 = int(rsector_id2[i] * 50) + int((global_pos_z2[i] + 250) / 10)

    mc[i1][i2] += 1.
    mc[i2][i1] += 1.


def calculate_mc(arr, prob_matrix_size):
  """
    Calculate matrix with total events counts in simulation per matrix element

    Parameters
    ----------
    arr: awkward.array
        Structured array with input data
    prob_matrix_size: int
        Size of output array

    Returns
    -------
    np.ndarray
        Matrix and number of valid events

    """
  mc = np.zeros((prob_matrix_size, prob_matrix_size), dtype=np.float32)
  __calculate_mc(arr, mc)
  return mc


@nb.njit
def apply_fractions(mc, prob, prob_matrix_size):
  """
    Apply fractions to matrix with valid events count to generate probability matrix

    Parameters
    ----------
    mc: np.ndarray
        Matrix with MC data
    prob: np.ndarray
        Matrix with valid events data
    prob_matrix_size: int
        Size of mc and prob matrices

    """
  for i in range(prob_matrix_size):
    for j in range(prob_matrix_size):
      n = mc[i, j]
      if n == 0:
        prob[i, j] = 0.
        continue

      r = prob[i, j]
      fraction = (n - r) / n
      if fraction < 0.:
        prob[i, j] = 0.
      else:
        prob[i, j] = fraction


def generate_prob_matrix(input_files, output_file, prob_matrix_size):
  """
    Generate random probability matrix

    Parameters
    ----------
    input_files: list[str]
        List of paths to input files
    output_file: str
        Path to output file
    prob_matrix_size: int
        Matrix size
    """
  he_singles_data = load_he_singles_data(input_files)
  prob, _ = calculate_valid_events_numbers(he_singles_data, prob_matrix_size)

  coincidences_data = load_coincidences_data(input_files)
  mc = calculate_mc(coincidences_data, prob_matrix_size)

  apply_fractions(mc, prob, prob_matrix_size)

  np.savetxt(output_file, prob, fmt='%.18g')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Tool to filter scatter and random events"
  )

  parser.add_argument(
      "-i",
      "--input",
      metavar="/path/to/input.root",
      type=str,
      dest="input_files",
      required=True,
      nargs='+',
      help="Paths to input files"
  )
  parser.add_argument(
      "-o",
      "--output",
      metavar="/path/to/output.txt",
      type=str,
      dest="output_file",
      required=True,
      help="Path to output file"
  )

  args = parser.parse_args()
  generate_prob_matrix(**vars(args), prob_matrix_size=1200)

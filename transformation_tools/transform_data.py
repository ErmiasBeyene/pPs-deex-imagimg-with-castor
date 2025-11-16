#!/usr/bin/env python
"""
Script to transform ROOT files from Gate simulations with the
'Coincidences' tree, to csv format.
"""
import argparse
from transformation_tools.transform_data_tools import get_dataframe_from_root_file
from transformation_tools.transform_data_tools import add_classes_and_features
from transformation_tools.transform_data_tools import BRANCHES_COINCIDENCES
from transformation_tools.transform_data_tools import ALIASES_COINCIDENCES
from transformation_tools.transform_data_tools import TREE_COINCIDENCES


def read_command_line():
  """Parse config from commandline parameters

  Returns:
    tuple: in_file(str), out_file(str)

    """
  _parser = argparse.ArgumentParser(
      description='J-PET transformer from GATE file to csv.'
  )
  _parser.add_argument('-i', dest='input', required=True, help='Input file')
  _parser.add_argument('-o', dest='output', required=True, help='Output file')
  _args = _parser.parse_args()

  return _args.input, _args.output


def transform_file_to_csv(in_file, out_file):
  """Reads file in Gate ROOT format  and transforma is to csv.

     Also, it adds features like LOR length and others based on
     input values read from ROOT. The data are read from TTree
     'Coincidences' structure. The values in the output file
     are separated by tab. The units are kept as mm and seconds.

  Args:
    in_file (str): input ROOT file
    out_file (str): output csv file

    """
  df = get_dataframe_from_root_file(
      in_file, TREE_COINCIDENCES, BRANCHES_COINCIDENCES, ALIASES_COINCIDENCES
  )
  df = add_classes_and_features(df)
  df.to_csv(out_file, header=True, index=False, sep='\t')


def main():
  in_file, out_file = read_command_line()
  transform_file_to_csv(in_file, out_file)


if __name__ == '__main__':
  main()

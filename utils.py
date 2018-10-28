import pickle as pkl
import csv
import os

def ls(dir_path, condition):
    """Lists out all files within given dir_path if they match condition"""
    return [x for x in os.listdir(dir_path) if condition(x)]

def load_pickle(pickle_path):
  """Loads pickle from given pickle path"""
  return pkl.load(open(pickle_path, 'rb'))

def dump_pickle(pickle_path, payload, protocol=4):
  """Dumps pickle payload to given pickle path"""
  pkl.dump(payload, open(pickle_path, 'wb', protocol=protocol))

def write_csv(rows, output_path, header=None, delimiter=','):
  """Writes out rows to csv file given output path"""
  with open(output_path, 'w') as csvfile:
    out_writer = csv.writer(csvfile, delimiter=delimiter)
    if header:
      out_writer.writerow(header)
    for row in rows:
      out_writer.writerow(row)

def read_lines(file_path):
    """Returns contents of given file as list of lines"""
    with open(file_path, 'r') as f:
        return f.readlines()

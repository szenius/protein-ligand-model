import matplotlib.pyplot as plt
import pickle as pkl
import csv
import os

def ls(dir_path, condition):
    """Lists out all files within given dir_path if they match condition"""
    return [x for x in os.listdir(dir_path) if condition(x)]

def load_pickle(pickle_path):
  """Loads pickle from given pickle path"""
  return pkl.load(open(pickle_path, 'rb'))

def dump_pickle(pickle_path, payload, protocol=pkl.HIGHEST_PROTOCOL):
  """Dumps pickle payload to given pickle path"""
  pkl.dump(payload, open(pickle_path, 'wb'), protocol=protocol)

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

def get_example_shape(examples):
  """Returns the shape of a single training example"""
  return examples.shape[1:]

def plot(data, labels, colours, xlabel, ylabel, title, filename):
    plt.figure()
    for i in range(len(data)):
      plt.plot(data[i], label=labels[i], c=colours[i])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_single_data(data, color, xlabel, ylabel, title, filename):
  plt.figure()
  plt.plot(data, c=color)
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.savefig(filename, bbox_inches='tight')
  plt.close()

def plot_performance(history, model_name, epochs, batch_size):
  filename = 'train_dist_{}_{}_{}.png'.format(model_name, str(epochs), str(batch_size))
  plot(
    [history['loss'], history['acc']],  # data
    ['loss', 'acc'],                    # labels
    ['b', 'r'],                         # colors
    'epochs',                           # xlabel
    '',                                 # ylabel
    'Training: {}'.format(model_name),  # title
    filename                            # filename
  )

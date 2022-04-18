import tensorflow as tf
import numpy as np
import pandas as pd
from pyfaidx import Fasta

from functools import partial
from random import randrange

# efficient way for one hot encoding DNA sequence from string
# modified from https://gist.github.com/hannes-brt/54ca5d4094b3d96237fa2e820c0945dd

embed = np.zeros([89, 4], np.float32)
embed[ord('A')] = np.array([1, 0, 0, 0])
embed[ord('C')] = np.array([0, 1, 0, 0])
embed[ord('G')] = np.array([0, 0, 1, 0])
embed[ord('T')] = np.array([0, 0, 0, 1])
embed[ord('a')] = np.array([1, 0, 0, 0])
embed[ord('c')] = np.array([0, 1, 0, 0])
embed[ord('g')] = np.array([0, 0, 1, 0])
embed[ord('t')] = np.array([0, 0, 0, 1])
embed[ord('.')] = np.array([.25, .25, .25, .25])

embedding_table = tf.convert_to_tensor(embed)

def one_hot_encode_seq(dna_input, embed, name = "encode_seq"):
  with tf.name_scope(name):
    b = bytearray()
    b.extend(map(ord, str(dna_input)))
    t = tf.convert_to_tensor(b)
    t = tf.cast(t, tf.int32)
    encoded_dna = tf.nn.embedding_lookup(embedding_table, t)

  return encoded_dna

# fetching longer context based on fasta file and pyfaidx

def get_datum(
  ind,
  fasta_ref,
  bed_df,
  context_length = None,
  rand_shift_range = None
):
  row = bed_df.iloc[ind]
  chrname, start, end, t = bed_df.iloc[ind].tolist()
  interval_length = end - start

  chromosome = fasta_ref[chrname]
  chromosome_length = len(chromosome)

  if rand_shift_range is not None:
    min_shift, max_shift = rand_shift_range

    adj_min_shift = max(start + min_shift, 0) - start
    adj_max_shift = min(end + max_shift, chromosome_length) - end

    left_padding = adj_min_shift - min_shift
    right_padding = max_shift - adj_max_shift

    start += adj_min_shift
    end += adj_max_shift

  if context_length is None or context_length <= interval_length:
    seq = chromosome[start:end]
    return one_hot_encode_seq(seq, embed)

  left_padding = right_padding = 0
  
  extra_seq = context_length - interval_length

  extra_left_seq = extra_seq // 2
  extra_right_seq = extra_seq - extra_left_seq

  start -= extra_left_seq
  end += extra_right_seq

  if start < 0:
    left_padding = -start
    start = 0

  if end > chromosome_length:
    right_padding = end - chromosome_length
    end = chromosome_length

  seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)
  return one_hot_encode_seq(seq, embed)

def get_dna_sample(
  bed_file,
  fasta_file,
  filter_type = None,
  context_length = None,
  rand_shift_range = (-2, 2)
):
  df = pd.read_csv(bed_file, sep = '\t', header = None)

  if filter_type is not None:
    df = df[df[3] == filter_type]

  fasta = Fasta(fasta_file, sequence_always_upper = True)
  yield_data_fn = partial(get_datum, fasta_ref = fasta, bed_df = df, context_length = context_length, rand_shift_range = rand_shift_range)

  def inner():
    for ind in range(len(df)):
      yield yield_data_fn(ind)

  return inner

# main function

if __name__ == '__main__':

  generator_fn = get_dna_sample(
    bed_file = './human-sequences.bed',
    fasta_file = './hg38.ml.fa',
    filter_type = 'valid',
    context_length = 196_608
  )

  dataset = tf.data.Dataset.from_generator(generator_fn, tf.float32)
  print(next(iter(dataset)).shape)

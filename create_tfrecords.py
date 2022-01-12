from itertools import islice
from functools import partial
import tensorflow as tf

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_single_example(seq, target):
  seq = seq.numpy()
  target = target.numpy()

  data = {
      'seq' : _float_feature(seq.flatten()),
      'target' : _float_feature(target.flatten()),
  }

  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out

NUM_TRACKS_CONFIG = dict(human = 5313, mouse = 1643)

def map_seq_target(
    element,
    seq_len,
    species,  # 'human' or 'mouse'
    shifts = None
):
  assert species in NUM_TRACKS_CONFIG, f'{species} not found in config'
  num_tracks = NUM_TRACKS_CONFIG[species]

  num_shifts = 0 if shifts is None else len(list(range(shifts[0], shifts[1] + 1)))

  data = {
    'seq':tf.io.FixedLenFeature([(seq_len + num_shifts) * 4], tf.float32),
    'target':tf.io.FixedLenFeature([896 * num_tracks], tf.float32),
  }
  
  content = tf.io.parse_single_example(element, data)
  return content

def create_tfrecords(ds, path = './', chunk_size = 256):
  for ind, batch in enumerate(chunk(iter(ds), chunk_size)):
    writer = tf.io.TFRecordWriter(f'{path}{ind}.tfrecord')

    for seq, target in batch:
      features = parse_single_example(seq, target)
      writer.write(features.SerializeToString())

    writer.close()

if __name__ == '__main__':
  dataset = tf.data.TFRecordDataset(['./0.tfrecord', './1.tfrecord'])
  map_element_fn = partial(map_seq_target, seq_len = 196608, species = 'human', shifts = (-2, 2))
  dataset = dataset.map(map_element_fn)

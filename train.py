import shutil

import tensorflow as tf
import numpy as np
import functools

from tf_metrics import precision, recall, f1

def parse_connl_format(filename, pos, sep=' '):
    with open(filename, 'r') as f:
        current_sequence = {}
        for line in f:
            line = line.strip()
            if line == "":
                if len(current_sequence) > 0:
                    yield current_sequence
                current_sequence = {}
            else:
                items = line.split(sep)
                for p in pos:
                    if p not in current_sequence:
                        current_sequence[p] = []
                    current_sequence[p].append(items[pos[p]])
        if len(current_sequence) > 0:
            yield current_sequence


def get_words(dataset_fn):
    words = set()
    for line in dataset_fn():
        words.update(line['words'])
    return list(words)


def get_tags(dataset_fn):
    tags = set()
    for line in dataset_fn():
        tags.update(line['tags'])
    return list(tags)


def get_chars(dataset_fn):
    chars = set()
    for word in get_words(dataset_fn):
        chars.update(word)
    return list(chars)


def read_glove(filename, vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(vocab), 300))
    with open(filename) as f:
        for line_idx, line in enumerate(f):
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    return embeddings


def input_fn(dataset_fn, batch_size=100, shuffle_and_repeat=False, buffer=15000, epochs=25):
    
    def generator_fn():
        for sentence in dataset_fn():
            words = [w.encode() for w in sentence['words']]
            tags = [t.encode() for t in sentence['tags']]

            chars = [[c.encode() for c in w] for w in sentence['words']]
            lengths = [len(c) for c in chars]
            max_len = max(lengths)
            chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

            yield ((words, len(words)), (chars, lengths)), tags
            
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    
    dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=shapes, output_types=types)
    
    if shuffle_and_repeat:
        dataset = dataset.shuffle(buffer).repeat(epochs)
        
    dataset = dataset \
                .padded_batch(batch_size, shapes, defaults) \
                .prefetch(1)
    
    return dataset


def masked_conv1d_and_max(t, weights, filters, kernel_size):
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = functools.reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))
        
    (words, nwords), (chars, nchars) = features
    
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    vocab_words = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params['words']), num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params['chars']), num_oov_buckets=params['num_oov_buckets'])
    indices = [idx for idx, tag in enumerate(params['tags']) if tag != 'O']
    num_tags = len(indices) + 1
    num_chars = len(params['chars']) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=params['dropout'],
                                        training=training)

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    variable = np.vstack([params['glove'], [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=params['dropout'], training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=params['dropout'], training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_tensor(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    chars = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                           name='chars')
    nchars = tf.placeholder(dtype=tf.int32, shape=[None, None],
                            name='nchars')
    receiver_tensors = {'words': words, 'nwords': nwords,
                        'chars': chars, 'nchars': nchars}
    features = {'words': words, 'nwords': nwords,
                'chars': chars, 'nchars': nchars}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':

    train_set = functools.partial(parse_connl_format, 'dataset/train.txt', {'words': 0, 'tags': 1})
    valid_set = functools.partial(parse_connl_format, 'dataset/valid.txt', {'words': 0, 'tags': 1})

    train_input_fn = functools.partial(input_fn, 
                                       dataset_fn=train_set, 
                                       batch_size=100, 
                                       shuffle_and_repeat=True, 
                                       buffer=15000, 
                                       epochs=25)

    valid_input_fn = functools.partial(input_fn, 
                                       dataset_fn=valid_set)

    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': get_words(train_set),
        'chars': get_chars(train_set),
        'tags': get_tags(train_set),
        'glove': read_glove('dataset/glove_s300.txt', get_words(train_set))
    }

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)

    estimator = tf.estimator.Estimator(model_fn, 'model', cfg, params)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10000)
    valid_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, throttle_secs=120)

    tf.estimator.train_and_evaluate(estimator, train_spec, valid_spec)

    saved_model = estimator.export_saved_model('saved_model', serving_input_receiver_fn)
    shutil.rmtree('dist_model', ignore_errors=True)
    shutil.copytree(saved_model.decode(), 'dist_model')

    
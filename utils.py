from __future__ import print_function
import six
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

def l1_loss(x):
  return tf.reduce_sum(tf.abs(x))

def l2_loss(x):
  return tf.nn.l2_loss(x)

class cachedproperty(object):
  """Simplified https://github.com/pydanny/cached-property"""
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value

def isint(x):
  return isinstance(x, (int, np.int32, np.int64))

def onehot(Y, K=None):
  if K is None:
    K = np.unique(Y)
  elif isint(K):
    K = list(range(K))
  data = np.array([[y == k for k in K] for y in Y]).astype(int)
  return data

def minibatch_indexes(lenX, batch_size=256, num_epochs=50, **kw):
  n = int(np.ceil(lenX / batch_size))
  for epoch in range(num_epochs):
    for batch in range(n):
      i = epoch*n + batch
      yield i, epoch, slice((i%n)*batch_size, ((i%n)+1)*batch_size)

def train_feed(idx, models, **kw):
  """Convert a set of models, a set of indexes, and numpy arrays given by the
  keyword arguments to a set of feed dictionaries for each model."""
  feed = {}
  for m in models:
    feed[m.is_train] = True
    for dictionary in [kw, kw.get('feed_dict', {})]:
      for key, val in six.iteritems(dictionary):
        attr = getattr(m, key) if isinstance(key, str) and hasattr(m, key) else key
        if type(attr) == type(m.X):
          if len(attr.shape) >= 1:
            if attr.shape[0].value is None:
              feed[attr] = val[idx]
  return feed

def train_batches(models, X, y, **kw):
  for i, epoch, idx in minibatch_indexes(len(X), **kw):
    yield i, epoch, train_feed(idx, models, X=X, y=y, **kw)

def reinitialize_variables(sess):
  """Construct a Tensorflow operation to initialize any variables in its graph
  which are not already initialized."""
  uninitialized_vars = []
  for var in tf.global_variables():
    try:
      sess.run(var)
    except tf.errors.FailedPreconditionError:
      uninitialized_vars.append(var)
  return tf.variables_initializer(uninitialized_vars)

def minimize(sess, loss_fn, batches, operations={}, learning_rate=0.001, print_every=None, var_list=None, **kw):
  """Minimize a loss function over the provided batches of data, possibly
  printing progress."""
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_fn, var_list=var_list)
  op_keys = sorted(list(operations.keys()))
  ops = [train_op] + [operations[k] for k in op_keys]
  t = time.time()
  sess.run(reinitialize_variables(sess))
  for i, epoch, batch in batches:
    results = sess.run(ops, feed_dict=batch)
    if print_every and i % print_every == 0:
      s = 'Batch {}, epoch {}, time {:.1f}s'.format(i, epoch, time.time() - t)
      for j,k in enumerate(op_keys, 1):
        s += ', {} {:.4f}'.format(k, results[j])
      print(s)

def tt_split(X, y, test_size=0.2):
  return train_test_split(X, y, test_size=test_size, stratify=y)

def elemwise_sq_cos_sim(v, w, eps=1e-8):
  assert(len(v.shape) == 2)
  assert(len(w.shape) == 2)
  num = np.sum(v*w, axis=1)**2
  den = np.sum(v*v, axis=1) * np.sum(w*w, axis=1)
  return num / (den + eps)

def yules_q_statistic(e1, e2, y_test):
  n = len(y_test)
  n00 = len(e1.intersection(e2))
  n01 = len(e1.difference(e2))
  n10 = len(e2.difference(e1))
  n11 = n - len(e1.union(e2))
  assert(n00+n01+n10+n11 == n)
  numer = n11*n00 - n01*n10
  denom = n11*n00 + n01*n10
  if numer == 0:
    return 0
  else:
    return numer / float(denom)

def disagreement_measure(e1, e2, y_test):
  n = len(y_test)
  n01 = len(e1.difference(e2))
  n10 = len(e2.difference(e1))
  return (n01 + n10) / n

def scoring_fun(y_pred, y_true):
  if len(y_true.shape) == 1:
    assert(y_true.max() == 1) # binary
    if len(y_pred.shape) == 1:
      preds = y_pred
    else:
      preds = y_pred[:,1]
    return roc_auc_score(y_true, preds)
  else:
    return accuracy_fun(y_pred, y_true)

def accuracy_fun(y_pred, y_true):
  if len(y_true.shape) == 1:
    assert(y_true.max() == 1) # binary
    if len(y_pred.shape) == 1:
      preds = (y_pred > 0.5).astype(int)
    else:
      preds = np.argmax(y_pred, axis=1)
    return np.mean(y_true == preds)
  else:
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def error_masks(y_pred, y_true):
  if len(y_true.shape) == 1:
    assert(y_true.max() == 1) # binary
    if len(y_pred.shape) == 1:
      preds = (y_pred > 0.5).astype(int)
    else:
      preds = np.argmax(y_pred, axis=1)
    return (preds != y_true).astype(int)
  else:
    return (np.argmax(y_true, axis=1) != np.argmax(y_pred, axis=1)).astype(int)

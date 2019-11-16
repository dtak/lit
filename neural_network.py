import uuid
import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import *

"""
Object-oriented class for handling neural networks implemented
in Tensorflow.
"""
@add_metaclass(ABCMeta)
class NeuralNetwork():
  def __init__(self, name=None, dtype=tf.float32, **kwargs):
    self.vals = None # Holds the trained weights of the network
    self.name = (name or str(uuid.uuid4())) # Tensorflow variable scope
    self.dtype = dtype
    self.setup_model(**kwargs)
    assert(hasattr(self, 'X'))
    assert(hasattr(self, 'y'))
    assert(hasattr(self, 'logits'))

  def setup_model(self, X=None, y=None, **kw):
    """Defines common placeholders, then calls rebuild_model"""
    with tf.name_scope(self.name):
      self.X = tf.placeholder(self.dtype, self.x_shape, name="X") if X is None else X
      self.y = tf.placeholder(self.dtype, self.y_shape, name="y") if y is None else y
      self.sample_weight = tf.placeholder(self.dtype, [None], name="sample_weight")
      self.is_train = tf.placeholder_with_default(
          tf.constant(False, dtype=tf.bool), shape=(), name="is_train")
    self.model = self.rebuild_model(self.X, **kw)
    self.recompute_vars()

  def rebuild_model(self, X, reuse=None, **kw):
    """Override this in subclasses. Define Tensorflow operations and return
    list whose last entry is logits."""

  @property
  def logits(self):
    return self.model[-1]

  @abstractproperty
  def x_shape(self):
    """Specify the shape of X; for MNIST, this could be [None, 784]"""

  @abstractproperty
  def y_shape(self):
    """Specify the shape of y; for MNIST, this would be [None, 10]"""

  @property
  def num_features(self):
    return np.product(self.x_shape[1:])

  @property
  def num_classes(self):
    return np.product(self.y_shape[1:])

  @property
  def trainable_vars(self):
    """Return this model's trainable variables"""
    return [v for v in tf.trainable_variables() if v in self.vars]

  def input_grad(self, f):
    """Helper to take input gradients"""
    return tf.gradients(f, self.X)[0]

  def cross_entropy_with(self, y):
    """Compute sample-weighted cross-entropy classification loss"""
    w = self.sample_weight / tf.reduce_sum(self.sample_weight)
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y) * w)

  @cachedproperty
  def preds(self):
    """Tensorflow operation to return predicted labels."""
    return tf.argmax(self.logits, axis=1)

  @cachedproperty
  def probs(self):
    """Tensorflow operation to return predicted label probabilities."""
    return tf.nn.softmax(self.logits)

  @cachedproperty
  def logps(self):
    """Tensorflow operation to return predicted label log-probabilities."""
    return self.logits - tf.reduce_logsumexp(self.logits, 1, keepdims=True)

  @cachedproperty
  def grad_sum_logps(self):
    """Tensorflow operation returning gradient of the sum of log-probabilities.
    Can be used as the gradient for LIT for multi-class classification (doesn't
    require labels)."""
    return self.input_grad(self.logps)

  @cachedproperty
  def l2_weights(self):
    """Tensorflow operation returning sum of squared weight values"""
    return tf.add_n([l2_loss(v) for v in self.trainable_vars])

  @cachedproperty
  def cross_entropy(self):
    """Tensorflow operation returning classification cross-entropy"""
    return self.cross_entropy_with(self.y)

  @cachedproperty
  def cross_entropy_input_gradients(self):
    """Tensorflow operation returning gradient of the loss. Can be used as the
    gradient for LIT for multi-class classification but does require labels."""
    return self.input_grad(self.cross_entropy)

  @cachedproperty
  def predicted_logit_input_gradients(self):
    """Tensorflow operation returning gradient of the predicted log-odds.
    Mostly useful for visualization rather than training."""
    return self.input_grad(self.logits * self.y)

  @cachedproperty
  def binary_logits(self):
    """Tensorflow operation returning the actual predicted log-odds (binary
    only)."""
    assert(self.num_classes == 2)
    return self.logps[:,1] - self.logps[:,0]

  @cachedproperty
  def binary_logit_input_gradients(self):
    """Tensorflow operation returning gradient of the predicted binary
    log-odds. This is what we use for LIT in binary classification."""
    return self.input_grad(self.binary_logits)

  @cachedproperty
  def accuracy(self):
    """Tensorflow operation returning classification accuracy."""
    return tf.reduce_mean(tf.cast(tf.equal(self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))

  def score(self, X, y, **kw):
    """Compute classification accuracy for numpy inputs and labels."""
    if len(y.shape) == 2:
      return np.mean(self.predict(X, **kw) == np.argmax(y, 1))
    else:
      return np.mean(self.predict(X, **kw) == y)

  def predict(self, X, **kw):
    """Compute predictions for numpy inputs."""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.preds, X, **kw)

  def predict_logits(self, X, **kw):
    """Compute raw logits for numpy inputs."""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.logits, X, **kw)

  def predict_binary_logodds(self, X, **kw):
    """Compute predicted binary log-odds for numpy inputs."""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.binary_logits, X, **kw)

  def predict_proba(self, X, **kw):
    """Compute predicted probabilities for numpy inputs."""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.probs, X, **kw)

  def batch_eval(self, sess, quantity, X, n=256):
    """Internal helper to batch computations (prevents memory issues)"""
    vals = sess.run(quantity, feed_dict={ self.X: X[:n] })
    stack = np.vstack if len(vals.shape) > 1 else np.hstack
    for i in range(n, len(X), n):
      vals = stack((vals, sess.run(quantity, feed_dict={ self.X: X[i:i+n] })))
    return vals

  def input_gradients(self, X, y=None, n=256, **kw):
    """Computes different kinds of input gradients for inputs (and optionally
    labels). See input_gradients_ for details."""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_input_gradients_(sess, X, y, n, **kw)

  def batch_input_gradients_(self, sess, X, y=None, n=256, **kw):
    yy = y[:n] if y is not None and not isint(y) else y
    grads = self.input_gradients_(sess, X[:n], yy, **kw)
    for i in range(n, len(X), n):
      yy = y[i:i+n] if y is not None and not isint(y) else y
      grads = np.vstack((grads,
        self.input_gradients_(sess, X[i:i+n], yy, **kw)))
    return grads

  def input_gradients_(self, sess, X, y=None, logits=False, quantity=None):
    if quantity is not None:
      return sess.run(quantity, feed_dict={ self.X: X })
    if y is None:
      return sess.run(self.grad_sum_logps, feed_dict={ self.X: X })
    elif logits and self.num_classes == 2:
      return sess.run(self.binary_logit_input_gradients, feed_dict={ self.X: X })
    elif isint(y):
      y = onehot(np.array([y]*len(X)), self.num_classes)
    feed = { self.X: X, self.y: y, self.sample_weight: np.ones(len(X)) }
    if logits:
      return sess.run(self.predicted_logit_input_gradients, feed_dict=feed)
    else:
      return sess.run(self.cross_entropy_input_gradients, feed_dict=feed)

  def loss_function(self, l2_weights=0., **kw):
    """Construct the loss function Tensorflow op given hyperparameters."""
    log_likelihood = self.cross_entropy
    if l2_weights > 0:
      log_prior = l2_weights * self.l2_weights
    else:
      log_prior = 0
    return log_likelihood + log_prior

  def fit(self, X, y, loss_fn=None, init=False, sample_weight=None, **kw):
    """Fit the neural network on the particular dataset."""
    if loss_fn is None:
      loss_fn = self.loss_function(**kw)
    if len(y.shape) == 1:
      y = onehot(y, self.num_classes)
    if sample_weight is None:
      sample_weight = np.ones(len(X))
    ops = { 'xent': self.cross_entropy, 'loss': loss_fn, 'accu': self.accuracy }
    batches = train_batches([self], X, y, sample_weight=sample_weight, **kw)
    with tf.Session() as sess:
      if init: self.init(sess)
      minimize(sess, loss_fn, batches, ops, **kw)
      self.vals = [v.eval() for v in self.vars]

  def recompute_vars(self):
    """Determine which Tensorflow variables are associated with this
    network."""
    self.vars = tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

  def init(self, sess):
    """Prepare this network to be used in a Tensorflow session."""
    if self.vals is None:
      sess.run(tf.global_variables_initializer())
    else:
      for var, val in zip(self.vars, self.vals):
        sess.run(var.assign(val))

  def save(self, filename):
    """Save the weights of the network to a pickle file."""
    with open(filename, 'wb') as f:
      pickle.dump(self.vals, f)

  def load(self, filename):
    """Load the weights of the network from a pickle file."""
    with open(filename, 'rb') as f:
      self.vals = pickle.load(f)

import numpy as np
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier
from utils import *

def squared_cos_sim(v,w,eps=1e-6):
  """Tensorflow operation to compute the elementwise squared cosine
  similarity between two sets of vectors."""
  num = tf.reduce_sum(v*w, axis=1)**2
  den = tf.reduce_sum(v*v, axis=1)*tf.reduce_sum(w*w, axis=1)
  return num / (den + eps)

def train_diverse_models(cls, n, X, y,
    grad_quantity='binary_logit_input_gradients',
    lambda_overlap=0.01, **kw):
  """Main method implementing local independence training."""

  if len(y.shape) == 1:
    y = onehot(y)

  if y.shape[1] > 2 and grad_quantity == 'binary_logit_input_gradients':
    grad_quantity = 'cross_entropy_input_gradients'

  # Instantiate neural networks
  models = [cls() for _ in range(n)]

  # Gather their input gradients
  igrads = [getattr(m, grad_quantity) for m in models]

  # Compute the prediction loss (sum of indiv. losses)
  regular_loss = tf.add_n([m.loss_function(**kw) for m in models])

  # Compute the diversity loss (average CosIndepErr of pairs)
  diverse_loss = tf.add_n([tf.reduce_sum(squared_cos_sim(igrads[i], igrads[j]))
                           for i in range(n)
                           for j in range(i+1, n)]) * lambda_overlap

  # Combine losses and train
  loss = regular_loss + diverse_loss
  ops = { 'xent': regular_loss, 'same': diverse_loss }
  for i, m in enumerate(models, 1):
    ops['acc{}'.format(i)] = m.accuracy
  sw = np.ones(len(X))
  data = train_batches(models, X, y, sample_weight=sw, **kw)
  with tf.Session() as sess:
    minimize(sess, loss, data, operations=ops, **kw)
    for m in models:
      m.vals = [v.eval() for v in m.vars]

  # Return trained models
  return models

def train_amended_xent_models(cls, n, X, y, lambda_overlap=0.01, **kw):
  if len(y.shape) == 1:
    y = onehot(y)

  # Instantiate models
  models = [cls() for _ in range(n)]

  # Compute the prediction loss (sum of indiv. losses)
  regular_loss = tf.add_n([m.loss_function(**kw) for m in models])

  # Compute the diversity loss (cross-entropy between models)
  diverse_loss = -tf.add_n([
      tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=models[i].logits, labels=models[j].probs))
      for i in range(n) for j in range(n) if i != j
  ]) * (lambda_overlap / (n*(n-1)*0.5))

  # Combine losses and train
  loss = regular_loss + diverse_loss
  ops = { 'xent': regular_loss, 'same': diverse_loss }
  for i, m in enumerate(models, 1):
    ops['acc{}'.format(i)] = m.accuracy
  sw = np.ones(len(X))
  data = train_batches(models, X, y, sample_weight=sw, **kw)
  with tf.Session() as sess:
    minimize(sess, loss, data, operations=ops, **kw)
    for m in models:
      m.vals = [v.eval() for v in m.vars]

  # Return trained models
  return models

def train_restart_models(cls, n, X, y, **kw):
  """Fit a collection of models over random restarts."""
  models = [cls() for _ in range(n)]
  for i in range(n):
    models[i].fit(X,y,**kw)
  return models

def train_bagged_models(cls, n, X, y, **kw):
  """Fit a collection of models using bagging."""
  models = [cls() for _ in range(n)]
  for i in range(n):
    idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True).astype(int)
    models[i].fit(X[idx],y[idx],**kw)
  return models

def train_neg_corr_models(cls, n, X, y, lambda_overlap=0.01, **kw):
  """Fit a collection of models using negative correlation learning. Note
  this uses a 0-1 MSE loss rather than cross-entropy."""

  if len(y.shape) == 1:
    y = onehot(y)

  # Instantiate models
  models = [cls() for _ in range(n)]

  # Compute their mean predicted probability
  mean_pred = tf.add_n([m.probs for m in models]) / len(models)

  # Compute MSE prediction loss
  zero_one_losses = [tf.nn.l2_loss(m.probs-m.y) for m in models]

  # Compute diversity losses (difference from mean)
  neg_corr_losses = [-tf.nn.l2_loss(m.probs-mean_pred) for m in models]

  # Combine losses and train
  regular_loss = tf.add_n(zero_one_losses)
  diverse_loss = tf.add_n(neg_corr_losses)
  loss = regular_loss + lambda_overlap * diverse_loss
  ops = { 'xent': regular_loss, 'same': diverse_loss }
  for i, m in enumerate(models, 1):
    ops['acc{}'.format(i)] = m.accuracy
  sw = np.ones(len(X))
  data = train_batches(models, X, y, sample_weight=sw, **kw)
  with tf.Session() as sess:
    minimize(sess, loss, data, operations=ops, **kw)
    for m in models:
      m.vals = [v.eval() for v in m.vars]

  # Return trained models
  return models


def train_adaboost_models(cls, n, X, y, **kw):
  """Fit a collection of neural networks using AdaBoost."""
  if len(y.shape) == 1:
    classes = np.array([0.,1.])
    y_ = y
  else:
    classes = np.arange(y.shape[1]).astype(float)
    y_ = np.argmax(y, axis=1)

  # First, create a wrapper class that can be interepreted as a model from
  # within scikit-learn.
  class sklearn_compatible_mlp():
    def __init__(self, **kwargs):
      self.mlp = cls()
      self.params_ = kwargs
    def get_params(self, **kwargs): return self.params_
    def set_params(self, **kwargs): self.params_ = kwargs
    @property
    def classes_(self): return classes
    @property
    def n_classes_(self): return len(classes)
    def fit(self, X, y, sample_weight=None, **_):
      N = len(X)
      assert(y.shape == (N,))
      assert(np.abs(sample_weight.sum()-1) < 0.001)
      self.mlp = cls()
      self.mlp.fit(X,y,sample_weight=sample_weight, **kw)
    def predict_proba(self, X, **_):
      return self.mlp.predict_proba(X)

  # Now, use scikit-learn's implementation of AdaBoost to fit the ensemble.
  ab = AdaBoostClassifier(base_estimator=sklearn_compatible_mlp(), n_estimators=n)
  ab.fit(X, y_)

  # Return the scikit-learn ensemble instance.
  return ab

def train_diverse_models_w_projection(cls, n, X, y, projections,
    grad_quantity='binary_logit_input_gradients',
    lambda_overlap=0.01, **kw):
  """Local independence training modified to first project gradients to a
  lower dimensional space. This can be used to implement the local
  independence penalty over a manifold."""

  if len(y.shape) == 1:
    y = onehot(y)
  
  # Define Tensorflow operation to project gradients
  D = X.shape[1]
  def project_to(low_dim_basis, high_dim_vectors):
    return tf.reduce_sum(tf.reshape(high_dim_vectors, (-1,1,D)) * low_dim_basis, axis=2)
  
  # Instantiate models and input-space gradients
  models = [cls() for _ in range(n)]
  igrads = [getattr(m, grad_quantity) for m in models]

  # Add a placeholder to each model for the projection matrices
  for m in models:
    m.proj = tf.placeholder(tf.float32, [None,projections.shape[1],projections.shape[2]])

  # Compute prediction and diversity losses
  regular_loss = tf.add_n([m.loss_function(**kw) for m in models])
  diverse_loss = tf.add_n([tf.reduce_sum(squared_cos_sim(
                                              project_to(models[i].proj, igrads[i]),
                                              project_to(models[j].proj, igrads[j])))
                           for i in range(n)
                           for j in range(i+1, n)]) * lambda_overlap
  loss = regular_loss + diverse_loss

  # Train ensemble, passing in the additional projection matrices
  ops = { 'xent': regular_loss, 'same': diverse_loss }
  for i, m in enumerate(models, 1):
    ops['acc{}'.format(i)] = m.accuracy
  sw = np.ones(len(X))
  data = train_batches(models, X, y, sample_weight=sw, proj=projections, **kw)
  with tf.Session() as sess:
    minimize(sess, loss, data, operations=ops, **kw)
    for m in models:
      m.vals = [v.eval() for v in m.vars]

  # Return trained models
  return models

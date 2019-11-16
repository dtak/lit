import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from utils import *
from neural_network import *
from ensembling_methods import *

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--n_models", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str, default='none')
FLAGS = parser.parse_args()

save_dir = FLAGS.save_dir
n_models = FLAGS.n_models
dataset = FLAGS.dataset
split = FLAGS.split

# Load the dataset
X = np.load('datasets/{}_inputs.npy'.format(dataset))
y = np.load('datasets/{}_targets.npy'.format(dataset))

# Decide how to split it
if split == 'norm':
  # If splitting by "norm", split train and test by distance from origin,
  # but subsplit train and validation randomly
  norms = np.linalg.norm(X, axis=1)
  midpt = np.median(norms)
  X_test = X[np.argwhere(norms > midpt)[:,0]]
  y_test = y[np.argwhere(norms > midpt)[:,0]]
  X_train = X[np.argwhere(norms <= midpt)[:,0]]
  y_train = y[np.argwhere(norms <= midpt)[:,0]]
  X_test, X_val, y_test, y_val = tt_split(X_test, y_test)
else:
  # Otherwise, split train/test/val completely randomly
  X_train, X_test, y_train, y_test = tt_split(X, y)
  X_train, X_val, y_train, y_val = tt_split(X_train, y_train)

if split == 'limit':
  # If splitting by "limit", use a smaller training set
  X_train = X_train[:1000]
  y_train = y_train[:1000]

# Define helpers for printing performance metrics to a csv.
cols = (
    ['ensemble_type', 'ensem_val_auc']
    + ['indiv_auc_avg', 'indiv_auc_std', 'ensem_auc', 'indiv_auc_max', 'indiv_auc_min']
    + ['indiv_acc_avg', 'indiv_acc_std', 'ensem_acc', 'indiv_acc_max', 'indiv_acc_min']
    + ['q_stat', 'interrater', 'err_corr', 'grad_cos2']
)

def write_row(row, mode='a+'):
  csv = open(save_dir + 'auc_results.csv', mode)
  csv.write(','.join(row) + '\n')
  csv.close()

write_row(cols, mode='w')

# Define main helper for evaluating models and saving results
def save_models(models, name, moe_auc=None, moe_val=None, moe_acc=None):
  row = {'ensemble_type': name}
  print(name)
  test_preds = []
  val_preds = []
  accs = []
  aucs = []
  grads = []

  # For each model, save its parameters, compute its individual AUC, and
  # compile its predictions
  for i, m in enumerate(models):
    m.save('{}{}_model{}.pkl'.format(save_dir, name, i))
    testp = m.predict_proba(X_test)
    valp = m.predict_proba(X_val)
    auc = scoring_fun(testp, y_test)
    acc = accuracy_fun(testp, y_test)
    print('  Model #{} AUC: {:.4f}, acc: {:.4f}'.format(i+1,auc,acc))
    aucs.append(auc)
    accs.append(acc)
    test_preds.append(testp)
    val_preds.append(valp)
    grads.append(m.input_gradients(X_test, logits=(y_test.max() == 1)))

  # Save max, min, mean, and standard deviation of individual model AUC
  print('  Indiv AUC max: {:.4f}'.format(np.max(aucs)))
  print('  Indiv AUC min: {:.4f}'.format(np.min(aucs)))
  print('  Indiv AUC mu: {:.4f}'.format(np.mean(aucs)))
  print('  Indiv AUC sd: {:.4f}'.format(np.std(aucs)))

  row['indiv_auc_max'] = '{:.6f}'.format(np.max(aucs))
  row['indiv_auc_min'] = '{:.6f}'.format(np.min(aucs))
  row['indiv_auc_avg'] = '{:.6f}'.format(np.mean(aucs))
  row['indiv_auc_std'] = '{:.6f}'.format(np.std(aucs))

  row['indiv_acc_max'] = '{:.6f}'.format(np.max(accs))
  row['indiv_acc_min'] = '{:.6f}'.format(np.min(accs))
  row['indiv_acc_avg'] = '{:.6f}'.format(np.mean(accs))
  row['indiv_acc_std'] = '{:.6f}'.format(np.std(accs))

  # Compute AUC of average prediction
  val_preds = np.array(val_preds)
  test_preds = np.array(test_preds)
  grads = np.array(grads)
  avg_auc = scoring_fun(test_preds.mean(axis=0), y_test)
  avg_acc = accuracy_fun(test_preds.mean(axis=0), y_test)
  print('  Ens. Avg AUC: {:.4f}, acc: {:.4f}'.format(avg_auc, avg_acc))

  avg_auc_val = scoring_fun(val_preds.mean(axis=0), y_val)

  # Report it (unless it's adaboost and we've been passed its weighted
  # predictions)
  if moe_auc is None:
    row['ensem_auc'] = '{:.6f}'.format(avg_auc)
    row['ensem_acc'] = '{:.6f}'.format(avg_acc)
    row['ensem_val_auc'] = '{:.6f}'.format(avg_auc_val)
  else:
    row['ensem_auc'] = '{:.6f}'.format(moe_auc)
    row['ensem_acc'] = '{:.6f}'.format(moe_acc)
    row['ensem_val_auc'] = '{:.6f}'.format(moe_val)

  # If the ensemble had more than one model (i.e. if it wasn't AdaBoost
  # terminating early), then compute standard diversity measures (+ ours).
  if len(models) > 1:
    # First determine where each model erred
    ens_errors = [error_masks(preds, y_test) for preds in test_preds]
    ens_errsets = [set(np.argwhere(error)[:,0]) for error in ens_errors]

    # Compute the error correlation rho_avg
    error_corr = np.mean([
      pearsonr(err1, err2)[0]
      for i,err1 in enumerate(ens_errors)
      for err2 in ens_errors[i+1:]])

    # Compute the q-statistic
    q_stat = np.mean([
      yules_q_statistic(e1,e2,y_test)
      for i,e1 in enumerate(ens_errsets)
      for e2 in ens_errsets[i+1:]])

    # Compute the interrater agreement (See Eq. 16 of Kuncheva & Whitaker 2003)
    Dis_av = np.mean([
      disagreement_measure(e1,e2,y_test)
      for i,e1 in enumerate(ens_errsets)
      for e2 in ens_errsets[i+1:]])
    avg_acc = np.mean(accs)
    try:
      kappa = 1 - Dis_av / (2 * avg_acc * (1-avg_acc))
    except ZeroDivisionError:
      kappa = np.nan

    # Compute the value of the LIT penalty
    gradcos2 = np.mean([elemwise_sq_cos_sim(g1, g2)
      for i,g1 in enumerate(grads)
      for g2 in grads[i+1:]])

    print('  Q. statistic: {:.4f}'.format(q_stat))
    print('  Interrater agg: {:.4f}'.format(kappa))
    print('  Err. correl.: {:.4f}'.format(error_corr))
    print('  Av grad cos2: {:.4f}'.format(gradcos2))

    row['q_stat'] = '{:.6f}'.format(q_stat)
    row['interrater'] = '{:.6f}'.format(kappa)
    row['err_corr'] = '{:.6f}'.format(error_corr)
    row['grad_cos2'] = '{:.6f}'.format(gradcos2)
  else:
    row['q_stat'] = 'nan'
    row['interrater'] = 'nan'
    row['err_corr'] = 'nan'
    row['grad_cos2'] = 'nan'

  assert(set(row.keys()) == set(cols))

  # Print everything to the CSV.
  write_row([row[k] for k in cols])

# Define the neural network architecture - 256-unit hidden layer w/ dropout
if len(y_train.shape) == 1:
  y_shape = 2
else:
  y_shape = y_train.shape[1]
class Net(NeuralNetwork):
  @property
  def x_shape(self): return [None, X_train.shape[1]]
  @property
  def y_shape(self): return [None, y_shape]
  def rebuild_model(self, X, **_):
    L0 = X
    L1 = tf.layers.dense(L0, 256, name=self.name+'/L1', activation=tf.nn.relu)
    L1 = tf.layers.dropout(L1, training=self.is_train)
    L2 = tf.layers.dense(L1, y_shape, name=self.name+'/L2', activation=None)
    return [L1, L2]

# Set up training parameters -- we'll use 0.0001 weight decay and train for the
# minimum epochs to run for 5000 iterations.
num_epochs = int(np.ceil(np.ceil((5000*128) / float(len(X_train)))))
train_args = [Net, n_models, X_train, y_train]
train_kwargs = {
  'num_epochs': num_epochs,
  'l2_weights': 0.0001,
  'print_every': 100
}

# Train random restarts
tf.reset_default_graph()
save_models(train_restart_models(*train_args, **train_kwargs), 'restarts')

# Train bagging
tf.reset_default_graph()
save_models(train_bagged_models(*train_args, **train_kwargs), 'baggings')

# Train adaboost (using scikit-learn's default implementation)
tf.reset_default_graph()
adaboost = train_adaboost_models(*train_args, **train_kwargs)
adaboost_models = [e.mlp for e in adaboost.estimators_]
save_models(adaboost_models, 'adaboost',
        moe_auc=scoring_fun(adaboost.predict_proba(X_test), y_test),
        moe_val=scoring_fun(adaboost.predict_proba(X_val), y_val),
        moe_acc=accuracy_fun(adaboost.predict_proba(X_test), y_test))

for penalty in np.logspace(-4, 1, 16):
  # Run LIT
  tf.reset_default_graph()
  save_models(
    train_diverse_models(*train_args, lambda_overlap=penalty, **train_kwargs),
    'diverse-{:.4f}'.format(penalty))

  # Run NCL
  tf.reset_default_graph()
  save_models(
    train_neg_corr_models(*train_args, lambda_overlap=penalty, **train_kwargs),
    'negcorr-{:.4f}'.format(penalty))

  # Run ACE
  tf.reset_default_graph()
  save_models(
    train_amended_xent_models(*train_args, lambda_overlap=penalty, **train_kwargs),
    'amended-{:.4f}'.format(penalty))

import os
import re
import csv
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

"""This is a script to aggregate the results of running `launch_jobs.py` into a
set of CSV files that, for each dataset and splitting method, summarize
mean/std/sem predictive performances and diversity metrics. These are then used
by additional code to output plots and LaTeX tables."""

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str)
FLAGS = parser.parse_args()

files = glob.glob(os.path.join(FLAGS.result_dir, '**/*.csv'))

methods = ['restarts','baggings','adaboost','negcorr','amended','diverse']

def method_label(m):
  if '-' in m:
    lmb = ', $\lambda=10^{'+'{:.1f}'.format(np.log10(float(m.split('-')[-1])))+'}$'
    if m.startswith('diverse'):
      return 'LIT'+lmb
    elif m.startswith('negcorr'):
      return 'NCL'+lmb
    elif m.startswith('amended'):
      return 'ACE'+lmb
    else:
      assert(False)
  else:
    return {
      'diverse': 'LIT',
      'negcorr': 'NCL',
      'amended': 'ACE',
      'restarts': 'RRs',
      'baggings': 'Bag',
      'adaboost': 'Ada' }[m]

def load_run_with_cross_validation(f):
  df = pd.read_csv(f)
  df['n_models'] = int(re.search('n-models-(\d+)', f).group(1))
  df['reg_param'] = [float(name.split('-')[-1]) if '-' in name else np.nan for name in df.ensemble_type]
  for prefix in ['diverse','negcorr','amended']:
    rows = df[df.ensemble_type.str.startswith(prefix)]
    if len(rows) > 0:
      max_idx = rows.ensem_val_auc.idxmax()
      max_row = rows.loc[max_idx]
      max_row['ensemble_type'] = prefix
      df = df.append(max_row)
  return df

def aggregate(fs):
  dfs = [load_run_with_cross_validation(f) for f in fs]
  cols = list(dfs[0].columns)
  cols.remove('ensemble_type')
  aggs = dict((c, ['mean','std','sem']) for c in cols)
  return pd.concat(dfs).groupby('ensemble_type').agg(aggs)

def load_experiment(ds, split):
  return aggregate([f for f in files if (('dataset-{}'.format(ds) in f) and ('split-{}'.format(split) in f))])

for ds in ['mushroom','ionosphere','sonar','spectf','electricity','icu']:
  for split in ['none','norm']:
    if ds == 'icu' and split == 'norm': split = 'limit'
    print(ds, split)

    exp = load_experiment(ds, split)
    cols = ['method']
    for col in exp.columns.levels[0]:
      cols.append(col + '_mu')
      cols.append(col + '_sd')
      cols.append(col + '_se')

    result_file = os.path.join(FLAGS.result_dir, '{}-{}.csv'.format(ds, split))
    with open(result_file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=cols)
      writer.writeheader()
      rows = []
      methods = [exp.iloc[i].name for i in range(len(exp))]
      for method in methods:
        row = { 'method': method_label(method) }
        for col in exp.columns.levels[0]:
          row[col + '_mu'] = exp.loc[method][col]['mean']
          row[col + '_sd'] = exp.loc[method][col]['std']
          row[col + '_se'] = exp.loc[method][col]['sem']
        writer.writerow(row)

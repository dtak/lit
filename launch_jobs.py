import os
import argparse
import numpy as np

"""This is a script to repeatedly launch experiments (i.e. invoke
`run_experiment.py`) and generate the full set of results from the paper.

NOTE: Some of this code is specific to Harvard Odyssey, and would need to be
rewritten to work on different research clusters."""

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str)
parser.add_argument("--conda_env", type=str)
parser.add_argument("--partition", type=str)
parser.add_argument("--mem_limit", type=str, default='20000')
FLAGS = parser.parse_args()

slurm_template = """#!/bin/bash
#SBATCH --mem={mem_limit}
#SBATCH -t {time_limit}
#SBATCH -p {partition}
#SBATCH -o {out_file}
#SBATCH -e {err_file}

module load Anaconda3/5.0.1-fasrc01
source activate {conda_env}
{job_command}
"""

def launch_job(restart, dataset, n_models, split, time_limit=None, mem_limit=None):
  if time_limit is None: time_limit = '0-{0:02d}:00'.format(min(24, n_models*2))
  if mem_limit is None: mem_limit = FLAGS.mem_limit

  save_dir = 'restart-{}__dataset-{}__n-models-{}__split-{}/'.format(restart+1, dataset, n_models, split)
  save_dir = os.path.join(FLAGS.base_dir, save_dir)
  out_file = os.path.join(save_dir, 'job-%j.out')
  err_file = os.path.join(save_dir, 'job-%j.err')
  slurm_file = os.path.join(save_dir, 'job.slurm')
  os.system('mkdir -p {}'.format(save_dir))

  job_command = "python -u run_experiment.py --save_dir={} --n_models={} --dataset={} --split={}".format(save_dir, n_models, dataset, split)
  slurm_command = slurm_template.format(
    job_command=job_command,
    time_limit=time_limit,
    mem_limit=mem_limit,
    partition=FLAGS.partition,
    conda_env=FLAGS.conda_env,
    out_file=out_file,
    err_file=err_file)
  with open(slurm_file, "w") as f: f.write(slurm_command)
  os.system("cat {} | sbatch".format(slurm_file))

datasets = ['covertype', 'ionosphere', 'sonar', 'spectf', 'mushroom', 'electricity']
datasets += ['icu'] # available on request if you have access to MIMIC-III.

for restart in range(10):
  for n_models in [2,3,5,8,13]:
    for dataset in datasets:
      splits = ['none', 'limit'] if 'icu' in dataset else ['none', 'norm']
      for split in splits:
        launch_job(restart, dataset, n_models, split)

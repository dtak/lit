# Ensembles of Locally Independent Prediction Models

This repository contains code used to generate the non-clinical results in [Ensembles of Locally Independent Prediction Models](https://arxiv.org/abs/1911.01291).

## Main Idea

Ensembling is a subfield of machine learning that studies how predictions from multiple models, all trying to solve the same task, can be combined to improve performance. One intuition behind ensembling is that there is "wisdom in the crowds" -- that individual models may be fallible, but that many together may be more correct. However, this intuition rests on the assumption that the models are unbiased (i.e. make errors independent of the true label) and that they aren't all wrong in the same ways (i.e. make errors independent of each other).

This second property -- the property of making independent errors on new data -- is often called the "diversity" of the ensemble. Although it cannot be measured directly without access to that new data, many ensembling methods still try to encourage diversity by optimizing proxies for it that _can_ be evaluated on training data. One example is [negative correlation learning](https://ieeexplore.ieee.org/document/809027) (NCL), which penalizes models for making correlated predictions on the training set, even as it still encourages them all to make correct predictions.

However, these two goals are clearly at odds. On the training set, which we assume to be "true" in some way, we really do want all of our models to make correct predictions. If that happens their predictions will of course be statistically dependent, but the statistical dependence of training set predictions shouldn't necessarily imply anything about the independence of errors on _new data_, especially under distributional shift.

So instead of trying to reduce the dependence of training predictions, our method (local independence training, or LIT) tries to enforce independence between _changes_ in training predictions when we _locally extrapolate_ away from the data --- where we define these extrapolations in terms of infinitesimal Gaussian perturbations, possibly projected down to the data manifold. What we find is that this procedure produces a qualitatively different kind of model diversity than prediction-based methods like NCL, and that in many cases, it can lead to much more diverse and accurate predictions on new data. However, the number of locally independent models we can successfully train without an accuracy tradeoff depends on the amount of ambiguity present in the dataset.

## Repository Structure

- [2D-Synthetic.ipynb](./2D-Synthetic.ipynb) contains the 2D synthetic experiments that we use to demonstrate the qualitative differences between random restarts, LIT, and NCL. It's a good starting point for understanding the contribution intuitively.
- [2D-Manifold.ipynb](./2D-Manifold.ipynb) demonstrates how LIT can be generalized to work with projections down to a data manifold, providing a strategy for addressing an important limitation.
- [ensembling_methods.py](./ensembling_methods.py) contains implementations of LIT as well as baseline methods (NCL, bagging, AdaBoost, and amended cross-entropy), building on top of [neural_network.py](./neural_network.py) (which abstracts away some of the Tensorflow boilerplate code).
- [run_experiment.py](./run_experiment.py) contains the core script that generated the main quantitative results (to fully replicate the results, see also [launch_jobs.py](./launch_jobs.py) and [aggregate_results.py](./aggregate_results.py).

## Note About Data

We provide the (z-scored) benchmark classification datasets used to generate the paper in the [datasets](./datasets) directory, but we omit the ICU mortality dataset (generated using the same procedures from [Ghassemi et al. 2017](https://www.ncbi.nlm.nih.gov/pubmed/28815112)) from this public repository since it requires access to the [MIMIC-III database](https://mimic.physionet.org/). If you have already obtained access to MIMIC-III and seek to reproduce those results, please contact us using the emails provided in the paper.

## Citation

You can cite this work using

```
@inproceedings{ross2020ensembles,
  author    = {Ross, Andrew and Pan, Weiwei and Doshi-Velez, Finale},
  title     = {Ensembles of Locally Independent Prediction Models},
  booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.01291},
}
```

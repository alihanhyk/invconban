# Inverse Contextual Bandits: Learning How Behavior Evolves over Time
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository contains the necessary code to replicate the main experimental results in the ICML 2022 paper 'Inverse Contextual Bandits: Learning How Behavior Evolves over Time.' Our proposed methods, *Bayesian ICB* and *Nonparametric Bayesian ICB*, are implemented in `src/main-bicb.py` and `src/main-nbicb.py` respectively.

### Usage

First, install the required python packages by running:
```shell
    python -m pip install -r requirements.txt
```

Then, the main experimental results in the paper can be replicated by running:
```shell
    ./run.sh
    python src/eval1.py  # Table 2
    python src/eval2.py  # Table 3
```

Note that, in order to run these experiments, you need to get access to the [Organ Procurement and Transplantation Network (OPTN)](https://optn.transplant.hrsa.gov) dataset for liver transplantations as of December 4, 2020.

### Citing
If you use this software please cite as follows:
```
@inproceedings{huyuk2022inverse,
  author={Alihan H\"uy\"uk and Daniel Jarrett and Mihaela van der Schaar},
  title={Inverse contextual bandits: Learning how behavior evolves over time},
  booktitle={Proceedings of the 39th International Conference on Machine Learning (ICML)},
  year={2022}
}
```

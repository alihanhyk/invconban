import dill
import matplotlib.pyplot as plt
import numpy as np

methods = ['birl', 'ispi', 'trex', 'bicb']
agents = [(0, 'stationary'), (1, 'sampling'), (11, 'optimistic'), (12, 'greedy'), (2, 'stepping'), (21, 'linear'), (3, 'regressing')]

with open('data/agent0-key0.obj', 'rb') as f:
    data = dill.load(f)
    x_true = data['rhox']

for agent, agent_tag in agents:
    errs = {method: list() for method in methods}
    errs_baseline = list()

    for key in range(5):
        for method in methods:

            with open('res/{}-agent{}-key{}.obj'.format(method, agent, key), 'rb') as f:
                res = dill.load(f)
                x = res['rhox']

            err = np.abs(x-x_true).sum(axis=-1)
            errs[method].append(err)

        K = x_true.shape[0]
        errs_baseline.append(np.abs(-np.ones(K)/K-x_true).sum(axis=-1))

    errs = {method: np.array(errs[method]) for method in methods}
    errs_baseline = np.array(errs_baseline)

    print('--- {} ---'.format(agent_tag))
    print('baseline: {} ({})'.format(errs_baseline.mean(), errs_baseline.std()))
    for method in methods:
        print('{}: {} ({})'.format(method, errs[method].mean(), errs[method].std()))
    print()

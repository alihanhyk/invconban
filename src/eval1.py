import dill
import matplotlib.pyplot as plt
import numpy as np

methods = ['birl', 'irl-kfold-k5', 'irl-kfold-k10', 'irl-kfold-k500', 'ns-irl-k5', 'ns-irl-k10', 'bicb', 'nbicb']
agents = [(0, 'stationary'), (1, 'sampling'), (11, 'optimistic'), (12, 'greedy'), (2, 'stepping'), (21, 'linear'), (3, 'regressing')]

for agent, agent_tag in agents:
    errs = {method: list() for method in methods}
    errs_baseline = list()

    for key in range(5):

        with open('data/agent{}-key{}.obj'.format(agent, key), 'rb') as f:
            data = dill.load(f)
            if 'betas_mean' in data:
                x_true = data['betas_mean']
                x_true = x_true / np.abs(x_true).sum(axis=-1, keepdims=True)
            elif 'rhos' in data:
                x_true = np.array(data['rhos'])
            elif 'rhox' in data:
                x_true = data['rhox'][None,...].repeat(len(data['x']), axis=0)

        for method in methods:

            with open('res/{}-agent{}-key{}.obj'.format(method, agent, key), 'rb') as f:
                res = dill.load(f)
                if 'betas_mean' in res:
                    x = res['betas_mean']
                    x = x / np.abs(x).sum(axis=-1, keepdims=True)
                elif 'betas' in res:
                    x = res['betas']
                elif 'rhos' in res:
                    x = res['rhos']
                elif 'rhox' in res:
                    x = res['rhox'][None,...].repeat(x_true.shape[0], axis=0)

            err = np.abs(x-x_true).sum(axis=-1).mean()
            errs[method].append(err)

        K = x_true.shape[-1]
        errs_baseline.append(np.abs(-np.ones(K)/K-x_true).sum(axis=-1).mean())

    errs = {method: np.array(errs[method]) for method in methods}
    errs_baseline = np.array(errs_baseline)

    print('--- {} ---'.format(agent_tag))
    print('baseline: {} ({})'.format(errs_baseline.mean(), errs_baseline.std()))
    for method in methods:
        print('{}: {} ({})'.format(method, errs[method].mean(), errs[method].std()))
    print()

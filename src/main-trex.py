import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import tqdm

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/trex.obj')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 20

def _likelihood_inner(arg0, arg1):
    (res, t1, qt1, rhox), t0 = arg0, arg1
    qt0 = alpha * np.sum(data_x[t0,data_a[t0]] * rhox)
    res = res + (t0 < t1) * (qt1 - logsumexp(np.array([qt0, qt1])))
    return (res, t1, qt1, rhox), None

def _likelihood_outer(arg0, arg1):
    (res, rhox), t1 = arg0, arg1
    qt1 = alpha * np.sum(data_x[t1,data_a[t1]] * rhox)
    (res, *_), _ = jax.lax.scan(_likelihood_inner, (res, t1, qt1, rhox), np.arange(T))
    return (res, rhox), None

@jax.jit
def likelihood(_log_rhox):
    rhox = -np.exp(_log_rhox - logsumexp(_log_rhox))
    (res, *_), _ = jax.lax.scan(_likelihood_outer, (0., rhox), np.arange(T))
    return res

grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)


_log_rhox = np.zeros(K)
grad_mnsq = np.zeros(K)

trange = tqdm.tqdm(range(10_000), miniters=100)
for iter in trange:

    grad = grad_likelihood(_log_rhox)
    grad_mnsq = .1 * grad**2 + .9 * grad_mnsq
    _log_rhox += .001 * grad / (np.sqrt(grad_mnsq) + 1e-9)

    if iter % 100 == 0:
        trange.set_postfix({'like': likelihood(_log_rhox)})

rhox = -np.exp(_log_rhox - logsumexp(_log_rhox))
print('rhox: {}'.format(rhox))

res = dict()
res['rhox'] = rhox
with open(args.output, 'wb') as f:
    dill.dump(res, f)

import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/lfl.obj')
args = parser.parse_args()

hyper = dict()
hyper['sample_start'] = 10_000
hyper['sample_stop'] = 20_000
hyper['sample_step'] = 10
hyper['delta'] = .05

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 2 ###

def _compute_qs(arg0, arg1):
    (q, v, rhox), t = arg0, arg1
    q1 = np.einsum('ijk,k->ij', data_x, rhox) + np.mean(v)
    v1 = logsumexp(q1 * alpha, axis=-1) / alpha
    return (q1, v1, rhox), (q1[t], v1[t])

@jax.jit
def compute_qs(rhox):
    q = np.zeros((T,A))
    v = logsumexp(q * alpha, axis=-1) / alpha
    _, (qs, vs) = jax.lax.scan(_compute_qs, (q, v, rhox), np.arange(T))
    return qs, vs

def _sample_like0(q, v, a):
    return (q[a] - v) * alpha
_sample_like0 = jax.vmap(_sample_like0)

def _sample_like(rhox):
    qs, vs = compute_qs(rhox)
    return _sample_like0(qs, vs, data_a).sum()

def _sample(arg0, arg1):
    (rate, rhox, like), key = arg0, arg1
    keys = jax.random.split(key, 2)
    _rhox = rhox + hyper['delta'] * jax.random.normal(keys[0], shape=(K,))
    _like = _sample_like(_rhox)
    cond = _like - like > np.log(jax.random.uniform(keys[1]))
    rate = rate + cond
    rhox = jax.lax.select(cond, _rhox, rhox)
    like = jax.lax.select(cond, _like, like)
    return (rate, rhox, like), rhox

def sample(rhox, key, count):
    like = _sample_like(rhox)
    (rate, rhox, *_), RHOX = jax.lax.scan(_sample, (0., rhox, like), jax.random.split(key, count))
    return rate/count, rhox, RHOX
sample = jax.jit(sample, static_argnums=2)

key, subkey = jax.random.split(key)
rhox = -np.ones(K)/K + hyper['delta'] * jax.random.normal(subkey, shape=(K,))

key, subkey = jax.random.split(key)
rate, rhox, RHOX = sample(rhox, subkey, hyper['sample_stop'])
print('rate: {}'.format(rate))

rhox = RHOX[hyper['sample_start']::hyper['sample_step']].mean(axis=0)
rhox = rhox / np.abs(rhox).sum()
print('rhox: {}'.format(rhox))

res = dict()
res['rhox'] = rhox
with open(args.output, 'wb') as f:
    dill.dump(res, f)

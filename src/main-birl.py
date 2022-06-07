import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/irl.obj')
args = parser.parse_args()

hyper = dict()
hyper['sample_start'] = 10_000
hyper['sample_stop'] = 20_000
hyper['sample_step'] = 10
hyper['delta'] = .005

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 20

def _sample_like0(rhox, x, a):
    q = alpha * np.einsum('ij,j->i', x, rhox)
    return q[a] - logsumexp(q)

_sample_like0 = jax.vmap(_sample_like0, in_axes=(None,0,0))
_sample_like = lambda rhox: _sample_like0(rhox, data_x, data_a).sum()

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

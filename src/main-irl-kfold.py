import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/irl-kfold.obj')
parser.add_argument('-k', type=int, default=10)
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

TT = T // args.k

def _sample_like0(rho, x, a):
    q = alpha * np.einsum('ij,j->i', x, rho)
    return q[a] - logsumexp(q)
_sample_like0 = jax.vmap(_sample_like0)

def _sample_like(rhos):
    _rhos = rhos.repeat(TT, axis=0)
    return _sample_like0(_rhos, data_x, data_a).reshape(-1,TT).sum(axis=1)

def _sample0(rho, like, _rho, _like, key):
    cond = _like - like > np.log(jax.random.uniform(key))
    rho = jax.lax.select(cond, _rho, rho)
    like = jax.lax.select(cond, _like, like)
    return rho, like
_sample0 = jax.vmap(_sample0)

def _sample(arg0, arg1):
    (rhos, likes), key = arg0, arg1
    keys = jax.random.split(key, 2)
    _rhos = rhos + hyper['delta'] * jax.random.normal(keys[0], shape=(T//TT,K))
    _likes = _sample_like(_rhos)
    rhos, likes = _sample0(rhos, likes, _rhos, _likes, jax.random.split(keys[1], T//TT))
    return (rhos, likes), rhos

def sample(rhos, key, count):
    likes = _sample_like(rhos)
    (rhos, *_), RHOS = jax.lax.scan(_sample, (rhos, likes), jax.random.split(key, count))
    return rhos, RHOS
sample = jax.jit(sample, static_argnums=2)

key, subkey = jax.random.split(key)
rhos = -np.ones((T//TT,K))/K + hyper['delta'] * jax.random.normal(subkey, shape=(T//TT,K))

key, subkey = jax.random.split(key)
rhos, RHOS = sample(rhos, subkey, hyper['sample_stop'])

rhos = RHOS[hyper['sample_start']::hyper['sample_step']].mean(axis=0)
rhos = rhos.repeat(TT, axis=0)
rhos = rhos / np.abs(rhos).sum(axis=-1, keepdims=True)

res = dict()
res['rhos'] = rhos
with open(args.output, 'wb') as f:
    dill.dump(res, f)

import argparse
import dill
import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import tqdm

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/general.obj')
args = parser.parse_args()

hyper = dict()
hyper['sample_start'] = 10_000
hyper['sample_stop'] = 20_000
hyper['sample_step'] = 10
hyper['variance_rho'] = 5e-4
hyper['variance_beta'] = 5e-5
hyper['offset'] = -np.ones(8)/8

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 20

_cov_T = lambda t0, t1: np.minimum(t0, t1) + 1
_cov_T = jax.vmap(jax.vmap(_cov_T, in_axes=(None,0)), in_axes=(0,None))
cov_T = _cov_T(np.arange(T), np.arange(T))

_cov_K = np.eye(K)
_cov_K = _cov_K.at[:,0].set(np.ones(K))
_cov_K = _cov_K.at[:,0].set(_cov_K[:,0] / np.sum(_cov_K[:,0]**2)**.5)
for i in range(1,K):
    for j in range(i):
        _cov_K = _cov_K.at[:,i].add(-np.sum(_cov_K[:,i] * _cov_K[:,j]) * _cov_K[:,j])
    _cov_K = _cov_K.at[:,i].set(_cov_K[:,i] / np.sum(_cov_K[:,i]**2)**.5)
_scale = np.eye(K)
_scale = _scale.at[0,0].set(.1)
_cov_K = _cov_K @ _scale @ np.linalg.inv(_cov_K)
cov_K = _cov_K @ _cov_K.T

cov_rho = cov_K * hyper['variance_rho']
cov_rhos = np.kron(np.eye(T), cov_K) * hyper['variance_rho']
cov_betas = np.kron(cov_T, cov_K) * hyper['variance_beta']
invcov_rhos = np.linalg.inv(cov_rhos)
invcov_betas = np.linalg.inv(cov_betas)
mean_betas = hyper['offset'][None,...].repeat(T, axis=0).reshape(-1)
cov = np.linalg.inv(invcov_rhos + invcov_betas)
cov_at_invcov_betas_at_mean_betas = cov @ invcov_betas @ mean_betas
cov_at_invcov_rhos = cov @ invcov_rhos

cov_rho_L = np1.linalg.cholesky(cov_rho)
cov_L = np1.linalg.cholesky(cov)

def _sample_rhos_like(rho, x, a):
    q = alpha * np.einsum('ij,j->i', x, rho)
    return q[a] - logsumexp(q)

def _sample_rhos(beta, x, a, key):
    keys = jax.random.split(key, 3)
    rho = beta + cov_rho_L @ jax.random.normal(keys[0], shape=(K,))
    _rho = beta + cov_rho_L @ jax.random.normal(keys[1], shape=(K,))
    like = _sample_rhos_like(rho, x, a)
    _like = _sample_rhos_like(_rho, x, a)
    cond = _like - like > np.log(jax.random.uniform(keys[2]))
    return jax.lax.select(cond, _rho, rho)
_sample_rhos = jax.vmap(_sample_rhos)

def _sample_betas(rhos, key):
    mean = cov_at_invcov_betas_at_mean_betas + cov_at_invcov_rhos @ rhos.reshape(-1)
    _betas = mean + cov_L @ jax.random.normal(key, shape=(T*K,))
    return _betas.reshape(-1,K)

def _sample(arg0, arg1):
    (rhos, betas), key = arg0, arg1
    keys = jax.random.split(key, 2)
    rhos = _sample_rhos(betas, data_x, data_a, jax.random.split(keys[0], T))
    betas = _sample_betas(rhos, keys[1])
    return (rhos, betas), (rhos, betas)

def sample(rhos, betas, key, count):
    (rhos, betas), (RHOS, BETAS) = jax.lax.scan(_sample, (rhos, betas), jax.random.split(key, count))
    return rhos, betas, RHOS, BETAS
sample = jax.jit(sample, static_argnums=3)

rhos = np.zeros((T,K))
betas = np.zeros((T,K)) + hyper['offset']

BETAS = np.zeros((0,T,K))
for i in tqdm.tqdm(range(hyper['sample_stop'] // 200), unit_scale=200):
    key, subkey = jax.random.split(key)
    rhos, betas, _RHOS, _BETAS = sample(rhos, betas, subkey, 200)
    BETAS = np.concatenate((BETAS, _BETAS))

betas = BETAS[hyper['sample_start']::hyper['sample_step']].mean(axis=0)
betas = betas / np.abs(betas).sum(axis=-1, keepdims=True)

res = dict()
res['betas'] = betas
with open(args.output, 'wb') as f:
    dill.dump(res, f)

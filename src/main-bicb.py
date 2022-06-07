import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import tqdm

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/bandit.obj')
args = parser.parse_args()

hyper = dict()
hyper['sample_start'] = 1_000
hyper['sample_stop'] = 2_000
hyper['sample_step'] = 1
hyper['iter'] = 100

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 20
sigma = .10

_xs = lambda t0, t1: jax.lax.select(t1 <= t0, data_x[t1,data_a[t1]], np.zeros(K))
_xs = jax.vmap(jax.vmap(_xs, in_axes=(None,0)), in_axes=(0,None))
xs = _xs(np.arange(T-1), np.arange(T-1))

__betas_N = lambda t: np.einsum('i,j->ij', data_x[t,data_a[t]], data_x[t,data_a[t]])
__betas_N = jax.vmap(__betas_N)
_betas_N = __betas_N(np.arange(T-1)).cumsum(axis=0)
_betas_N = np.concatenate((np.zeros((K,K))[None,...], _betas_N))

__betas_y = lambda r, t: r * data_x[t,data_a[t]]
__betas_y = jax.vmap(__betas_y)
_betas_y = lambda rs: np.concatenate((np.zeros(K)[None,...], __betas_y(rs, np.arange(T-1)).cumsum(axis=0)))
_betas_y = jax.jit(_betas_y)
_BETAS_Y = jax.jit(jax.vmap(_betas_y))

@jax.jit
def decode(params):
    beta0 = np.exp(20 * params['beta0'])
    beta0_y = -np.ones(K)/K * beta0
    beta0_N = np.eye(K) * beta0
    return beta0_y, beta0_N

def _sample_rhos_like(rho, x, a):
    q = alpha * np.einsum('ij,j->i', x, rho)
    return q[a] - logsumexp(q)

def _sample_rhos(beta_mean, beta_cov, x, a, key):
    keys = jax.random.split(key, 3)
    rho = jax.random.multivariate_normal(keys[0], beta_mean, beta_cov)
    _rho = jax.random.multivariate_normal(keys[1], beta_mean, beta_cov)
    like = _sample_rhos_like(rho, x, a)
    _like = _sample_rhos_like(_rho, x, a)
    cond = _like - like > np.log(jax.random.uniform(keys[2]))
    return jax.lax.select(cond, _rho, rho)
_sample_rhos = jax.vmap(_sample_rhos)

def _sample_rs_init(rhox, key):
    mean = np.einsum('ij,j->i', xs[-1], rhox)
    rs = mean + sigma * jax.random.normal(key, shape=(T-1,))
    return rs

def _sample_rs(rhox, rhos, beta0_y, betas_invN, key):
    invcov = np.eye(T-1)
    invcov = invcov + np.einsum('abc,acd,aed->be', xs, betas_invN[1:], xs)
    invcov_at_mean = np.einsum('ij,j->i', xs[-1], rhox)
    invcov_at_mean = invcov_at_mean + np.einsum('ijk,ik->j', xs, rhos[1:] - np.einsum('ijk,k->ij', betas_invN[1:], beta0_y))
    cov = np.linalg.inv(invcov)
    mean = cov @ invcov_at_mean
    rs = jax.random.multivariate_normal(key, mean, cov * sigma**2)
    return rs

def _sample(arg0, arg1):
    (rhos, rs, rhox, beta0_y, betas_invN), key = arg0, arg1
    keys = jax.random.split(key, 2)
    betas_mean = np.einsum('ijk,ik->ij', betas_invN, beta0_y + _betas_y(rs))
    betas_cov = betas_invN * sigma**2
    rhos = _sample_rhos(betas_mean, betas_cov, data_x, data_a, jax.random.split(keys[0], T))
    rs = _sample_rs(rhox, rhos, beta0_y, betas_invN, keys[1])
    return (rhos, rs, rhox, beta0_y, betas_invN), (rhos, rs)

@jax.jit
def sample(rhox, beta0_y, beta0_N, key):
    keys = jax.random.split(key, 2)
    betas_invN = np.linalg.inv(beta0_N + _betas_N)
    rs = _sample_rs_init(rhox, keys[0])
    _, (_RHOS, _RS) = jax.lax.scan(_sample, (np.zeros((T,K)), rs, rhox, beta0_y, betas_invN), jax.random.split(keys[1], hyper['sample_stop']))
    RHOS = _RHOS[hyper['sample_start']::hyper['sample_step']]
    RS = _RS[hyper['sample_start']::hyper['sample_step']]
    return RHOS, RS

def compute_rhox(RS):
    _beta_y = _BETAS_Y(RS)[:,-1,:].mean(axis=0)
    _beta_N = _betas_N[-1]
    rhox = np.einsum('ij,j->i', np.linalg.inv(_beta_N), _beta_y)
    return rhox

def _likelihood0(rho, beta_mean, beta_invcov):
    res = -np.einsum('i,ij,j->', rho-beta_mean, beta_invcov, rho-beta_mean)
    res = res + np.log(np.linalg.det(beta_invcov))
    return res
_likelihood0 = jax.vmap(_likelihood0)

def _likelihood1(params, rhos, rs):
    beta0_y, beta0_N = decode(params)
    betas_y = beta0_y + _betas_y(rs)
    betas_N = beta0_N + _betas_N
    betas_invN = np.linalg.inv(betas_N)
    betas_mean = np.einsum('ijk,ik->ij', betas_invN, betas_y)
    betas_invcov = betas_N / sigma**2
    return _likelihood0(rhos, betas_mean, betas_invcov).sum()
_likelihood1 = jax.vmap(_likelihood1, in_axes=(None,0,0))

def likelihood(params, RHOS, RS):
    return _likelihood1(params, RHOS, RS).mean()

grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

###

rhox = -np.ones(K)/K

params = {'beta0': 0.}
grad_mnsq = {'beta0': 0.}
beta0_y, beta0_N = decode(params)

for i in tqdm.tqdm(range(hyper['iter'])):

    key, subkey = jax.random.split(key)
    RHOS, RS = sample(rhox, beta0_y, beta0_N, subkey)

    rhox = compute_rhox(RS)

    grad = grad_likelihood(params, RHOS, RS)
    grad_mnsq['beta0'] = .1 * grad['beta0']**2 + .9 * grad_mnsq['beta0']
    params['beta0'] += .001 * grad['beta0'] / (np.sqrt(grad_mnsq['beta0']) + 1e-8)
    beta0_y, beta0_N = decode(params)

    # print(rhox, beta0_N[0,0])

rhox = rhox / np.abs(rhox).sum()

res = dict()
res['rhox'] = rhox
res['beta0_y'] = beta0_y
res['beta0_N'] = beta0_N

key, subkey = jax.random.split(key)
_, RS = sample(rhox, beta0_y, beta0_N, subkey)

BETAS_Y = beta0_y + _BETAS_Y(RS)
betas_invN = np.linalg.inv(beta0_N + _betas_N)
betas_mean = np.einsum('ijk,lik->lij', betas_invN, BETAS_Y).mean(axis=0)
betas_cov = betas_invN * sigma**2

res['betas_mean'] = betas_mean
res['betas_cov'] = betas_invN

with open(args.output, 'wb') as f:
    dill.dump(res, f)

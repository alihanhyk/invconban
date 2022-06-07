import argparse
import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/ns-irl.obj')
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

TT = 10
KK = args.k

def sample_like(rho, x, a):
    q = alpha * np.einsum('ij,j->i', x, rho)
    return q[a] - logsumexp(q)
sample_like = jax.vmap(sample_like, in_axes=(None,0,0))
sample_like = jax.jit(sample_like)

def _sample(arg0, arg1):
    (rate, rho, like, xs, aa), key = arg0, arg1
    keys = jax.random.split(key, 2)
    _rho = rho + hyper['delta'] * jax.random.normal(keys[0], shape=(K,))
    _like = sample_like(_rho, xs, aa).sum()
    cond = _like - like > np.log(jax.random.uniform(keys[1]))
    rate = rate + cond
    rho = jax.lax.select(cond, _rho, rho)
    like = jax.lax.select(cond, _like, like)
    return (rate, rho, like, xs, aa), rho

def sample(rho, xs, aa, key, count):
    like = sample_like(rho, xs, aa).sum()
    (rate, rho, *_), RHO = jax.lax.scan(_sample, (0., rho, like, xs, aa), jax.random.split(key, count))
    return rate/count, rho, RHO
sample = jax.jit(sample, static_argnums=4)

###

rhos = [[None for _ in range(T//TT+1)] for i in range(T//TT)]
likes = [[[None for _ in range(T//TT+1)] for i in range(T//TT)] for _ in range(KK)]

print('ns-irl: computing rhos ...')

for i in range(T//TT):
    for j in range(i+1, T//TT+1):

        key, subkey = jax.random.split(key)
        _, _, RHO = sample(-np.ones(K)/K, data_x[i*TT:j*TT], data_a[i*TT:j*TT], subkey, hyper['sample_stop'])

        rho = RHO[hyper['sample_start']::hyper['sample_step']].mean(axis=0)
        rho = rho / np.abs(rho).sum()
        like = sample_like(rho, data_x[i*TT:j*TT], data_a[i*TT:j*TT]).sum()

        rhos[i][j] = rho
        likes[0][i][j] = like

print('ns-irl: computing likes ...')

for k in range(1,KK):
    for i in range(T//TT):
        for j in range(i+k+1, T//TT+1):
            likes[k][i][j] = max([likes[k-1][i][t] + likes[0][t][j] for t in range(i+k,j)])

print('ns-irl: computing change points ...')

ts = [None for _ in range(KK+1)]
ts[0] = 0
ts[KK] = T//TT
for k in reversed(range(1,KK)):
    max = None
    ind = None
    for t in range(k,ts[k+1]):
        if max is None or likes[k-1][0][t] + likes[0][t][ts[k+1]] > max:
            max = likes[k-1][0][t] + likes[0][t][ts[k+1]]
            ind = t
    ts[k] = ind

rhos = np.concatenate([rhos[i][j][None,...].repeat((j-i)*TT, axis=0) for i, j in zip(ts[:-1], ts[1:])], axis=0).reshape(T,K)
rhos = rhos / np.abs(rhos).sum(axis=-1, keepdims=True)

res = dict()
res['rhos'] = rhos
with open(args.output, 'wb') as f:
    dill.dump(res, f)

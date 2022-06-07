import argparse
import dill
import jax
import numpy as np
import pandas as pd
from scipy.special import logsumexp

parser = argparse.ArgumentParser()
parser.add_argument('--key', type=int, default=0)
args = parser.parse_args()

jax.config.update('jax_platform_name', 'cpu')
key = jax.random.PRNGKey(args.key)

df = pd.read_csv('data/liver-clean.csv')

df = df[(-5 < df.AGE) & (df.AGE < 5)]
df = df[(-5 < df.CREAT_TX) & (df.CREAT_TX < 5)]
df = df[(-5 < df.INR_TX) & (df.INR_TX < 5)]
df = df[(-5 < df.TBILI_TX) & (df.TBILI_TX < 5)]
df = df[(-5 < df.WGT_DIFF) & (df.WGT_DIFF < 5)]

X = df[df.columns.drop('SURVIVAL')].values
X = np.concatenate((np.ones((X.shape[0],1)),X), axis=-1)
y = df[['SURVIVAL']].values
rhox = np.ravel(np.linalg.inv(X.T @ X) @ X.T @ y)[1:]
rhox = rhox / np.abs(rhox).sum()

T = 500
A = 2
K = 8
alpha = 20
sigma = .10

key, subkey = jax.random.split(key)
inds = jax.random.choice(subkey, np.arange(df.index.size), shape=(T,A))
data_xs = df[df.columns.drop('SURVIVAL')].values[inds].reshape(T,A,K)
data_xs = [x for x in data_xs]

key1 = key
hyper = dict()
hyper['beta0_y'] = -np.ones(K)/K * (T//50)
hyper['beta0_N'] = np.eye(K) * (T//50)
hyper['stepping_tx'] = T//2
hyper['regressing_tx'] = T//2
hyper['regressing_gamma'] = 0

### agent0: stationary

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhox'] = rhox

for t, x in zip(range(T), data_xs):
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rhox)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))
    data['a'].append(a)

data['key'] = key
with open('data/agent0-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent1: sampling

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhox'] = rhox
data['rhos'] = list()
data['betas_mean'] = list()

beta_y = hyper['beta0_y']
beta_N = hyper['beta0_N']

for t, x in zip(range(T), data_xs):

    # belief
    key, subkey = jax.random.split(key)
    beta_mean = np.linalg.inv(beta_N) @ beta_y
    beta_cov = sigma**2 * np.linalg.inv(beta_N)
    rho = np.array(jax.random.multivariate_normal(subkey, beta_mean, beta_cov))

    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rho)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    # reward
    key, subkey = jax.random.split(key)
    r = np.sum(x[a] * rhox) + sigma * float(jax.random.normal(subkey))


    data['a'].append(a)
    data['rhos'].append(rho)
    data['betas_mean'].append(beta_mean)

    # belief update
    beta_y = beta_y + r * x[a]
    beta_N = beta_N + np.einsum('i,j->ij', x[a], x[a])

data['key'] = key
with open('data/agent1-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent11: optimistic

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhox'] = rhox
data['betas_mean'] = list()

beta_y = hyper['beta0_y']
beta_N = hyper['beta0_N']

for t, x in zip(range(T), data_xs):

    # belief
    beta_mean = np.linalg.inv(beta_N) @ beta_y
    beta_cov = sigma**2 * np.linalg.inv(beta_N)

    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, beta_mean) + np.einsum('ij,jk,ik->i', x, beta_cov, x)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    # reward
    key, subkey = jax.random.split(key)
    r = np.sum(x[a] * rhox) + sigma * float(jax.random.normal(subkey))

    data['a'].append(a)
    data['betas_mean'].append(beta_mean)

    # belief update
    beta_y = beta_y + r * x[a]
    beta_N = beta_N + np.einsum('i,j->ij', x[a], x[a])

data['key'] = key
with open('data/agent11-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent12: greedy

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhox'] = rhox
data['rhos'] = list()

beta_y = hyper['beta0_y']
beta_N = hyper['beta0_N']

for t, x in zip(range(T), data_xs):

    # belief
    rho = np.linalg.inv(beta_N) @ beta_y
    
    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rho)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    # reward
    key, subkey = jax.random.split(key)
    r = np.sum(x[a] * rhox) + sigma * float(jax.random.normal(subkey))

    data['a'].append(a)
    data['rhos'].append(rho)

    # belief update
    beta_y = beta_y + r * x[a]
    beta_N = beta_N + np.einsum('i,j->ij', x[a], x[a])

data['key'] = key
with open('data/agent12-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent2: stepping

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhos'] = list()

for t, x in zip(range(T), data_xs):

    # belief
    rho = -np.ones(K)/K if t < hyper['stepping_tx'] else rhox

    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rho)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    data['a'].append(a)
    data['rhos'].append(rho)

data['key'] = key
with open('data/agent2-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent21: linear

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhos'] = list()

for t, x in zip(range(T), data_xs):

    # belief
    rho = t/T * rhox + (1-t/T) * -np.ones(K)/K

    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rho)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    data['a'].append(a)
    data['rhos'].append(rho)

data['key'] = key
with open('data/agent21-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

### agent3: regressing

key = key1
data = dict()
data['x'] = data_xs
data['a'] = list()
data['rhos'] = list()

for t, x in zip(range(T), data_xs):

    # belief
    rho_gamma = hyper['regressing_gamma'] * rhox + (1-hyper['regressing_gamma']) * -np.ones(K)/K
    rho = (t/hyper['regressing_tx']) * rhox + (1-t/hyper['regressing_tx']) * -np.ones(K)/K if t < hyper['regressing_tx'] else ((t-hyper['regressing_tx'])/(T-hyper['regressing_tx'])) * rho_gamma + (1-(t-hyper['regressing_tx'])/(T-hyper['regressing_tx'])) * rhox

    # action
    key, subkey = jax.random.split(key)
    q = alpha * np.einsum('ij,j->i', x, rho)
    a = int(jax.random.choice(subkey, np.arange(A), p=np.exp(q-logsumexp(q))))

    data['a'].append(a)
    data['rhos'].append(rho)

data['key'] = key
with open('data/agent3-key{}.obj'.format(args.key), 'wb') as f:
    dill.dump(data, f)

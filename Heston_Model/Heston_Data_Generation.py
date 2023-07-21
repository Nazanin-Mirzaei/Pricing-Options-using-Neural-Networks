# Tools
#!pip install pyDOE tensorflow_addons
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.stats import norm
from pyDOE import lhs


def heston_price(m, v0, kappa, theta, sigma, rho, tau, r, a, b, N_cos):

    sum=0
    sum1=0

    moneyness = m - r* tau
    for k in range(0, N_cos):
      args = (m, v0, kappa, theta, sigma, rho, tau, r, a, b)
      res= heston_charfunc(k* np.pi/(b-a), *args)* U_k(k, a, b)* np.exp(-1j* k* np.pi* ((moneyness+ a)/(b-a)))
      if k==0 :
        sum1=res
      else:
        sum+=res

    result= np.exp(-r* tau)* np.real(0.5* sum1+ sum)

    return result

def heston_price(m, v0, kappa, theta, sigma, rho, tau, r, a, b, N_cos):

    sum=0
    sum1=0

    moneyness = m - r* tau
    for k in range(0, N_cos):
      args = (m, v0, kappa, theta, sigma, rho, tau, r, a, b)
      res= heston_charfunc(k* np.pi/(b-a), *args)* U_k_calc(k, a, b)* np.exp(-1j* k* np.pi* ((moneyness+ a)/(b-a)))
      if k==0 :
        sum1=res
      else:
        sum+=res


    result= np.exp(-r* tau)* np.real(0.5* sum1+ sum)


    return result

def heston_charfunc(w, m, v0, kappa, theta, sigma, rho, tau, r, a, b):

    args = (w, m, v0, kappa, theta, sigma, rho, tau, r, a, b)
    D= D_calc(*args)
    G= G_calc(*args)
    result= np.exp(1j* w* r* tau+ (v0/(sigma**2))* ((1- np.exp(-1* D* tau))/(1-G* np.exp(-1* D* tau)))
    * (kappa- (1j* rho* sigma* w)- D))* np.exp((kappa* theta/ (sigma**2))* (tau* (kappa- 1j* rho* sigma* w- D)-
                                                                                   2* np.log((1- G* np.exp(-1* D* tau))/(1-G))))

    return result

def D_calc(w, m, v0, kappa, theta, sigma, rho, tau, r, a, b):

    result= np.sqrt((kappa- 1j* rho* sigma* w)**2 + (w**2+ 1j* w)* (sigma**2))

    return result

def G_calc(w, m, v0, kappa, theta, sigma, rho, tau, r, a, b):

    args = (w, m, v0, kappa, theta, sigma, rho, tau, r, a, b)
    D= D_calc(*args)
    result= (kappa- 1j* rho* sigma* w- D )/ (kappa- 1j* rho* sigma* w+ D)

    return result

def U_k_calc(k, a, b):

    result= (2/(b-a))* (Chi_k(k, 0, b, a, b)- Psi_k(k, 0, b, a, b))

    return result

def Chi_k(k, c, d, a, b):

    result= (1/((1+ (k* np.pi/ (b-a))**2) ))* ((np.cos(k* np.pi * ((d-a)/(b-a)))* np.exp(d))- (np.cos(k* np.pi* ((c-a)/(b-a)))* np.exp(c))
    + ((k* np.pi/(b-a)) *np.sin(k* np.pi* ((d-a)/(b-a)))* np.exp(d))- ((k* np.pi/(b-a)) *np.sin(k* np.pi* ((c-a)/(b-a)))* np.exp(c) ))

    return result

def Psi_k(k, c, d, a, b):

    if k==0:
      result= d-c
    else:
      result=(np.sin(k* np.pi* ((d-a)/(b-a)))- np.sin(k* np.pi* ((c-a)/(b-a))))* (b-a)/ (k* np.pi)

    # print('say_x', result)
    return result

def calc_a_b(m, v0, kappa, theta, sigma, rho, tau, r, L_cos):

    c1= r* tau+ (1- np.exp(-1* kappa* tau))* ((theta- v0)/ (2* kappa))- 0.5* theta* tau

    c2= (1/(8* (kappa**3)))* (sigma* tau* kappa* np.exp(-1* kappa* tau)*(v0-theta)* (8* kappa* rho- 4* sigma)+
                            kappa* rho* sigma* (1- np.exp(-1* kappa* tau))* (16* theta- 8* v0)+
                            2* theta* kappa* tau* (-4* kappa* rho* sigma+ sigma**2+ 4* (kappa**2))+
                            sigma**2 * ((theta- 2*v0)* np.exp(-2* kappa* tau)+ theta* (6* np.exp(-1* kappa* tau)-7)+ 2* v0)+
                            8* (kappa**2) * (v0- theta)* (1-np.exp(-1* kappa * tau)))
    
    a= c1- L_cos* (np.sqrt(np.abs(c2)))
    b= c1+ L_cos* (np.sqrt(np.abs(c2)))

    return a, b

# Define the LHS method with lhs function in Python library pyDOE

def lhs_sample(range_values):
    lower_bound, upper_bound = range_values
    samples = lhs(1, samples=1)  # Generating a single sample
    scaled_sample = samples * (upper_bound - lower_bound) + lower_bound
    return scaled_sample.item()

# Create an empty data frame to store the values
df = pd.DataFrame(columns=['m', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'v0', 'option_price'])


num_samples = 1000000

# Define the ranges for input parameters
dataset = []
for i in range(num_samples):
    if i % 3000 == 0:
        print(i)
    m = lhs_sample([0.6, 1.4])
    tau = lhs_sample([0.1, 1.4])
    r = lhs_sample([0.0, 0.1])
    rho = lhs_sample([-0.95, 0.0])
    kappa = lhs_sample([0.0, 2.0])
    theta = lhs_sample([0.0, 0.5])
    sigma = lhs_sample([0.0, 0.5])
    v0 = lhs_sample([0.05, 0.5])

    a, b = calc_a_b(np.log(m), v0, kappa, theta, sigma, rho, tau, r, L_cos=50)
        #print(a, b)
    option_price = heston_price(np.log(m), v0, kappa, theta, sigma, rho, tau, r, a, b, N_cos=1500)
        # dataset.append([m, tau, r, rho, kappa, theta, sigma, v0, ])
        # Append the values to the data frame
    df = df.append({
        'm': m,
        'tau': tau,
        'r': r,
        'rho': rho,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'v0': v0,
        'option_price': option_price
    }, ignore_index=True)

with open('df_{}.pickle'.format(len(df)), 'wb') as f:
  pickle.dump(df, f)
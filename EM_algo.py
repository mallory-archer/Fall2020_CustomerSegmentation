import pandas as pd
import numpy as np
from scipy.stats import norm


def manual_fit_GMM(x, initial_vals):
    def calc_expectation(x, u1, std1, u0, std0, alpha):
        f0 = norm.pdf(x, loc=u0, scale=std0)
        f1 = norm.pdf(x, loc=u1, scale=std1)
        likelihood_i = (1 - alpha) * f0 + (alpha * f1)
        return (alpha * f1) / likelihood_i

    # maximization step
    def calc_maximization(x, gamma, u0_prev, u1_prev):
        u0 = sum((1 - gamma) * x) / sum(1 - gamma)
        u1 = sum(gamma * x) / sum(gamma)
        var0 = sum((1 - gamma) * ((x - u0_prev) ** 2)) / sum(1 - gamma)
        var1 = sum((gamma) * ((x - u1_prev) ** 2)) / sum(gamma)
        alpha = sum(gamma / len(x))
        return {'u0': u0, 'u1': u1, 'std0': max(var0 ** 0.5, 1e-6), 'std1': max(var1 ** 0.5, 1e-6), 'alpha': alpha}

    def calc_LL(x, u0, std0, u1, std1, alpha):
        f0 = norm.pdf(x, loc=u0, scale=std0)
        f1 = norm.pdf(x, loc=u1, scale=std1)
        likelihood_i = (1 - alpha) * f0 + (alpha * f1)
        return sum(np.log(likelihood_i))

    t_dict = initial_vals
    LL = calc_LL(x, u0=t_dict['u0'], std0=t_dict['std0'], u1=t_dict['u1'], std1=t_dict['std1'], alpha=t_dict['alpha'])

    df_EM_evolution = pd.DataFrame([t_dict])
    df_gamma_evolution = pd.DataFrame([None] * len(x))

    for i in range(0, 10):
        t_gamma = calc_expectation(x, u1=t_dict['u1'], std1=t_dict['std1'], u0=t_dict['u0'], std0=t_dict['std0'],
                                   alpha=t_dict['alpha'])
        t_dict = calc_maximization(x, t_gamma, u0_prev=t_dict['u0'], u1_prev=t_dict['u1'])
        t_LL = calc_LL(x, u0=t_dict['u0'], std0=t_dict['std0'], u1=t_dict['u1'], std1=t_dict['std1'],
                       alpha=t_dict['alpha'])
        df_EM_evolution = pd.concat([df_EM_evolution, pd.DataFrame([t_dict])])
        df_gamma_evolution = pd.concat([df_gamma_evolution, pd.Series(t_gamma)], axis=1)
        print('LL: % 3.6f' % (abs((t_LL - LL) / LL)))
        if abs((t_LL - LL) / LL) < 1e-6:
            break
        LL = t_LL

    return t_dict, t_gamma, LL, df_EM_evolution, df_gamma_evolution


# initialize params
proportion_split = 0.3
n = 1000
true_draws = 3 + 1.5 * norm.rvs(size=int(proportion_split*n))
false_draws = 10 + 0.75 * norm.rvs(size=int((1-proportion_split)*n))
x = pd.Series(np.concatenate((true_draws, false_draws)))
GMM_params, _, _, df_EM_evolution, df_gamma_evolution = manual_fit_GMM(x, initial_vals={'u0': 15, 'u1': 1, 'std0': 1,
                                                                                        'std1': 1, 'alpha': 0.25})
print(GMM_params)

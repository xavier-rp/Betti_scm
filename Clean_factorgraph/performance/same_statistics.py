import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.stats import chi2
from loglin_model import mle_2x2_ind


def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        print('HERE')
        return 0, 1
    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val


#def chisq_root(x, chi_square, N, a, b, c):

#    e_11 = (a + b)*(a + c) / N
#    e_12 = (a + b)*(b + x) / N
#    e_21 = (c + x)*(a + c) / N
#    e_22 = (c + x)*(b + x) / N

#    O_ij = np.array([[a, b], [c, x]])
#    E_ij = np.array([[e_11, e_12,], [e_21, e_22]])

#    return np.sum((O_ij-E_ij)**2/E_ij) - chi_square

def chisq_root(p, chi_square, N, b, c):

    x,y = p
    e_11 = (x + b)*(x + c) / N
    e_12 = (x + b)*(b + y) / N
    e_21 = (c + y)*(x + c) / N
    e_22 = (c + y)*(b + y) / N

    O_ij = np.array([[x, b], [c, y]])
    E_ij = np.array([[e_11, e_12,], [e_21, e_22]])

    return (np.sum((O_ij-E_ij)**2/E_ij) - chi_square, x + y + b + c - N)




if __name__ == '__main__':
    factor = 4

    model_table = np.array([[0, factor*20], [factor*15, 0]], dtype=np.float64)


    sol = fsolve(chisq_root, (factor*25, factor*40), args=(0.5, factor*100, model_table[0,1], model_table[1,0]))

    print(sol)

    model_table[0, 0] = sol[0]
    model_table[1, 1] = sol[1]

    print(model_table)
    print(np.sum(model_table))
    expected = mle_2x2_ind(model_table)
    print(expected)
    print(chisq_test(model_table, expected))

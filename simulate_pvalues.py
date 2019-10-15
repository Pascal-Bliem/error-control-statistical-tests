from scipy.stats import t, ttest_ind, norm
from statsmodels.stats.power import tt_ind_solve_power
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.multicomp
import numpy as np
import pandas as pd
from itertools import combinations
import pickle
import os


def simulate_pvalues(powers=np.arange(0,1.1,0.1)):
    """
    Simualte p-values from a t-test with powers ranging by 
    default from 0 to 1 in steps of 0.1, with 100 obervations 
    and 10000 simulated tests per power step.
    Returns the p-values as data frame.
    """
    
    
    if os.path.isfile("./data/pvalues.dat"):
        return
    
    all_pvalues=[]
    
    # for every desired power level
    for power in powers:
        pvalues=[]
        
        # no effects if power = 0, else use statsmodels  power solver to
        # calculate effect size corresponding  to desired power
        if power == 0:
            d = 0
        else:
            d = tt_ind_solve_power(effect_size=None, nobs1=100, alpha=0.05, 
                                   power=power, ratio=1.0, alternative='two-sided')
        
        # simulate 10000 samplings from 2 populations and t-tests
        for i in range(10000):
            a = np.random.normal(0,1,100)
            b = np.random.normal(d,1,100)
            t, p = ttest_ind(a,b)
            pvalues.append(p)
        
        all_pvalues.append(pvalues)
    
    pvalues = pd.DataFrame(np.array(all_pvalues).T,
                        columns=np.round(powers,decimals=2))
    # convert results to pandas data frame with power levels as column names
    
    pickle.dump(pvalues,open("./data/pvalues.dat","wb"))
    
    return pvalues


def simulate_multiple_test_pvalues():
    """
    This function simulates ANOVA tests with n factors
    with two levels each, considering 2-way interactions and
    1-way main effects so that number of tests per 
    ANOVA = sum(range(1,n+1)). Each situation is simulated 10000
    times, p-values are calculated and adjusted with Bonferroni 
    and Holm correction as well as false discovery rate.
    Returns dictionaries with p-values for each number of tests.
    """
    
    # check if files with simulation results already exist
    if (os.path.isfile("./data/pvalues_multi_uncor.dat")
        and os.path.isfile("./data/pvalues_multi_bonf.dat")
        and os.path.isfile("./data/pvalues_multi_holm.dat")
        and os.path.isfile("./data/pvalues_multi_fdr.dat")):
        return
    
    # dictionaries to store results
    pvalues_uncor = dict()
    pvalues_bonf = dict()
    pvalues_holm = dict()
    pvalues_fdr = dict()
    
    # for tests with n number of factors
    for n in [3,4,5,6,7]:
        
        # number of test considering all 2-way interactions and 1-way
        ntests = sum(range(1,n+1))
        
        # store p-values for this n
        pvalues = []
        pvalues_h = []
        pvalues_f = []
        
        # repeat the simulation 10000 times for each condition
        for i in range(10000):
            
            # a counter 
            if i % 1000 == 0:
                print(f"n: {n}, i: {i}")
            
            # create n categorical variabes with 2 levels each
            X = np.array(list(set(combinations(n*[0,1],n))))
            # create random data drawn from normal distribution
            y = np.random.normal(0,1,2**n).reshape(2**n,1)
            # put them together in data frame and give the columns char names
            df = pd.DataFrame(np.concatenate([X,y],axis=1),
                              columns=[chr(c) for c in range(97,97+n)]+["y"])
    
            # build a string as a formular for the OLS model
            # with all 2-way variable interactions 
            formular = "y ~ "
            for c in list(set(combinations([chr(c) for c in range(97,97+n)],2))):
                formular += f"C({c[0]})*C({c[1]})+"
            
            # fit an OLS model
            model = ols(formular[:-1],data=df).fit()
            # get the ANOVA p-values for the 2-way interaction terms and the 1-way terms
            pvs = sm.stats.anova_lm(model,type=2)["PR(>F)"].iloc[:-1].tolist()
            # append calculated p-values to list of all p-values
            for p in pvs:
                pvalues.append(p)
            
            # perform the Holm crrection
            # rank the p-values
            i = np.array(list(range(1,len(pvs)+1)))
            o = np.array(sorted(pvs))
            # multiply each p with its inverse rank
            # concat it with array of ones for comparison in next step
            p_or_one = np.concatenate([((len(pvs)-i+1)*o).reshape(-1,1),
                                       np.ones(len(pvs)).reshape(-1,1)],
                                       axis=1)
            # if calculated p-value larger than 1, set to 1
            pmin = np.min(p_or_one,axis=1)
            # set each p-value to the accumulated maximum
            # (meaning, if one of the previous p-values was larger, set current 
            # p-value to its value, see wikipedia article for the equation)
            pvs_h = np.maximum.accumulate(pmin)
            # append Holm-corrected p-values to list of all Holm-corrected p-values
            for p in pvs_h:
                pvalues_h.append(p)
                
            # perform the false discovery rate (Benjamini-Hochberg) correction
            # rank the p-values and order reverse
            i = np.array(list(range(len(pvs),0,-1)))
            o = np.array(sorted(pvs,reverse=True))
            # multiply each p with ntests divided by its rank
            # concat it with array of ones for comparison in next step
            p_or_one = np.concatenate([(o*len(pvs)/i).reshape(-1,1),
                                         np.ones(len(pvs)).reshape(-1,1)],
                                         axis=1)
            # if calculated p-value larger than 1, set to 1
            pmin = np.min(p_or_one,axis=1)
            # set each p-value to the accumulated minimum
            # (meaning, if one of the previous p-values was smaller,
            # set current p-value to its value
            pvs_f = np.minimum.accumulate(pmin)
            
            for p in pvs_f:
                pvalues_f.append(p)
            
            
        # apply the Bonferroni correction by multiplying each p-value
        # by the number of tests
        pvalues_bonf[ntests] = np.array(pvalues)*ntests
        pvalues_holm[ntests] = np.array(pvalues_h)
        pvalues_fdr[ntests] = np.array(pvalues_f)
        pvalues_uncor[ntests] = np.array(pvalues)
        
    # save the results to file
    pickle.dump(pvalues_uncor, open("./data/pvalues_multi_uncor.dat","wb"))
    pickle.dump(pvalues_bonf, open("./data/pvalues_multi_bonf.dat","wb"))
    pickle.dump(pvalues_holm, open("./data/pvalues_multi_holm.dat","wb"))
    pickle.dump(pvalues_fdr, open("./data/pvalues_multi_fdr.dat","wb"))
    
    return (pvalues_uncor, pvalues_bonf, pvalues_holm, pvalues_fdr)

def simulate_optional_stopping_pvalues():
    """
    This fuction simulates nsim studies where t-tests are performed 
    with nlooks during data collection and ppssible optional stopping 
    if significant p-value was found before the last look. The p-values
    are calculated unadjusted and with Pocock and O'Brien-Fleming (OBF)
    boundary corrections.
    """
    
    # check if files with simulation results already exist
    if (os.path.isfile("./data/pvalues_optstp_uncor.dat")
        and os.path.isfile("./data/pvalues_optstp_poc.dat")
        and os.path.isfile("./data/pvalues_optstp_obf.dat")):
        return
    
    # paramters for the simulation
    n = 100        # sample size
    alpha = 0.05   # overall significance level
    nsim = 100000  # number of  simulations
    d = 0          # effect size
    
    pvalues_uncor = dict()
    pvalues_poc = dict()
    pvalues_obf = dict()
    
    # each number of looks option
    for nlook in [2,4,5]:
        # dictionries with the corrected alphas for the Pocock and the OBF boundaries
        pocock_dict = {2:np.array([0.0294,0.0294]),
                       4:np.array([0.0182,0.0182,0.0182,0.0182]),
                       5:np.array([0.0158,0.0158,0.0158,0.0158,0.0158])}
        obf_dict = {2:np.array([0.0054,0.0492]),
                    4:np.array([0.00005,0.0039,0.0184,0.0412]),
                    5:np.array([0.000005,0.0013,0.0085,0.0228,0.0417])}
        
        # warn if n is not dividable by nlook without rest
        if n % nlook !=0:
            print("sample size should be dividable by number of looks, check results")
        
        # calculate idices for looks
        looks = np.arange(0,n+1,int(n/nlook))[np.arange(0,n+1,int(n/nlook))>2]
        
        # make matrix nsim x nlook for storing p-values
        p_mat = np.zeros([nsim,nlook])
        
        # for each simulation
        for i in range(nsim):
            # draw samples from normal distribution
            a = np.random.normal(0, 1, n)
            b = np.random.normal(d, 1, n)
            
            # perform t-test for each look
            for j,l in enumerate(looks):
                t, p_mat[i,j] = ttest_ind(a[:l+1],b[:l+1])   
            
        # create a matrix of Pocock-boundary-corrected p-values
        p_mat_pocock = np.zeros_like(p_mat)
        for l in range(nlook):
            p_mat_pocock[:,l] = p_mat[:,l] * (alpha/pocock_dict[nlook][l])
        
        # create a matrix of OBF-boundary-corrected p-values
        p_mat_obf = np.zeros_like(p_mat)
        for l in range(nlook):
            p_mat_obf[:,l] = p_mat[:,l] * (alpha/obf_dict[nlook][l])
        
        # get the error rate for each out of nlook looks
        err_looks = np.array([np.sum(p_mat[:,l]<alpha)/nsim 
                              for l in range(len(looks))])

        # if a p-value was below alpha at one of the looks,
        # get the first index as index for optional stopping,
        # if nothing below alpha, just take the last look
        opt_stop_idx = np.array([np.min(
                                 np.where(p_mat[i,:]<alpha)[0]
                                 ) 
                                 if len(np.where(p_mat[i,:]<alpha)[0])>0
                                 else len(looks)-1
                                 for i in range(nsim)])
        # get the optionally stopped (if applicable) pvalues for each simulation,
        # if ther was no stopping, its just the last p-value
        opt_stop_pvalues = np.array([p_mat[i,opt_stop_idx[i]] 
                                     for i in range(nsim)])
        # calculate overall error
        #err_all = np.sum(opt_stop_pvalues<alpha)/nsim
        
        # do the same steps as above but for the Pocock-corrected p-values
        opt_stop_idx_pocock = np.array([np.min(
                                        np.where(p_mat_pocock[i,:]<alpha)[0]
                                        ) 
                                        if len(np.where(p_mat_pocock[i,:]<alpha)[0])>0
                                        else len(looks)-1
                                        for i in range(nsim)])
        opt_stop_pvalues_pocock = np.array([p_mat_pocock[i,opt_stop_idx_pocock[i]] 
                                            for i in range(nsim)])
        #err_all_pocock = np.sum(opt_stop_pvalues_pocock<alpha)/nsim
        
        # do the same steps as above but for the obf-corrected p-values
        opt_stop_idx_obf = np.array([np.min(
                                     np.where(p_mat_obf[i,:]<alpha)[0]
                                     ) 
                                     if len(np.where(p_mat_obf[i,:]<alpha)[0])>0
                                     else len(looks)-1
                                     for i in range(nsim)])
        opt_stop_pvalues_obf = np.array([p_mat_obf[i,opt_stop_idx_obf[i]] 
                                         for i in range(nsim)])
        #err_all_obf = np.sum(opt_stop_pvalues_obf<alpha)/nsim

        # add p-values to respective dicts
        pvalues_uncor[nlook] = opt_stop_pvalues
        pvalues_poc[nlook] = opt_stop_pvalues_pocock
        pvalues_obf[nlook] = opt_stop_pvalues_obf
    
    # save the results to file
    pickle.dump(pvalues_uncor, open("./data/pvalues_optstp_uncor.dat","wb"))
    pickle.dump(pvalues_poc, open("./data/pvalues_optstp_poc.dat","wb"))
    pickle.dump(pvalues_obf, open("./data/pvalues_optstp_obf.dat","wb"))
    
    return (pvalues_uncor,pvalues_poc,pvalues_obf)

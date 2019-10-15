from scipy.stats import t, ttest_ind, norm
from statsmodels.stats.power import tt_ind_solve_power
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_style("darkgrid")
import panel as pn
from panel import widgets
pn.extension()
import simulate_pvalues

def plot_null_hypothesis():
    """ 
    This fucntion plots a normal distribution arounf zero
    representing the null hypothesis and an observation
    with corresponding p-value.
    """
    
    # switch off interactibe plotting to avoid double-plotting
    # when calling the function
    plt.ioff()
    
    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
    # calculate the null and alternative hypothesis distribution
    x = np.arange(-5,5.01,0.01)
    null_dist = norm.pdf(x=x,loc=0,scale=1)
    
    # plot the hypotheses distribution
    null = ax.plot(x, null_dist, "--k")
    
    # annotate the dist
    null_top = norm.pdf(x=0,loc=0,scale=1)
    h0txt = ax.annotate("$H_0$",[0,null_top*1.03],
                        horizontalalignment="center",fontsize=14)
    
    # make a observation data point and plot it
    obs = 0
    i_obs = np.argwhere(np.round(x,decimals=2)==1)[0][0]
    
    ax.plot(1, obs, ".",markerfacecolor="g",markersize=30,
            markeredgewidth=3,markeredgecolor="k"
            ,alpha=0.5,)
    ax.annotate("observation",[0.9,0.025],
                horizontalalignment="right",fontsize=12)
    
    # fill the p.value areas
    pvalue_fill = ax.fill_between(x=x[i_obs:],
                                  y1= np.zeros_like(x[i_obs:]),
                                   y2= null_dist[i_obs:],
                                   alpha=.2, color = "r",
                                   label="probability of \ndata >= observation")
    
    
    ax.legend(frameon=False,fontsize=12)
    ax.set_ylim([-0.05,0.45])
    ax.set_ylabel("probability density",fontsize=12)
    ax.set_xlabel("observed effect",fontsize=12)
    ax.set_title("Probability of data under $H_0$",fontsize=14)
    
    
    return fig

def plot_significance_level():
    """ 
    This fucntion plots a normal distribution around zero
    representing the null hypothesis and and shaded areas 
    corrsponding to the significance level alpha = 0.05.
    """
    
    # switch off interactibe plotting to avoid double-plotting
    # when calling the function
    plt.ioff()
    
    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
    # calculate the null and alternative hypothesis distribution
    x = np.arange(-5,5.01,0.01)
    null_dist = norm.pdf(x=x,loc=0,scale=1)
    
    # plot the hypotheses distribution
    null = ax.plot(x, null_dist, "--k")
    
    # annotate the dist
    null_top = norm.pdf(x=0,loc=0,scale=1)
    h0txt = ax.annotate("$H_0$",[0,null_top*1.03],
                        horizontalalignment="center",fontsize=14)
    
    # get upper and lower significance boundary for alpha=0.05
    i_hi = np.argwhere(np.round(x,decimals=2)==1.96)[0][0]
    i_lo = np.argwhere(np.round(x,decimals=2)==-1.96)[0][0]
    
    # fill the significance boundary areas
    between_fill = ax.fill_between(x=x[i_lo:i_hi+1],
                              y1= np.zeros_like(x[i_lo:i_hi+1]),
                              y2= null_dist[i_lo:i_hi+1],
                              alpha=.2, color = "w",
                              label="retain $H_0$")
    
    hi_fill = ax.fill_between(x=x[i_hi:],
                              y1= np.zeros_like(x[i_hi:]),
                              y2= null_dist[i_hi:],
                              alpha=.2, color = "r",
                              label="reject $H_0$")

    lo_fill = ax.fill_between(x=x[:i_lo+1],
                              y1= np.zeros_like(x[:i_lo+1]),
                              y2= null_dist[:i_lo+1],
                              alpha=.2, color = "r")
    
    # annotate 
    ax.annotate("95%",[0,0.2],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12)
    
    ax.annotate("$\\alpha}$/2",[-3,0.05],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12)
    
    ax.annotate("$\\alpha}$/2",[3,0.05],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12)
    
    ax.legend(frameon=False,fontsize=12)
    ax.set_ylim([-0.05,0.45])
    ax.set_ylabel("probability density",fontsize=12)
    ax.set_xlabel("observed effect",fontsize=12)
    ax.set_title("Significance level $\\alpha$ = 0.05",fontsize=14)
    
    
    return fig


def plot_hypotheses_params():
    """wrapper offering interactivity with panel widgits"""
    
    def plot_hypotheses(d=0.5,n=30,alpha=0.05):
        """
        This fucntion plots two normal distributions (null and alternative
        hypotheses) with a difference in means of d and standard deviations 
        given by standard errors of 1/sqrt(n), where n is the number of 
        observations. Areas corrsponding to the false positive (type 1), 
        false negative (type 2) error rates, and the power will be shaded.
        
        Parameters:
        -----------
            d: float
               Standardized difference in means (e.g. Cohen's d).
               Default is 0.5.
            n: integer
               number of observations per group
               Default is 30.
        alpha: float
               Significance level alpha.
               Default is 0.05.
               
        Returns:
        ----------
          fig: matplotlib figure
               The complete figure.
               
        """
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
        
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # calculate the null and alternative hypothesis distribution
        x = np.arange(-5,5.01,0.01)
        null_dist = norm.pdf(x=x,loc=0,scale=1/np.sqrt(n))
        alt_dist = norm.pdf(x=x,loc=d,scale=1/np.sqrt(n))
        
        # calculate critical values and index corresponding to significance level alpha
        crit_hi = np.round(norm.ppf(1-alpha/2,loc=0,scale=1/np.sqrt(n)),decimals=2)
        i_hi = np.argwhere(np.round(x,decimals=2)==crit_hi)[0][0]
        crit_lo = np.round(norm.ppf(alpha/2,loc=0,scale=1/np.sqrt(n)),decimals=2)
        i_lo = np.argwhere(np.round(x,decimals=2)==crit_lo)[0][0]
        
        # get distribution tops
        null_top = norm.pdf(x=0,loc=0,scale=1/np.sqrt(n))
        alt_top = norm.pdf(x=d,loc=d,scale=1/np.sqrt(n))
        
        # plot the hypotheses distribution
        null = ax.plot(x, null_dist, "--k")
        alt = ax.plot(x, alt_dist,"-k")
        
        # fill the error and power areas
        alpha_hi_fill = ax.fill_between(x=x[i_hi:],
                                       y1= np.zeros_like(x[i_hi:]),
                                       y2= null_dist[i_hi:],
                                       alpha=.2, color = "r",
                                       label="False pos. $\\alpha$")
        
        alpha_lo_fill = ax.fill_between(x=x[:i_lo+1],
                                       y1= np.zeros_like(x[:i_lo+1]),
                                       y2= null_dist[:i_lo+1],
                                       alpha=.2, color = "r")
        
        trueneg_fill = ax.fill_between(x=x[i_lo:i_hi+1],
                                       y1= np.zeros_like(x[i_lo:i_hi+1]),
                                       y2= null_dist[i_lo:i_hi+1],
                                       alpha=.2, color = "w",
                                       label="True neg. 1- $\\alpha$")
        
        beta_fill = ax.fill_between(x=x[:i_hi+1],
                                       y1= np.zeros_like(x[:i_hi+1]),
                                       y2= alt_dist[:i_hi+1],
                                       alpha=.3, color = sns.color_palette("deep")[0], 
                                       label="False neg. $\\beta$")
        
        power_fill = ax.fill_between(x=x[i_hi:],
                                       y1= np.zeros_like(x[i_hi:]),
                                       y2= alt_dist[i_hi:],
                                       alpha=.2, color = "g", 
                                       label="True pos. 1- $\\beta$")
        
        # display the effect size d as a line between the dists and text
        dline = ax.errorbar(x=[0,d],y=[null_top*1.12,null_top*1.12],
                    yerr=[null_top/50,null_top/50],
                    color="k", lw = 0.8)
        dtxt = ax.annotate(f"$d={d:.1f}$",[0+d/2,null_top*1.15],
                           horizontalalignment="center",fontsize=12)
        
        
        # annotate the dists
        h0txt = ax.annotate("$H_0$",[0,null_top*1.03],
                            horizontalalignment="center",fontsize=14)
        hatxt = ax.annotate("$H_a$",[d,alt_top*1.03],
                            horizontalalignment="center",fontsize=14)
        
        # set axes limits
        _ = ax.set_xlim([0-5/np.sqrt(n),d+5/np.sqrt(n)])
        _ = ax.set_ylim([-0.05,null_top*1.25])
        _ = ax.set_yticklabels([])
        
        # annotate the error rates and power
        beta = norm.cdf(crit_hi,loc=d,scale=1/np.sqrt(n))
        power = 1-beta
        errtxt = ax.text(0.22,0.78,"$\\alpha$: {:5.1f}%\n\
                                    $\\beta$: {:5.1f}%\n\
                                    Power: {:5.1f}%\n\
                                    n: {:9}".format(alpha*100,beta*100,power*100,n),
                                    transform=ax.transAxes,fontsize=12,
                                    horizontalalignment="right")
        # create legend
        _ = ax.legend(fontsize=12,loc="upper right")
    
        return fig
    p = pn.interact(plot_hypotheses,d=np.arange(0,1.51,0.1),
                n=np.arange(5,1001,1),
                alpha=np.arange(0.001,0.1001,0.001),)
    return pn.Column(pn.Column("###Parameter cotrols",p[0],align="center"),p[1])

def plot_pvalues_with_power():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues.dat"):
        pvalues = simulate_pvalues.simulate_pvalues()
    else:
        with open("./data/pvalues.dat","rb") as file:
            pvalues = pickle.load(file)
    
    def plot_pvalues(power = 0.0):
        """
        This fucntion plots a histogram of (precomputed) simulated 
        p-values for different values of power of the corresponding
        statistical tests.
        
        Parameters:
        -----------
            power: float in range[0,1]
                   The power of the statistical tests.
                   Default is 0.0.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired power level
        sns.distplot(pvalues[power],bins=np.arange(0,1.05,0.05),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        hline = ax.hlines(y=10000*0.05, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1000,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,10500])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"10000 simulated p-values for a power of {power}",fontsize=16)
        # color the first bar red
        ax.get_children()[2].set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,power=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])
    
    
    
def plot_pvalues_multiple_uncor():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_multi_uncor.dat"):
        pvalues = simulate_pvalues.simulate_pvalues()
    else:
        with open("./data/pvalues_multi_uncor.dat","rb") as file:
            pvalues = pickle.load(file)
    
    def plot_pvalues(ntests = 6):
        """
        This fucntion plots a histogram of (precomputed) simulated 
        p-values for different number of multiple statistical tests
        to show how the false positive rate is inflated.
        
        Parameters:
        -----------
           ntests: int
                   The number of the statistical tests performed.
                   Default is 6.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[ntests],bins=np.arange(0,1.1,0.05),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=10000*0.05, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1000,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,10500])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"10000 simulated studies with {ntests} tests each",fontsize=14)
        # color the first bar red
        ax.get_children()[2].set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,ntests=[6,10,15,21,28])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])

def plot_pvalues_multiple_bonf():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_multi_bonf.dat"):
        pvalues = simulate_pvalues.simulate_pvalues()
    else:
        with open("./data/pvalues_multi_bonf.dat","rb") as file:
            pvalues = pickle.load(file)
    
    for v in list(pvalues.values()):
        v[v>1.0]=1.0
        
    def plot_pvalues(ntests = 6):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        Bonferroni-corrected p-values for different number of 
        multiple statistical tests to show how the false positive rate 
        is controlled by the Bonferroni correction.
        
        Parameters:
        -----------
           ntests: int
                   The number of the statistical tests performed.
                   Default is 6.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[ntests],bins=np.arange(0,1.05,0.05),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=10000*0.05, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1000,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,10500])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"10000 simulated Bonferroni-corrected studies with {ntests} tests each",fontsize=14)
        # color the first bar red
        ax.get_children()[2].set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,ntests=[6,10,15,21,28])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])
  
    
def plot_pvalues_multiple_holm():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_multi_holm.dat"):
        pvalues = simulate_pvalues.simulate_pvalues()
    else:
        with open("./data/pvalues_multi_holm.dat","rb") as file:
            pvalues = pickle.load(file)
        
    def plot_pvalues(ntests = 6):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        Holm-corrected p-values for different number of 
        multiple statistical tests to show how the false positive rate 
        is controlled by the Holm correction.
        
        Parameters:
        -----------
           ntests: int
                   The number of the statistical tests performed.
                   Default is 6.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[ntests],bins=np.arange(0,1.05,0.05),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=10000*0.05, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1000,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,10500])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"10000 simulated Holm-corrected studies with {ntests} tests each",fontsize=14)
        # color the first bar red
        ax.get_children()[2].set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,ntests=[6,10,15,21,28])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])
  
    
def plot_pvalues_multiple_fdr():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_multi_fdr.dat"):
        pvalues = simulate_pvalues.simulate_pvalues()
    else:
        with open("./data/pvalues_multi_fdr.dat","rb") as file:
            pvalues = pickle.load(file)
        
    def plot_pvalues(ntests = 6):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        Holm-corrected p-values for different number of 
        multiple statistical tests to show how the false positive rate 
        is controlled by the false discovery rate.
        
        Parameters:
        -----------
           ntests: int
                   The number of the statistical tests performed.
                   Default is 6.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[ntests],bins=np.arange(0,1.05,0.05),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=10000*0.05, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1000,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,10500])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"10000 simulated FDR-corrected studies with {ntests} tests each",fontsize=14)
        # color the first bar red
        ax.get_children()[2].set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,ntests=[6,10,15,21,28])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])
  
    
def plot_pvalues_over_time():
    """wrapper offering interactivity with panel widgits"""
    
    def plot_pvalues_samplesize(n=200, d=0.0):
        """
        This fucntion plots p-values from t-tests as a function of sample
        size and for different effect size to show how statistical significance
        can be reached after a certain sample size (if there is an effect).
            
        Parameters:
        -----------
            d: float
               Effect size, standardized difference in means (e.g. Cohen's d).
               Default is 0.0.
            n: integer
               sample size, number of observations per group
               Default is 200.
                       
        Returns:
        ----------
            fig: matplotlib figure
                 The complete figure.
           
        """
            
        # crate empty arrays to store data and p-values
        a = np.zeros(n)
        b = np.zeros(n)
        pvalues = np.zeros(n)
        
        # sample a value for every i in range 1 to n and
        # perform a t-test, collect the p-value
        for i in range(n):
            a[i] = np.random.normal(0.0,1.0,1)
            b[i] = np.random.normal(d,1.0,1)
            t, pvalues[i] = ttest_ind(a[:i+1],b[:i+1])
    
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()  
        # make a figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        # plot the p-values as a function of n
        line = sns.lineplot(range(10,n),pvalues[10:n],ax=ax)
        # plot and annotate dashed line showing alpha = 0.05
        hline = ax.hlines(y=0.05, xmin=0, xmax=n, 
                          colors="red", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=0.12,
                s="$\\alpha$=0.05",fontsize=14,
                horizontalalignment="right",
                transform=ax.transAxes)
        # display lowest p-values as text
        ax.text(x=0.95, y=0.88,
                s=f"lowest p-value={np.min(pvalues[10:n]):.3f}",fontsize=14,
                horizontalalignment="right",
                transform=ax.transAxes)
    
        # set axes limits and labels
        ax.set_xlim([10,n])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlabel("sample size $n$",fontsize=14)
        ax.set_ylabel("p-value",fontsize=14)
        ax.set_title(f"P-value as a function of sample size at effect size d={d:.1f}",fontsize=14)
        
        return fig        
    
    p = pn.interact(plot_pvalues_samplesize,
    n=np.arange(20,2001,10,dtype=np.int),
    d=np.arange(0.0,1.6,0.1,dtype=np.float))
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])        

def plot_pvalues_optional_stopping_uncor():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_optstp_uncor.dat"):
        pvalues, _, _ = simulate_pvalues.simulate_optional_stopping_pvalues()
    else:
        with open("./data/pvalues_optstp_uncor.dat","rb") as file:
            pvalues = pickle.load(file)
        
    def plot_pvalues(nlook = 2):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        uncorrected p-values for different number of looks for 
        optionnal stopping tests to show how the false positive rate 
        is inflated by optional stopping without correction.
        
        Parameters:
        -----------
           nlook: int
                  The number of looks performed per study.
                  Default is 2.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[nlook],bins=np.arange(0,1.01,0.01),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=100000*0.01, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1200,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # display the false positive rate
        err = np.sum(pvalues[nlook]<0.05)/len(pvalues[nlook])
        ax.text(x=0.95, y=0.9, s=f"False positive rate = {err:.3f}",
                transform=ax.transAxes,fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,4000])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"100000 simulated studies with {nlook} looks each",fontsize=14)
        # color the first bar red
        for c in ax.get_children()[2:7]:
            c.set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,nlook=[2,4,5])
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])


def plot_pvalues_optional_stopping_pocock():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_optstp_poc.dat"):
       _, pvalues, _ = simulate_pvalues.simulate_optional_stopping_pvalues()
    else:
        with open("./data/pvalues_optstp_poc.dat","rb") as file:
            pvalues = pickle.load(file)
        
    def plot_pvalues(nlook = 2):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        uncorrected p-values for different number of looks for 
        optionnal stopping tests to show how the false positive rate 
        is controlled by optional stopping with Pocock boundary correction.
        
        Parameters:
        -----------
           nlook: int
                  The number of looks performed per study.
                  Default is 2.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        pvalues[nlook][pvalues[nlook]>1.0]=1.0
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[nlook],bins=np.arange(0,1.01,0.01),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=100000*0.01, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1200,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # display the false positive rate
        err = np.sum(pvalues[nlook]<0.05)/len(pvalues[nlook])
        ax.text(x=0.95, y=0.9, s=f"False positive rate = {err:.3f}",
                transform=ax.transAxes,fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,4000])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"100000 simulated Pocock-corrected studies with {nlook} looks each",fontsize=14)
        # color the first bar red
        for c in ax.get_children()[2:7]:
            c.set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,nlook=[2,4,5])   
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])


def plot_pvalues_optional_stopping_obf():
    """wrapper offering interactivity with panel widgits"""
    
    # load or simulate p-values
    if not os.path.isfile("./data/pvalues_optstp_obf.dat"):
       _, _, pvalues = simulate_pvalues.simulate_optional_stopping_pvalues()
    else:
        with open("./data/pvalues_optstp_obf.dat","rb") as file:
            pvalues = pickle.load(file)
        
    def plot_pvalues(nlook = 2):
        """
        This fucntion plots a histogram of (precomputed) simulated,
        uncorrected p-values for different number of looks for 
        optionnal stopping tests to show how the false positive rate 
        is controlled by optional stopping with O'brien-Fleming
        boundary correction.
        
        Parameters:
        -----------
           nlook: int
                  The number of looks performed per study.
                  Default is 2.
        Returns:
        -----------
              fig: matplotlib figure
                   The final figure.
        """
        # the pvalues data frame has to exist in the nonlocal envrirnonment
        nonlocal pvalues
        
        pvalues[nlook][pvalues[nlook]>1.0]=1.0
        
        # switch off interactibe plotting to avoid double-plotting
        # when used together with panel interactive widgets
        plt.ioff()
         
        # make figure
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        
        # plot the histogram for the desired number of tests
        sns.distplot(pvalues[nlook],bins=np.arange(0,1.01,0.01),
                     kde=False, ax=ax)
        
        # a line showing significance level alpha
        ax.vlines(x=0.05, ymin=0, ymax=1000, colors="red",
                  linestyles='--', transform=ax.transAxes, 
                  linewidth=1.5)
        ax.text(x=0.06, y=0.9, s="$\\alpha$ = 0.05",
                transform=ax.transAxes,fontsize=14)
        
        # a line showing expected distribution under null hypothesis
        # if the error rate was controlled properly
        hline = ax.hlines(y=100000*0.01, xmin=0, xmax=1, 
                          colors="black", linestyles='--',
                          linewidth=1.5)
        ax.text(x=0.95, y=1200,
                s="under $H_0$",fontsize=14,
                horizontalalignment="right")
        
        # display the false positive rate
        err = np.sum(pvalues[nlook]<0.05)/len(pvalues[nlook])
        ax.text(x=0.95, y=0.9, s=f"False positive rate = {err:.3f}",
                transform=ax.transAxes,fontsize=14,
                horizontalalignment="right")
        
        # set up axes ranges and labels
        ax.set_xlim([0,1])
        ax.set_ylim([0,4000])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_xlabel("p-value",fontsize=14)
        ax.set_ylabel("count",fontsize=14)
        ax.set_title(f"100000 simulated OBF-corrected studies with {nlook} looks each",fontsize=14)
        # color the first bar red
        for c in ax.get_children()[2:7]:
            c.set_color("lightcoral")
        
        #return the figure
        return fig
    p = pn.interact(plot_pvalues,nlook=[2,4,5])   
    
    return pn.Column(pn.Column(p[0],align="center"),p[1])
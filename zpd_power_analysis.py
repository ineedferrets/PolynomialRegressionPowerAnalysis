from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

import csv

from matplotlib import pyplot as plt

import scipy.stats as stt
from scipy.signal import savgol_filter

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import seaborn as sns

#### Global variables that will remain constant throughout analysis ####

# Empty class for holding parameters to functions
class Params:
    None
# Maximum reached score difference
max_score_delta = 0.6

# Zone of proximal development/Peak of polynomial
zpd = 0.2

# Distribution of amateur videos
am_a, am_b = (0 - 0.241) / 0.065226, (100 - 0.241) / 0.065226
am_dist = stt.truncnorm(am_a, am_b, loc=0.241, scale=0.065226)

# Distribution of intermediate videos
in_a, in_b = (0 - 0.44375) / 0.109799, (1 - 0.44375) / 0.109799
in_dist = stt.truncnorm(in_a, in_b, loc=0.44375, scale=0.109799)

# Distribution of expert videos
ex_a, ex_b = (0 - 0.777) / 0.074989, (1 - 0.777) / 0.074989
ex_dist = stt.truncnorm(ex_a, ex_b, loc=0.777, scale=0.074989) 


#### Functions for analysis ####

def learnerVideoScoreToScoreImprovement(x, effect_size):
    '''
    Map learner video score difference to a score improvement based upon a polynomial function.
    
    Keyword arguments:
    x -- Learner video score
    effect_size -- Non-standardized effect size.
    '''
    return (-effect_size * (x - zpd) ** 2) + max_score_delta

def olsRegression(x, y, formula):
    '''
    Perform and ordinary least squares regression using the data and formula provided.
    
    Keyword arguments:
    x, y -- Dependent and independent data (must be equal in length).
    formula -- Formula to give the statsmodels.ols. For more information please read the statsmodels.formula.api documentation.
    
    Returns:
    results -- A statsmodels.ols results object.
    '''
    df = pd.DataFrame(columns=['y','x'])
    df['x'] = x
    df['y'] = y
    results = smf.ols(formula=formula, data=df).fit()
    return results

def playerVideoDistributionSample(min_value, max_value, num_of_samples=1):
    '''
    Randomly generate pre-intervention scores through a uniform distribution and subtract that from the sum of three
    randomly sampled videos (distribution of videos given by mean and standard deviation of video groups).
    
    Keyword arguments:
    min_value, max_value -- Minimum and maximum values.
    num_of_samples -- Number of samples to generate.
    
    Returns:
    return_samples -- A numpy array of all samples.    
    '''
    _mean = 0.2
    _std = 0.1
    range_value = max_value - min_value
    pre_scores = stt.uniform.rvs(size=num_of_samples) * range_value + min_value
    
    return_samples = []
    
    for i in range(0, num_of_samples):
        if i % 3 == 0:
            return_samples.append(am_dist.rvs() - pre_scores[i])
        elif i % 3 == 1:
            return_samples.append(in_dist.rvs() - pre_scores[i])
        else:
            return_samples.append(ex_dist.rvs() - pre_scores[i])
    return np.array(return_samples)
    

def generateRegressionData(params):
    """
    Generate data and perform a regression.
    
    Keyword arguments:
    params -- Options for the statistical test. Example given below.
    
    Returns:
    x, y -- Dependent and independent variable as a list of values.
    results -- A statsmodels.ols results object.
    """
    # Generate random samples from a distribution for X values (step 1)
    x = params.dep_distribution(params.value_range['min'], params.value_range['max'], params.current_sample_size)
    
    # Use X values, ground truth function, and some randome noise to generate values for y (steps 2 & 3)
    y = params.indep_dep_relation(x, params.effect_size) + np.random.normal(0, params.residual_std, params.current_sample_size)
    min_adj_value = params.value_range['min'] - params.value_range['max']
    max_adj_value = params.value_range['max'] - params.value_range['min']
    y = [max(min_adj_value, min(max_adj_value, z)) for z in y]

    # Fit curve and evaluate F-test p-value (step 4)
    results = params.model(x, y, params.formula)
    
    return x, y, results

def powerAnalysisPolynomialRegression(params):
    """Perform a power analysis of some statistical test.
    
    Keyword arguments:
    params -- Options for the statistical test. Example given below.
    
    Returns:
    sample_data -- A panda dataframe containing the raw data of all 'experiments' as well as the results objects and
                   significance values of all 'experiments'.
    """    
    sample_data = {}

    i_range = tqdm(params.sample_size_range, desc="Sample Size") if params.show_progress_bar else params.sample_size_range
    for i in i_range:
        sample_data[i] = {
            'Raw': {},
            'StatResults': [],
            'SigValues': []
        }
        sample_data[i]['Raw']['x'] = {}
        sample_data[i]['Raw']['y'] = {}

        significance_sum = 0

        for j in range(0, params.num_samples):
            params.current_sample_size = i
            sample_data[i]['Raw']['x'][j], sample_data[i]['Raw']['y'][j], results = generateRegressionData(params)
            
            sample_data[i]['StatResults'].append(results)
            sample_data[i]['SigValues'].append(results.f_pvalue)
            # Record whether test was significant
            significance_sum += 1 if results.f_pvalue <= params.significance else 0

        sample_data[i]['SigPercentage'] = significance_sum / params.num_samples * 100
        
    return sample_data

def plotPowerAnalysisData(sample_data, params):
    """
    Plot the sample data provided given the params set.
    """
    ground_truth = {}
    ground_truth['x'] = np.arange(-1, 1.01, 0.01)
    ground_truth['y'] = learnerVideoScoreToScoreImprovement(ground_truth['x'], 0.01)

    # Plot important information
    fig1, axs1 = plt.subplots(1,1)
    fig1.set_size_inches(10,5)
    
    fig2, axs2 = plt.subplots(2,1, sharex=True)
    fig2.set_size_inches(10, 10)
    fig1.suptitle("Residual $\sigma$ = " + str(params.residual_std)
                  + " and effect size ($b_0$) = " + str(params.effect_size) + ".")
    
    #axs1.set_title("Sample Curves vs Ground Truth")
    axs2[0].set_title("Significance spread")
    axs2[1].set_title("Power of tests at different sample sizes")

    bw_plot_data = {}
    bw_plot_data['sample_size'] = []
    bw_plot_data['significance_value'] = []
    br_plot_data = {}
    br_plot_data['sample_size'] = []
    br_plot_data['significance_percentage'] = []
    plot_colors = []

    for i in params.sample_size_range:
        rgb = np.random.rand(3,)
        plot_colors.append(rgb)

        for j in range(0,1):
            _,iv_l,iv_u = wls_prediction_std(sample_data[i]['StatResults'][j])
            # Smooth out confidence intervals (data is lost so is purely visual)
            iv_l_hat = savgol_filter(iv_l, len(iv_l)-1, 2)
            iv_u_hat = savgol_filter(iv_u, len(iv_u)-1, 2)

            x = np.linspace(-1, 1, len(iv_l))

            # Plot confidence, line, and values of sample
            p = sample_data[i]['StatResults'][j].params
            axs1.fill_between(x, iv_u_hat, iv_l_hat, facecolor=(rgb), alpha=0.3)
            axs1.plot(x, p[0] + p[1] * x + p[2] * (x ** 2), color=(rgb), alpha = 0.7)
            axs1.scatter(sample_data[i]['Raw']['x'][j], sample_data[i]['Raw']['y'][j],
                             color=(rgb), alpha=0.4, label="n = " + str(i))
        
        for j in range(0,params.num_samples):
            bw_plot_data['sample_size'].append(i)
            bw_plot_data['significance_value'].append(sample_data[i]['SigValues'][j])
        
        br_plot_data['sample_size'].append(i)
        br_plot_data['significance_percentage'].append(sample_data[i]['SigPercentage'])

    axs1.plot(ground_truth['x'], ground_truth['y'], 'r', label="Ground Truth")
    axs1.legend(loc='best')
    axs1.set_xlabel('$\Delta LV$')
    axs1.set_ylabel('$\Delta SI$')

    bw_df = pd.DataFrame(data = bw_plot_data)
    sns.boxplot(x='sample_size', y='significance_value', order=params.sample_size_range, ax=axs2[0], data=bw_df)
    axs2[0].set_ylabel('p')
    axs2[0].set_xlabel(None)
    
    br_df = pd.DataFrame(data = br_plot_data)
    sns.barplot(x='sample_size', y='significance_percentage', order=params.sample_size_range, ax=axs2[1], data=br_df)
    axs2[1].set_xlabel('Sample size (n)')
    axs2[1].set_ylabel('Percentage significance from ' + str(params.num_samples) + ' tests')

# Messy as fuck, but optimised so I don't generate samples for all sample sizes, just enough to find the lowest
# size that meets the power provided
def findLowestSampleSizeMeetingPower(params):  
    data = pd.DataFrame(columns=params.residual_std_range, index=params.effect_size_range, data=None)
    for std in tqdm(params.residual_std_range, desc="Working through std"):
        params.residual_std = std
        for es in tqdm(params.effect_size_range, desc="Working through es" ):
            params.effect_size = es
            for i in params.sample_size_range:
                sum_of_significant = 0
                params.current_sample_size = i
                for j in range(0,params.num_samples):
                    _, _, results = generateRegressionData(params)
                    sum_of_significant += 1 if results.f_pvalue <= params.significance else 0
                if sum_of_significant / params.num_samples * 100 >= params.target_power:
                    data[params.effect_size][params.residual_std] = i
                    break            
    return data



#### Functions for illustration ####

def plotDependentVariableSampling(num_of_samples):
    x = playerVideoDistributionSample(0, 1, num_of_samples)
    plt.hist(x, density=True, bins = 60)
    plt.ylabel('Probability')
    plt.xlabel('$\Delta S_{V-PV}$')
    plt.title('Histogram of random samples taken from the dependent variable sampling method')

def plotIndependentVariableMapping(num_of_samples):  
    # Generate x data
    x = playerVideoDistributionSample(0.0, 1.0, num_of_samples)
    
    # Map x to y
    y = learnerVideoScoreToScoreImprovement(x, 0.1) + np.random.normal(0, 0.1, num_of_samples)
    y = [max(-1.0, min(1.0, z)) for z in y]

    # Get ground truth
    x_truth = np.arange(-1.0, 1.0, 0.01)
    y_truth = learnerVideoScoreToScoreImprovement(x_truth, 0.1)

    plt.plot(x, y, 'bo')
    plt.plot(x_truth, y_truth, 'r-')
    plt.ylabel('$\Delta S_{Post-Pre}$')
    plt.xlabel('$\Delta S_{V-PV}$')
    plt.title('Effect size of 0.001 and residual standard deviation of 30')

def plotStatisticalTest(num_of_samples):
    params = Params()
    params.model = olsRegression
    params.formula = 'y ~ x + I(x**2)'
    params.dep_distribution = playerVideoDistributionSample
    params.indep_dep_relation = learnerVideoScoreToScoreImprovement
    params.value_range = {'min': 0.0, 'max': 1.0}
    params.effect_size =  0.1
    params.residual_std = 0.1
    params.current_sample_size = 1000
    
    # Generate x, y, and regression data
    x, y, results = generateRegressionData(params)

    # Get ground truth
    x_truth = np.arange(-1.0, 1.0, 0.01)
    y_truth = learnerVideoScoreToScoreImprovement(x_truth, 0.1)

    # Get regresssion function
    x_reg = np.arange(-1.0, 1.0, 0.01)
    y_reg = results.params[0] + results.params[1] * x_reg + results.params[2] * (x_reg ** 2)

    plt.plot(x, y, 'bo', alpha=0.2)
    plt.plot(x_reg, y_reg, 'b-', label="Regression result")
    plt.plot(x_truth, y_truth, 'r-', label="Ground truth")
    plt.ylabel('$\Delta S_{Post-Pre}$')
    plt.xlabel('$\Delta S_{V-PV}$')
    plt.legend(loc='best')
    plt.title('Effect size of 0.1 and residual standard deviation of 0.1')

    results.summary()

def calculateSignificanceOfSampleSize(sample_size, num_of_experiments, significance):
    params = Params()
    params.model = olsRegression
    params.formula = 'y ~ x + I(x**2)'
    params.dep_distribution = playerVideoDistributionSample
    params.indep_dep_relation = learnerVideoScoreToScoreImprovement
    params.value_range = {'min': 0.0, 'max': 1.0}
    params.effect_size =  0.1
    params.residual_std = 0.1
    params.current_sample_size = 1000
    
    num_of_significant_results = 0

    for i in tqdm(range(0,num_of_experiments), desc="Running experiments"):
        x, y, results = generateRegressionData(params)
        if results.f_pvalue <= significance:
            num_of_significant_results += 1

    print("After " + str(num_of_experiments) + " experiments for sample size " + str(sample_size) + " and a significance of "
     + str(significance) + ", it appears that " + str(num_of_significant_results/num_of_experiments * 100)
     + "% were significant.")
    print("This is for an effect size of " + str(params.effect_size) + " and residual standard deviation of "
     + str(params.residual_std))

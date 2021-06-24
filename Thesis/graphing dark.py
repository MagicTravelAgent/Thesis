import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
import matplotlib

cl1040 = pd.read_csv("Clark_dark_data_10_40.csv") 
cl2080 = pd.read_csv("Clark_dark_data_20_80.csv")
cl30120 = pd.read_csv("Clark_dark_data_30_120.csv") 
cl40160 = pd.read_csv("Clark_dark_data_40_160.csv") 
cl50200 = pd.read_csv("Clark_dark_data_50_200.csv") 
cl60240 = pd.read_csv("Clark_dark_data_60_240.csv") 
cl70280 = pd.read_csv("Clark_dark_data_70_280.csv") 
cl80320 = pd.read_csv("Clark_dark_data_80_320.csv") 
cl90360 = pd.read_csv("Clark_dark_data_90_360.csv")
cl100400 = pd.read_csv("Clark_dark_data_100_400.csv") 
cl110440 = pd.read_csv("Clark_dark_data_110_440.csv")
cl120480 = pd.read_csv("Clark_dark_data_120_480.csv") 
cl200800 = pd.read_csv("Clark_dark_data_200_800.csv") 
cl3001200 = pd.read_csv("Clark_dark_data_300_1200.csv")
cl4001600 = pd.read_csv("Clark_dark_data_400_1600.csv")
cl5002000 = pd.read_csv("Clark_dark_data_500_2000.csv")


box_data = {"10, 40": cl1040['time until adapted'],
            "20, 80": cl2080['time until adapted'],
            "30, 120": cl30120['time until adapted'],
            "40, 160": cl40160['time until adapted'],
            "50, 200": cl50200['time until adapted'],
            "60, 240": cl60240['time until adapted'],
            "70, 280": cl70280['time until adapted'],
            "80, 320": cl80320['time until adapted'],
            "90, 360": cl90360['time until adapted'],
            "100, 400": cl100400['time until adapted'],
            "110, 440": cl110440['time until adapted'],
            "120, 480": cl120480['time until adapted'],
            "200, 800": cl200800['time until adapted'],
            "300, 1200": cl3001200['time until adapted'],
            "400, 1600": cl4001600['time until adapted'],
            "500, 2000": cl5002000['time until adapted'],
}

df = pd.DataFrame(box_data)

a = [[x]*100 for x in [(i*40) for i in range(1,13)]]

a.append([800]*100)
a.append([1200]*100)
a.append([1600]*100)
a.append([2000]*100)
a = np.array(a).flatten().tolist()
b = np.array(list(box_data.values())).flatten()

condata = {"x" : a,
           "y" : b}

smalldata = {"x" : a[0:1200],
           "y" : b[0:1200]}

sns.lineplot(x = condata['x'], y = condata['y'], err_style="bars")

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Priors against adaptation time")
plt.xlabel("Prior expectation of food 2")
plt.ylabel("Time until adapted (iterations)")
plt.show()

def lin(data_in):
    formula_string = "y ~ x"

    model = sm.formula.ols(formula = formula_string, data = data_in)
    model_fitted = model.fit()


    print("R squared:",model_fitted.rsquared)
    print("Formula: y =",model_fitted.params[0],"+ x *",model_fitted.params[1])
    pred = []
    for i in data_in["x"]:
      pred.append(model_fitted.params[0] + i * model_fitted.params[1])
    
    sns.scatterplot(x = data_in['x'], y = data_in['y'])
    sns.lineplot(data_in["x"], pred)
    plt.title("Priors against adaptation time")
    plt.xlabel("Prior expectation of food level three")
    plt.ylabel("Time until adapted (iterations)")
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12,6.75)
    fig.savefig('question2.png', dpi=100)
    plt.show()
    plt.show()
    return(pred)

def plotter(data_in):
    for i in (range(0,len(new_pred))):
      resid.append(data_in['y'][i] - new_pred[i])
    
    sns.residplot(data_in['y'], resid)
    plt.title("Residuals from the regression model")
    plt.xlabel("x-value of the model")
    plt.ylabel("Distance from the model")
    plt.show()
    
    sns.distplot(new_pred, bins=20)
    plt.show()
    
    scipy.stats.probplot(new_pred, dist="norm", plot=plt)
    plt.show()
    
new_pred = lin(condata)
resid = []
plotter(condata)
print(np.mean([np.std(x) for x in box_data.values()]))

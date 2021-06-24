import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


#
cl1040 = pd.read_csv("Clark_light_data_10_40.csv") 
cl2080 = pd.read_csv("Clark_light_data_20_80.csv")
cl30120 = pd.read_csv("Clark_light_data_30_120.csv") 
cl40160 = pd.read_csv("Clark_light_data_40_160.csv") 
cl50200 = pd.read_csv("Clark_light_data_50_200.csv") 
cl60240 = pd.read_csv("Clark_light_data_60_240.csv") 
cl70280 = pd.read_csv("Clark_light_data_70_280.csv") 
cl80320 = pd.read_csv("Clark_light_data_80_320.csv") 
cl90360 = pd.read_csv("Clark_light_data_90_360.csv")
cl100400 = pd.read_csv("Clark_light_data_100_400.csv") 
cl110440 = pd.read_csv("Clark_light_data_110_440.csv")
cl120480 = pd.read_csv("Clark_light_data_120_480.csv") 
cl200800 = pd.read_csv("Clark_light_data_200_800.csv") 
cl3001200 = pd.read_csv("Clark_light_data_300_1200.csv")
cl4001600 = pd.read_csv("Clark_light_data_400_1600.csv")
cl5002000 = pd.read_csv("Clark_light_data_500_2000.csv")
fl100 = pd.read_csv("Friston_light_data_100.csv")

#sns.distplot(cl1040['amount of times swapped'], label = "Clark Agent 10, 40")
#sns.distplot(cl2080['amount of times swapped'], label = "Clark Agent 20, 80")
#sns.distplot(cl30120['amount of times swapped'], label = "Clark Agent 30, 120")
#sns.distplot(cl40160['amount of times swapped'], label = "Clark Agent 40, 160")
#sns.distplot(cl50200['amount of times swapped'], label = "Clark Agent 50, 200")
#sns.distplot(cl60240['amount of times swapped'], label = "Clark Agent 60, 240")
#sns.distplot(cl70280['amount of times swapped'], label = "Clark Agent 70, 280")
#sns.distplot(cl80320['amount of times swapped'], label = "Clark Agent 80, 320")
#sns.distplot(cl90360['amount of times swapped'], label = "Clark Agent 90, 360")
#sns.distplot(cl100400['amount of times swapped'], label = "Clark Agent 100, 400")
#sns.distplot(cl110440['amount of times swapped'], label = "Clark Agent 110, 440")
#sns.distplot(cl120480['amount of times swapped'], label = "Clark Agent 120, 480")
#sns.distplot(cl200800['amount of times swapped'], label = "Clark Agent 200, 800")
#sns.distplot(cl3001200['amount of times swapped'], label = "Clark Agent 300, 1200")
#sns.distplot(cl4001600['amount of times swapped'], label = "Clark Agent 400, 1600")
#sns.distplot(cl5002000['amount of times swapped'], label = "Clark Agent 500, 2000")

#sns.distplot(fl100['amount of times swapped'], label = "Friston Agent")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.title("Distributions of the amount of swaps per 100 iterations")
#plt.xlabel("Amount of swaps per 100 iterations")
#plt.ylabel("Probability")
plt.show()

#print(df['amount of times swapped'].mean())

box_data = {"10, 40": cl1040['amount of times swapped'],
            "20, 80": cl2080['amount of times swapped'],
            "30, 120": cl30120['amount of times swapped'],
            "40, 160": cl40160['amount of times swapped'],
            "50, 200": cl50200['amount of times swapped'],
            "60, 240": cl60240['amount of times swapped'],
            "70, 280": cl70280['amount of times swapped'],
            "80, 320": cl80320['amount of times swapped'],
            "90, 360": cl90360['amount of times swapped'],
            "100, 400": cl100400['amount of times swapped'],
            "110, 440": cl110440['amount of times swapped'],
            "120, 480": cl120480['amount of times swapped'],
            "200, 800": cl200800['amount of times swapped'],
            "300, 1200": cl3001200['amount of times swapped'],
            "400, 1600": cl4001600['amount of times swapped'],
            "500, 2000": cl5002000['amount of times swapped'],
#            "Friston Agent": fl100['amount of times swapped']}
            }
df = pd.DataFrame(box_data)
#sns.boxplot(data = df)

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

fl_x = [40]*100
fl2 = [2000]*100
fl_x = fl_x + fl2
fl_y = list(fl100['amount of times swapped']) + list(fl100['amount of times swapped'])


sns.lineplot(x = condata['x'], y = condata['y'], err_style="bars")
sns.lineplot(x = fl_x, y = fl_y)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Swaps per 100 iterations for each agent")
plt.xlabel("Prior observations of food level three")
plt.ylabel("Amount of swaps per 100 iterations")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12,6.75)
fig.savefig('question1.png', dpi=100)
plt.show()

#sns.lineplot(data=condata)

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.title("Boxplot for the amount of swaps per 100 iterations")
#plt.xlabel("Amount of swaps per 100 iterations")
#plt.ylabel("Amount of swaps")
#plt.show()
#
#print(stats.ttest_ind(fl100['amount of times swapped'], cl5002000['amount of times swapped'], alternative="greater", equal_var=False))
#print(fl100['amount of times swapped'].mean(), cl5002000['amount of times swapped'].mean())
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
var = 0
df_list = []
station_dir = 'gsom-latest/'
landcode ="DA"
yearly = True
monthly = False

loops = 0
for station in os.listdir(station_dir):
    if station.startswith(landcode):
        loops+=1

for station in os.listdir(station_dir):
    if station.startswith(landcode):
        df_list.append(pd.read_csv(os.path.join(station_dir, station)))
        var+=1
        print(str(np.round(var/loops*100,2))+"% done. "+str(var)+" out of "+str(loops)+" stations imported")
        
df = pd.concat(df_list, ignore_index=True,axis=0)

liste = []
dates = df["DATE"].drop_duplicates()
years = pd.Series(np.array(dates,dtype="<U4"))
df = df.assign(year=years)
years = years.drop_duplicates()

#%%
if yearly:
    for year in years:
        liste.append([year,float(np.mean(df[df["year"]==year]["TAVG"])),float(np.mean(df[df["year"]==year]["TMIN"])),float(np.mean(df[df["year"]==year]["TMAX"]))])
    
    liste = pd.DataFrame(liste)
    liste = liste.sort_values(by=[0]).reset_index(drop=True)
    

if monthly:
    for date in dates:
        liste.append([date,float(np.mean(df[df["DATE"]==date]["TAVG"])),float(np.mean(df[df["DATE"]==date]["TMIN"])),float(np.mean(df[df["DATE"]==date]["TMAX"]))])
    
    liste = pd.DataFrame(liste)
    liste = liste.sort_values(by=[0]).reset_index(drop=True)


#%%
x = np.arange(0,len(liste[0]),1)
y = liste[1].to_numpy(dtype=float)
index = np.isnan(y)==False
x = x[index].reshape(-1, 1)
y = y[index].reshape(-1, 1)
model = LinearRegression().fit(x,y)

Y = model.coef_*np.arange(0,len(liste[0]),1)+model.intercept_

plt.rcParams.update({"text.usetex": True, "font.family": "Sarif"})
fig, ax = plt.subplots(figsize=(14, 10))

plt.plot(liste[2],color="steelblue",label="Average minimum of temperatures")
plt.plot(liste[1],color="black",label="Average temperatures")
plt.plot(liste[3],color="firebrick",label="Average maximum of temperatures")
plt.plot(np.arange(0,len(liste[0]),1),Y[0,:],color="deeppink",linewidth=3,label="Linear regression of average temperatures")
tic = np.linspace(0,len(liste[2])-1,8).astype(int)
plt.xticks(tic,labels = np.array(liste[0][tic]),size=15)
plt.yticks(size=15)
plt.xlabel("Year",size=30)
plt.ylabel("yearly avg temperature [degree C$^\circ$]",size=30)
plt.legend(fontsize=15,loc="upper left")
plt.text(0.95, 0.95,"Data optained from https://www.ncei.noaa.gov/cdo-web/",
                     size=15,
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform = ax.transAxes,backgroundcolor='1',alpha=1)
plt.grid(True)





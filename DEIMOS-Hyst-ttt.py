# -*- coding: utf-8 -*-
"""
FR
Permet de normaliser, recentrer sur 0, combiner et corriger des cycles MH issus de DEIMOS-SOLEIL
Un numéro de scan en entrée (avec un paramètre Tz sans incidence sur les calculs),
deux scans combinés, sortie: Hc, H, cycle TEY et cycle Fluo. Les 2 scans doivent être
successifs, un dans un sens, l'autre dans le sens opposé

EN
Allow the import, normalisation, centring and merging of hysteresis cycles from DEIMOS-SOLEIL
One scan number to give (with a Tz parameter without incidence on calculations), return Hc, H, TEY and
Fluo cycles. The two scans must be successive. One in one direction, the other in the opposite direction
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import numpy_indexed as npi  # pip install numpy-indexed


""" -------- function find nearest value in array --------------- """
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
""" ----------------------------------------------------------- """



DebutScan=626
Tz=36




"""--------------- import of data ---------------"""
fichier = 'scan_'+ '%03i'%(DebutScan)+'.txt'
data = np.genfromtxt('./2020-02-12/'+fichier, skip_header=1)
field=data[:,1]
itio=data[:,7] # TEY
ifio=data[:,11] # fluo
energy=data[0,0] # to detect the edge

fichierb = 'scan_'+ '%03i'%(DebutScan+1)+'.txt'
datab = np.genfromtxt('./2020-02-12/'+fichierb, skip_header=1)
fieldb=datab[:,1]
itiob=datab[:,7]
ifiob=datab[:,11]
""" ----------------------------------------------------------- """



"""--------------------- edge detection ---------------------"""
Seuil='unknown edge'
if abs(energy-705)<10:
    Seuil='Fe'
elif abs(energy-775)<10:
    Seuil='Co'
""" ----------------------------------------------------------- """



"""----- raw data + crop NaN from file 1 + centring-----"""
itio = itio[np.logical_not(np.isnan(itio))] # remove NaN values
ifio = ifio[np.logical_not(np.isnan(ifio))]
field = field[np.logical_not(np.isnan(field))]
moyitio=itio.mean()
moyifio=ifio.mean()
itio = np.array([x-moyitio for x in itio]) # centring on 0
ifio = np.array([x-moyifio for x in ifio])

NbPts=len(itio)
index=np.linspace(1, NbPts, NbPts)
""" ----------------------------------------------------------- """




"""----- raw data + crop NaN from file 2 + centring-----"""
itiob = itiob[np.logical_not(np.isnan(itiob))] # remove NaN values
ifiob = ifiob[np.logical_not(np.isnan(ifiob))]
fieldb = fieldb[np.logical_not(np.isnan(fieldb))]
moyitiob=itiob.mean()
moyifiob=ifiob.mean()
itiob = np.array([x-moyitiob for x in itiob]) # centring on 0
ifiob = np.array([x-moyifiob for x in ifiob])

NbPtsb=len(itiob)
indexb=np.linspace(1, NbPtsb, NbPtsb)
""" ----------------------------------------------------------- """





"""----- Deglitch + interpolation + normalisation curve 1 -----"""
for i in range(1,NbPts-1):                                      # detection of glitched points
        if (abs(itio[i]-itio[i+1])>1e-6 and abs(field[i])<0.5) : #1e-6
            itio[i]=float('NaN')
        if (abs(ifio[i]-ifio[i+1])>0.02 and abs(field[i])<0.55) : #0.003
            ifio[i]=float('NaN')

maxitio=max(abs(itio))
maxifio=max(abs(ifio))
itio = np.array([x/maxitio for x in itio]) # normalisation
ifio = np.array([x/maxifio for x in ifio])

wTEY = np.isnan(itio) # put a weight of 0 to glitched points
itio[wTEY]=0.
wFluo = np.isnan(ifio)
ifio[wFluo]=0.

splTEY = UnivariateSpline(index, itio, w=~wTEY) # weighted interpolation
splFluo = UnivariateSpline(index, ifio, w=~wFluo)
splTEY.set_smoothing_factor(0.0008) # smoothing factor of the interpolation (0.0008)
splFluo.set_smoothing_factor(0.0008)

itio1=splTEY(index) # itio1 as array from the interpolated function
ifio1=splFluo(index)
""" ----------------------------------------------------------- """




"""----- Deglitch + interpolation + normalisation curve 2 -----"""
for i in range(1,NbPtsb-1):
        if (abs(itiob[i]-itiob[i+1])>1e-6 and abs(fieldb[i])<0.5) :
            itiob[i]=float('NaN')
        if (abs(ifiob[i]-ifiob[i+1])>0.02 and abs(fieldb[i])<0.55) :
            ifiob[i]=float('NaN')

maxitiob=max(abs(itiob))
maxifiob=max(abs(ifiob))
itiob = np.array([x/maxitiob for x in itiob])
ifiob = np.array([x/maxifiob for x in ifiob])

wTEYb = np.isnan(itiob)
itiob[wTEYb]=0.
wFluob = np.isnan(ifiob)
ifiob[wFluob]=0.

splTEYb = UnivariateSpline(indexb, itiob, w=~wTEYb)
splFluob = UnivariateSpline(indexb, ifiob, w=~wFluob)
splTEYb.set_smoothing_factor(0.0008)  # 0.0008
splFluob.set_smoothing_factor(0.0008)  # 0.0008

itio1b=splTEYb(indexb)
ifio1b=splFluob(indexb)

itio1b=-itio1b # return
ifio1b=-ifio1b
""" ----------------------------------------------------------- """





"""
merging and plot
"""
yt = np.concatenate([itio1,itio1b]) # TEY
xt = np.concatenate([index,indexb]) # must use the index (otherwise, with the two MH branches, the is 0)
Index, Itio = npi.group_by(xt).mean(yt) # mean of Y (TEY signal) at each X (index)

yf = np.concatenate([ifio1,ifio1b]) # fluo
xf = np.concatenate([index,indexb])
Index, Ifio = npi.group_by(xf).mean(yf)

y2 = np.concatenate([field,fieldb]) # H field
x2 = np.concatenate([index,indexb])
x2_unique, Field = npi.group_by(x2).mean(y2)

""" TEY """
plt.plot(field, itio1, 'r', ms=5)
plt.plot(fieldb, itio1b, 'b', lw=1)
plt.plot(Field, Itio, 'g', lw=1)
plt.axis([-6, 6, -1, 1])
plt.annotate('scan 1', xy=(0,1), xytext=(-5,0.8), color="r")
plt.annotate('scan 2', xy=(0,1), xytext=(-5,0.7), color="b")
plt.annotate('mean', xy=(0,1), xytext=(-5,0.6), color="g")
plt.annotate('TEY', xy=(0,1), xytext=(4.5,0.01), color="black")
plt.annotate('Tz='+str(Tz), xy=(0,1), xytext=(4.5,-0.1), color="black")
plt.annotate(Seuil, xy=(0,1), xytext=(4.5,-0.2), color="black")
plt.grid(True)
plt.show()

""" fluo """
plt.plot(field, ifio1, 'r', ms=5)
plt.plot(fieldb, ifio1b, 'b', lw=1)
plt.plot(Field, Ifio, 'g', lw=1)
plt.axis([-6, 6, -1, 1])
plt.annotate('scan 1', xy=(0,1), xytext=(-5,0.8), color="r")
plt.annotate('scan 2', xy=(0,1), xytext=(-5,0.7), color="b")
plt.annotate('mean', xy=(0,1), xytext=(-5,0.6), color="g")
plt.annotate('Fluo', xy=(0,1), xytext=(4.5,0.01), color="black")
plt.annotate('Tz='+str(Tz), xy=(0,1), xytext=(4.5,-0.1), color="black")
plt.annotate(Seuil, xy=(0,1), xytext=(4.5,-0.2), color="black")
plt.grid(True)
plt.show()
""" ----------------------------------------------------------- """


"""----- find coercive field -----"""
Itio0=find_nearest(Itio, 0) # value of Itio nearest of 0
indexItio0=np.where(Itio == Itio0) # index of Itio0
HcTEY=abs(Field[indexItio0]) # TEY coercive field

Ifio0=find_nearest(Ifio, 0)
indexIfio0=np.where(Itio == Itio0)
HcFluo=abs(Field[indexIfio0]) # fluo coercive field

Hc=(HcTEY+HcFluo)/2

print('coercive field:')
print('Hc='+str(Hc))
print(str(Tz)+'\t'+str(Hc[0]))
""" ----------------------------------------------------------- """



"""
write .txt file
"""
# f=open('./output/Cycle-'+ Seuil +'-Tz%3.1f'%Tz+'.txt','w+')
# f.write('H \t cycle TEY \t cycle fluo \n')
# f.write('  \t   \t   \t   \t   \n')
# f.write('H ('+Seuil+' edge), Hc='+str(Hc)+' \t Tz='+str(Tz)+' \t Tz='+str(Tz)+' \n')
# for i in range(0,NbPts):
#     f.write('%.6E \t %.6E \t %.6E \n' %(Field[i], Itio[i], Ifio[i]))
# f.close()

""" ----------------------------------------------------------- """



"""
To import in the same Origin workbook:
    create new workbook
    file > import > multiple ASCII files > import options > import mode: start new columns
    

copy and paste the following lines in the Origin console to automatically have
H fields as X on the common workbook:

wks.col1.type = 4;
wks.col4.type = 4;
wks.col7.type = 4;
wks.col10.type = 4;
wks.col13.type = 4;
wks.col16.type = 4;
wks.col19.type = 4;
wks.col22.type = 4;
wks.col25.type = 4;
wks.col28.type = 4;
wks.col31.type = 4;
wks.col34.type = 4;
wks.col37.type = 4;
wks.col40.type = 4;
wks.col43.type = 4;
wks.col46.type = 4;
wks.col49.type = 4;
wks.col52.type = 4;
wks.col55.type = 4;
wks.col58.type = 4;
wks.col61.type = 4;
wks.col64.type = 4;
wks.col67.type = 4;
wks.col70.type = 4;
wks.col73.type = 4;
wks.col76.type = 4;
wks.col79.type = 4;
wks.col82.type = 4;

    
"""

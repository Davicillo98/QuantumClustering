import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from scipy import stats
from scipy.optimize import curve_fit
def gaus(x,a,sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))



Chunk = 1000
Lista=[]
ruta_wfm = "/h/drodgon1/qChaos/WF/wfm_2P_Nstates_101__Np_249__Nwells_2__K_0__g_0.txt"


for chunk in pd.read_csv(ruta_wfm, chunksize=Chunk, delim_whitespace=True, header=None):
    Lista.append(chunk)
    
df_wfm = pd.concat(Lista)
df_wfm = df_wfm.to_numpy()
df_wfm = np.matrix(df_wfm)



etiquetas=[]
normal_nonormal=[]

state = 0
for i in range(1, 300):
    phi = np.split(df_wfm[:,i], 249)
    phi_min, phi_max = 0, np.max(phi)
    state_1=[]
    for item in phi:
        phi_t=item.T
        state_1.append(np.concatenate(phi_t))
        
    state_1=np.concatenate(state_1)
    
    x = np.linspace(-124, 124, 249)
    x1, x2 = np.meshgrid(x, x)
    iqr=stats.iqr(abs(df_wfm[:,i]))
    stdv=np.std(abs(df_wfm[:,i]))
    
    n_bin=int(phi_max/(3.49*stdv/np.cbrt(len(df_wfm[:,i])))) #Scott's normal reference rule
    
    binss = np.linspace(0,phi_max,n_bin)
    hist_prueba,binss =np.histogram(abs(df_wfm[:,i]),binss)
    hist_prueba=hist_prueba/62001
    weights = np.ones_like(df_wfm[:,i]) / len(df_wfm[:,i])
    
    p0 = [0.12, 0.003]
    popt,pcov = curve_fit(gaus,binss[1:],hist_prueba, p0=p0)
    gaussiana=gaus(binss[1:],*popt)
    
    
    # KOLMOGOROV TEST
    stasitic, pvalue = stats.ks_2samp(hist_prueba, gaussiana)
    
    if pvalue < 0.95:
        plt.axis('off')
        plt.gca().set_aspect("equal")
        plt.pcolor(np.array(x1), np.array(x2), np.array(state_1), cmap='cividis')
        name='Estado_'+str(state)+'_NonNormal'
        plt.savefig('/h/drodgon1/qChaos/Images/NonNormal/' + str(name)+ '.png', bbox_inches='tight', pad_inches=0.0)
        state += 1
    else:
        plt.axis('off')
        plt.gca().set_aspect("equal")
        plt.pcolor(np.array(x1), np.array(x2), np.array(state_1), cmap='cividis')
        name='Estado_'+str(state)+'_Normal'
        plt.savefig('/h/drodgon1/qChaos/Images/Normal/' + str(name)+ '.png', bbox_inches='tight', pad_inches=0.0)
        state += 1
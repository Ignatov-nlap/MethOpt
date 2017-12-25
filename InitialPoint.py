import pandas as pd;
import numpy as np;
from haversine import haversine

##longitude clustering, seek improvements by filling all trip up.
def longitude_cluster(data,delta=0.5,cluster=[0],theta=0.06,XXX=300):
    lim=980;#do not fill up 
    N = data.shape[0];
    data['lgrid'] = np.floor(data.Longitude/delta);
    data['xgrid'] = data.Latitude;#*2*(data.lgrid%2-0.5);
    data.sort(columns=['lgrid','xgrid'],inplace=True,ascending=True);
    
    tid = data['TripId'].values;
    w = data['Weight'].values;
    idx = data.lgrid.values;
    longi = data.Longitude.values;
    L = len(cluster);
    tid[:] = L;   
    CUT = 30; # set to 0 to fill each cluster to the limit.
    #theta = 0.06;
    prevfull = True;
    #if (L<=1) or (cluster[-2]>lim-CUT):
    #    prevfull = True;
    for i in range(N):
        #determine which cluster to put
        if not prevfull:
            nw = cluster[-2]+w[i];
            if nw<lim and (longi[i]-longi[i-3]<theta):# need cautious here
                cluster[-2] = nw;
                tid[i] = L-1;
                if nw>lim-CUT:
                    prevfull = True;
            else: 
                cluster[-1] += w[i];
                tid[i] = L;
                if longi[i]-longi[i-3]>theta:
                    prevfull=True;
        else:
            nw = cluster[-1] + w[i];
            if nw>lim or ((nw>lim-XXX) and (longi[i]-longi[i-1]>theta)):
                cluster.append(w[i]);                
                L = L + 1;
                tid[i] = L;
                if cluster[-2]>lim-CUT or (longi[i]-longi[i-1]>theta):
                    prevfull = True;
                else:
                    prevfull = False;
            else:
                cluster[-1] = nw;
                tid[i] = L;
                if nw>lim-2:
                    L = L+1;
                    prevfull = True;
                    cluster.append(0);          
    
    return data[['TripId','Weight']];

def get_initial_solution(gifts):   
    gifts.sort(columns='GiftId', inplace=True);

    clusters=[0];
    gifts['TripId'] = -1;
    gifts.set_index('GiftId',inplace=True);

    idx = (gifts.Latitude.values<=-60)# & (gifts.Latitude>-80);
    data = gifts[idx];
    res = longitude_cluster(data,delta=0.04,cluster=clusters,theta=0.06,XXX=280);
    gifts.loc[idx,'TripId'] = res.TripId;

    idx = (gifts.Latitude.values>=-60);
    data = gifts[idx];
    res = longitude_cluster(data,delta=0.02,cluster=clusters,theta=0.04);
    gifts.loc[idx,'TripId'] = res.TripId;

    gifts.reset_index(inplace=True);
    gifts.sort(columns=['TripId','Latitude'],inplace=True,ascending=False);
    return gifts
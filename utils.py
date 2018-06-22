import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np


def displayValueFunction(model,res,env):
    """Function for displaying the value function"""
    mapValue, mapS, mapV=mapValueFunction(model,res,env)
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    MapS, MapV = np.meshgrid(mapS, mapV)
    surf=ax.plot_surface(MapS, MapV, mapValue, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    ax.set_title('Visulaization of Value Function')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def mapValueFunction(model,res,env):
    """Function for calculating the value function"""
    maxS=env.observation_space.high[0]
    minS= env.observation_space.low[0]

    maxV = env.observation_space.high[1]
    minV = env.observation_space.low[1]


    mapS = np.linspace(minS,maxS,num=res)
    mapV = np.linspace(minV,maxV,num=res)
    mapValue = np.zeros((len(mapS),len(mapV)))

    l=0
    for i1 in (mapS):
        m = 0
        for i2 in (mapV):
            temp=model.predict(np.array([[i1, i2]]))
            mapValue[l,m]=np.amax(temp)
            m+=1
        l+=1

    return (mapValue,mapS,mapV)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

def gen_cm():
    # Blue
    cdict3 = {'red':  ((0.0, 1., 1.),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.02, 0.02)),

             'green': ((0.0, 1., 1.),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.30, 0.30)),

             'blue':  ((0.0, 1., 1.),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.68, 0.68))
            }

    cdict4 = cdict3.copy()

    cm_gray2blue = LinearSegmentedColormap('gray2blue', cdict4)
    cdict4 = cdict4.copy()
    cdict4['alpha'] = ((0.0, 0.0, .0),
                       (0.5, 0.0, 0.7),
                       (1.0, 1,1))

    cm_alpha2blue = LinearSegmentedColormap('alpha2blue', cdict4)


    ## Red
    cdict3 = {'red':  ((0.0, 1., 1.),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.7, 0.7)),

             'green': ((0.0, 1., 1.),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 1., 1.),
                       (0.5, 1.0, 1.),
                       (1.0, 0.02, 0.02))
            }

    cdict4 = cdict3.copy()


    cm_gray2red = LinearSegmentedColormap('gray2red', cdict4)
    cdict4 = cdict4.copy()
    cdict4['alpha'] = ((0.0, 0.0, .0),
                       (0.5, 0.0, 0.),
                       (1.0, 1,1))

    cm_alpha2red = LinearSegmentedColormap('alpha2red', cdict4)


    # Gray
    cg =1.
    cdict3 = {'red':  ((0.0, cg+cg, cg+cg),
                       (0.5, cg, cg),
                       (1.0, 1., 1.)),

             'green': ((0.0, cg+cg, cg+cg),
                       (0.5, cg, cg),
                       (1.0, 1., 1.)),

             'blue':  ((0.0, cg+cg, cg+cg),
                       (0.5, cg, cg),
                       (1.0, 1., 1.))
            }

    cdict4 = cdict3.copy()


    cdict4['alpha'] = ((0.0, 1., 1.),
                       (0.5, 1., 0.0),
                       (1.0, 0.0, 0.0))

    cm_gray2alpha = LinearSegmentedColormap('gray2alpha', cdict4)
    

    # Blue to Red
    cdict3 = {'red':  ((0.0, 0.02, 0.02),
                      # (0.25, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.7, 0.7)),

             'green': ((0.0, 0.30, 0.30),
                     #  (0.25, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                     #  (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.68, 0.68),
                     #  (0.25, 1.0, 1.0),
                       (0.5, 1.0, 1.0),
                     #  (0.75, 0.0, 0.0),
                       (1.0, 0.02, 0.02))
            }

    cdict4 = cdict3.copy()

    cm_blue2red = LinearSegmentedColormap('blue2red', cdict4)

    
    return cm_alpha2blue,cm_alpha2red,cm_gray2alpha,cm_blue2red


def scatter_points(res_x,res_y,new_fig=True,ax1=[],res_hitproba=[]):
    origin = 'lower'
    dotsize=30

    if new_fig:
        
        sns.set_style("white")
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)

        ax1.axis('off')
    
    labels_ = np.hstack(res_y)
    
    colors = np.array(['#0070c1']*len(labels_))
    colors[labels_==1] = '#d93857'#88419d
    colors = np.array(['#0570b0']*len(labels_))
    colors[labels_==1] = '#d7301f'#88419d
    colors[labels_==2] = '#fecc5c'
    
    ## left panel
    if res_hitproba==[]:
        ax1.scatter(np.vstack(res_x)[:,0],np.vstack(res_x)[:,1],c=colors,s=dotsize,edgecolor='none')     
    else:
        # add transparency on low hit probability
        hitprob = res_hitproba.copy()
        hitprob[res_hitproba<1.] = .25
        for ii in range(np.vstack(res_x).shape[0]):
            ax1.scatter(np.vstack(res_x)[ii,0],np.vstack(res_x)[ii,1],alpha=hitprob[ii],c=colors[ii],s=dotsize,edgecolor='none')
            
    ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square

    #plt.title('True labels')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')


def plot_2classes(X,y,hps,show_class=[0,1],ax1=[]):
    
    cm_alpha2blue,cm_alpha2red,cm_gray2alpha,cm_blue2red = gen_cm()
    
    h = .02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, h),
                         np.arange(X[:,1].min()-1, X[:,1].max()+1, h))
    
    if ax1==[]:
        sns.set_style("white")
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        ax1.axis('off')

    vmax=0.5
    vmin=-vmax
   
    if np.sum(np.array(show_class) == 0)>0:
        _, Z = hps.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z['s2_df'][:, 0]
        # Put the result into a color plot
        Z1 = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z1, cmap=cm_alpha2blue,vmin=vmin, vmax=vmax)

    if np.sum(np.array(show_class) == 1)>0:
        _,Z = hps.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z['s2_df'][:, 1]
        
        # Put the result into a color plot
        Z2 = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z2, cmap=cm_alpha2red,vmin=vmin, vmax=vmax)
        
    if np.sum(np.array(show_class) == -1)>0:
        vmax=0.1
        vmin=-vmax
        _,Z = hps.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z['s1_df'][:, 0]
        # Put the result into a color plot
        Z2 = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z2, cmap=cm_blue2red,vmin=vmin, vmax=vmax)

    #Z,_ = hps.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z[:, 1]
    # Put the result into a color plot
    #Z3 = Z.reshape(xx.shape)
    #plt.pcolormesh(xx, yy, Z3, cmap=cm_gray2alpha,vmin=-1, vmax=1)
    
    scatter_points(X,y,new_fig=False,ax1=ax1)
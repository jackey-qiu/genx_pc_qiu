import numpy as num
from matplotlib import pyplot
import matplotlib as mpt

#object_container=[[[ctr_00_g,ctr_00_g],[ctr_00_M,ctr_00_M],[ctr_00_i,ctr_00_i]],[[ctr_0m2_g,ctr_02_g],[ctr_0m2_M,ctr_02_M],[ctr_0m2_i,ctr_02_i]],[[ctr_m10_g,ctr_10_g],[ctr_m10_M,ctr_10_M],[ctr_m10_i,ctr_10_i]],[[ctr_m1m1_g,ctr_11_g],[ctr_m1m1_M,ctr_11_M],[ctr_m1m1_i,ctr_11_i]],[[ctr_m20_g,ctr_20_g],[ctr_m20_M,ctr_20_M],[ctr_m20_i,ctr_20_i]],[[ctr_m2m2_g,ctr_22_g],[ctr_m2m2_M,ctr_22_M],[ctr_m2m2_i,ctr_22_i]]]

def extract_data(object=[],hk=[]):
#this function will extract data from the full ctr dataset determined by HK
#object is a list of the full ctr datasets (eg different concentration levels),hk=[1,0]
#if hk=[1,0], this function will extract both 10L and -10L
    data_container=[]
    for i in object:
        temp=num.array([[0,0,0,0,0]])
        for j in range(len(i.H)):
            #print 'sensor'
            if (i.H[j]==0)&(i.K[j]==0):
                temp=num.append(temp,[[i.H[j],i.K[j],i.L[j],i.F[j],i.Ferr[j]]],axis=0)
            else:
                if ((i.H[j]==hk[0])&(i.K[j]==hk[1])):
                    #print 'write'
                    temp=num.append(temp,[[i.H[j],i.K[j],i.L[j],i.F[j],i.Ferr[j]]],axis=0)
                elif ((i.H[j]==-hk[0])&(i.K[j]==-hk[1])):
                    #print 'wirte'
                    temp=num.append(temp,[[-i.H[j],-i.K[j],-i.L[j],i.F[j],i.Ferr[j]]],axis=0)
        data_container.append(temp)
    return data_container
def plotting(object=[],fig=None,index=[2,3,1],color=['b'],label=['clean surface'],title=['10L'],marker=['o'],legend=True,fontsize=10,markersize=10):
    #this function will overlap a specific CTR profiles (one HKL set) at different conditions (like concentrations)
    #object=[[ctr_m10_conc1,ctr_10_conc1],[ctr_m10_conc2,ctr_10_conc2]] or object=[[ctr_10_conc1],[ctr_10_conc2]]
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    #handles=[]
    for ob in object:
        index=object.index(ob)
        if len(ob)==2:
            error1=ax.errorbar(num.append(-ob[0].L,ob[1].L),num.append(ob[0].F,ob[1].F),yerr=num.append(ob[0].Ferr,ob[1].Ferr),label=label[index],marker=marker[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],linestyle='None',markersize=2)
            #handles.append(error1)
        else:
            error1=ax.errorbar(ob[0].L,ob[0].F,yerr=ob[0].Ferr,marker=marker[index],label=label[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],linestyle='None',markersize=2)
            #handles.append(error1)
        pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
        pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
    if legend==True:
        ax.legend(bbox_to_anchor=(0.18,1.09,1.5,1.102),loc=3,ncol=3,borderaxespad=0.,prop={'size':fontsize})
        #ax.legend(bbox_to_anchor=(0.,1.02,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':fontsize})
        #fig.legend(handles,('clean surface','200 uM Pb(II) reacted surface'),loc='upper center',ncol=2,mode=None)
    return True
def plotting_many(object=[],fig=None,index=[2,3],color=['b','r','g'],label=['clean','300uM','1mM'],marker=['o','p','D'],title=['00L','02L','10L','11L','20L','22L'],legend=[True,True,True,False,False,False],fontsize=10,markersize=10):
    #do several overlapping simultaneously
    #object is a container of object that defined in the previous function
    for ob in object:
        order=object.index(ob)
        plotting(object=ob,fig=fig,index=[index[0],index[1],order+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize,markersize=markersize)
        
def plotting_raxs(object=[],fig=None,index=[2,3,1],color=['b','r'],label=['50 uM','200uM'],title=['10L'],marker=['o','p'],legend=True,xlabel=False,position=(0.5,1.05)):
    #overlapping raxs data
    #object=[raxs10_conc1,raxs10_con2]
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    ends=[min(object[0][:,4])-5,max(object[1][:,4])+50]
    for ob in range(len(object)):
        index=ob
        ax.errorbar(object[index][:,0]/1000,object[index][:,4],yerr=object[index][:,5],marker=marker[index],label=label[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],markeredgecolor=color[index],linestyle='None')
        pyplot.xlabel('Energy (kev)',axes=ax,fontsize=12)
        pyplot.ylabel('|F|',axes=ax,fontsize=12)
        pyplot.title(title[0],position=position,weight='bold',clip_on=True)
        xlabels = ax.get_xticklabels()
        for tick in xlabels:
            tick.set_rotation(30)
        if xlabel==False:
            ax.set_xticklabels([])
            pyplot.xlabel('')
        #ax.yaxis.set_major_locator(mpt.ticker.MaxNLocator())
        #print ax.get_yticks()
        #print ax.get_yticklabels()
    #ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    pyplot.ylim((ends[0]-10,ends[1]+50))
    step=int((ends[1]-ends[0])/5.)
    ax.set_yticks(list(num.arange(int(ends[0]),int(ends[1]),step)))
    ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    if legend==True:
        #ax.legend(bbox_to_anchor=(0.,0.95,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':13})
        ax.legend(bbox_to_anchor=(0.15,1.04,1.,1.102),loc=3,ncol=2,borderaxespad=0.,prop={'size':13})
def plotting_many_raxs(object=[],fig=None,index=[2,2,[1,2,3,4]],color=['0.5','0.3'],label=['50 uM Pb(II) reacted surface','200uM Pb(II) reacted surface'],marker=['p','*'],title=['RAXS_00_1.45','RAXS_00_2.77','RAXS_10_1.1','RAXS_-10_1.27'],legend=[True,False,False,False],xlabel=[False,False,True,True],position=[(0.5,0.85),(0.5,0.85),(0.5,1.05),(0.5,1.05)]):
    #plotting several raxs together
    for i in range(len(object)):
        order=i
        print 'abc'
        ob=object[i]
        plotting_raxs(object=ob,fig=fig,index=[index[0],index[1],index[2][order]],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],xlabel=xlabel[i],position=position[i])
        
def plotting_model(object=[],fig=None,index=[2,3,1],color=['b','r'],label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1],singel dataset here
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    #ends=[min(object[0][:,4])-5,max(object[1][:,4])+50]
    
    
    ax.scatter(object[0][:,0],object[0][:,2],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label=label[0])
    ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],fmt=None,color=color[0])
    ax.plot(object[0][:,0],object[0][:,1],color='r',lw=3,label=label[1])
    pyplot.xlabel('L',axes=ax,fontsize=12)
    pyplot.ylabel('|F|',axes=ax,fontsize=12)
    pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
        
        #ax.yaxis.set_major_locator(mpt.ticker.MaxNLocator())
        #print ax.get_yticks()
        #print ax.get_yticklabels()
    #ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    
    if legend==True:
        #ax.legend(bbox_to_anchor=(0.,1.02,1.,1.102),loc=3,ncol=3,mode='expand',borderaxespad=0.,prop={'size':fontsize})
        ax.legend(bbox_to_anchor=(0.5,1.02,2.,1.102),loc=3,ncol=1,borderaxespad=0.,prop={'size':fontsize})
    
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
        
def plotting_many_model(object=[],fig=None,index=[2,3],color=['b','r'],label=['Experimental data','Model fit'],marker=['p'],title=['00L','02L','10L','11L','20L','22L'],legend=[True,True,True,False,False,False],fontsize=10):
    #plotting model results simultaneously, object=[data1,data2,data3]
    for i in range(len(object)):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_model(object=[ob],fig=fig,index=[index[0],index[1],i+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)

def plotting_model_2(object=[],fig=None,index=[2,3,1],color=['b','r','g'],label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1,data2],multiple dataset here in the case you have fitting results under different conditions
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    #ends=[min(object[0][:,4])-5,max(object[1][:,4])+50]
    ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],marker=marker[0],label=label[0],ecolor=color[0],color=color[0],markerfacecolor=color[0],linestyle='None')
    #ax.scatter(object[0][:,0],object[0][:,2],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label=label[0])
    #ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],fmt=None,color=color[0])
    for i in range(len(object)):
      ax.plot(object[i][:,0],object[i][:,1],color=color[i+1],lw=3,label=label[i+1])
    pyplot.xlabel('L',axes=ax,fontsize=fontsize)
    pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
    pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
        
        #ax.yaxis.set_major_locator(mpt.ticker.MaxNLocator())
        #print ax.get_yticks()
        #print ax.get_yticklabels()
    #ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    
    if legend==True:
        ax.legend(bbox_to_anchor=(0.67,1.06,3.,1.102),mode='expand',loc=3,ncol=2,borderaxespad=0.,prop={'size':13})
        #ax.legend(bbox_to_anchor=(0.,1.02,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':13})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
    return ax
    
def plotting_many_model_2(object=[],fig=None,index=[2,3],color=['b','r','g'],label=['Experimental data','share face mode','share edge mode'],marker=['p'],title=['00L','02L','10L','11L','20L','22L'],legend=[True,True,True,False,False,False],fontsize=10):
    #plotting model results simultaneously, object=[[data1,data11],[data2,data21],[data3,data32]]
    ax_box=[]
    for i in range(len(object)):
        order=i
        #print 'abc'
        ob=object[i]
        ax=plotting_model_2(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        ax_box.append(ax)
    return ax_box
    
def plotting_model_rod(data,models=[],rods=[[0,0],[1,0]],colors=['b','g','r','c','m','y','k'],markers=['*',' ',' ',' ',' ',' ',' '],linestyles=[' ','-','-.','-','-.','-','-.'],labels=['data','model1'],fontsize=10):
    #rods=[[0,0],[1,0],[2,0],[0,2],[3,0],[2,-1],[1,1],[2,1],[2,2]]
    def _extract_dataset(data,rod=[0,0]):
        index=[]
        for i in range(len(data[:,0])):
            if (data[i,0]==rod[0])&(data[i,1]==rod[1]):
                index.append(i)
        
        new_data=data[index,:]
        return new_data[:,[2,3,4]]
    fig=pyplot.figure(figsize=(10,8))
    for rod in rods:
        data_temp=[_extract_dataset(data,rod)]
        for model in models:
            data_temp.append(_extract_dataset(model,rod))
        ax=fig.add_subplot(3,3,rods.index(rod)+1)
        ax.set_yscale('log')
        ax.errorbar(data_temp[0][:,0],data_temp[0][:,1],yerr=data_temp[0][:,2],marker=markers[0],label=labels[0],ecolor=colors[0],color=colors[0],markerfacecolor=colors[0],linestyle=' ')
        print len(data_temp)
        for i in range(len(data_temp)-1):
            ax.plot(data_temp[i+1][:,0],data_temp[i+1][:,1],color=colors[i+1],markerfacecolor=colors[i+1],markeredgecolor=colors[i+1],lw=2,label=labels[i+1],marker=markers[i+1],markersize=4,linestyle=linestyles[i+1])
        pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
        pyplot.title(str(rod[0])+str(rod[1])+'L',position=(0.5,0.85),weight='bold',clip_on=True)
        if rod==rods[0]:
            ax.legend(bbox_to_anchor=(0.2,1.06,3.,1.102),mode='expand',loc=3,ncol=3,borderaxespad=0.,prop={'size':13})
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines(): 
            l.set_markersize(5) 
            l.set_markeredgewidth(2)
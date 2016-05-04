import numpy as num
import numpy as np
from matplotlib import pyplot
import matplotlib as mpt
import pickle
import sys,os
from matplotlib import pyplot

"""
functions to make plots of CTR, RAXR, Electron Density using the dumped files created in GenX script (running_mode=False)
Formates for each kind of dumped files
1. CTR_dumped file:[experiment_data,model],both items in the list is a library of form {'HKL':[L,I,eI]} and {'HKL':[L,I]},respecitvely.
2. RAXR_dumped file: [experiment_data,model],both items in the list is a library of form {'HKL':[E,I,eI]} and {'HKL':[E,I]},respecitvely.
3. e_density_dumped file (model): [e_data, labels], where e_data=[[z,ed1],[z,ed2]...[z,ed_total]],labels=['Domain1A','Domain2A',...,'Total']
4. e_density_dumped file (imaging): [z_plot,eden_plot,eden_domains], where
    z_plot is a list [z1,z2,z3,...,zn]
    eden_plot is a list of [ed1,ed2,...,edn], which is the total e density for all domains
    eden_domains=[[ed_z1_D1,ed_z1_D2,...,ed_z1_Dm],[ed_z2_D1,ed_z2_D2,...,ed_z2_Dm],...,[ed_zn_D1,ed_zn_D2,...,ed_zn_Dm]] considering m domains
"""
bl_dl_muscovite_old={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '0_0':{'segment':[[0,20]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
    '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
    '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
bl_dl_muscovite={'0_0':{'segment':[[0,20]],'info':[[2,2]]}}

def generate_plot_files(output_file_path,sample,rgh,data,fit_mode, z_min=0,z_max=29,RAXR_HKL=[0,0,20],bl_dl=bl_dl_muscovite,height_offset=0):
    plot_data_container_experiment={}
    plot_data_container_model={}
    plot_raxr_container_experiment={}
    plot_raxr_container_model={}
    A_list_Fourier_synthesis=[]
    P_list_Fourier_synthesis=[]
    HKL_list_raxr=[[],[],[]]
    A_list_calculated,P_list_calculated,Q_list_calculated=sample.find_A_P_muscovite(h=RAXR_HKL[0],k=RAXR_HKL[1],l=RAXR_HKL[2])
    i=0
    for data_set in data:
        f=np.array([])   
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        I=data_set.y
        eI=data_set.error
        if x[0]>100:
            i+=1
            A_key_list,P_key_list=[key for key in sample.domain['raxs_vars'].keys() if 'A'+str(i)+'_D' in key and 'set' not in key and 'get' not in key],[key for key in sample.domain['raxs_vars'].keys() if 'P'+str(i)+'_D' in key and 'set' not in key and 'get' not in key]
            A_key_list.sort(),P_key_list.sort()
            A_list_Fourier_synthesis.append([sample.domain['raxs_vars'][each] for each in A_key_list])
            P_list_Fourier_synthesis.append([sample.domain['raxs_vars'][each] for each in P_key_list])
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5
            f=rough*abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=fit_mode))
            label=str(int(h[0]))+'_'+str(int(k[0]))+'_'+str(y[0])
            plot_raxr_container_experiment[label]=np.concatenate((x[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_raxr_container_model[label]=np.concatenate((x[:,np.newaxis],f[:,np.newaxis]),axis=1)
            HKL_list_raxr[0].append(h[0])
            HKL_list_raxr[1].append(k[0])
            HKL_list_raxr[2].append(y[0])
        else:
            f=np.array([])   
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            #make dumy hkl and f to make the plot look smoother
            if l[0]>0:
                l_dumy=np.arange(0.05,l[-1]+0.1,0.1)
            else:
                l_dumy=np.arange(l[0],l[-1]+0.1,0.1)
            N=len(l_dumy)
            h_dumy=np.array([h[0]]*N)
            k_dumy=np.array([k[0]]*N)
            LB_dumy=[]
            dL_dumy=[]
            f_dumy=[]
            
            for j in range(N):
                key=None
                if l_dumy[j]>=0:
                    key=str(int(h[0]))+'_'+str(int(k[0]))
                else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                for ii in bl_dl[key]['segment']:
                    if abs(l_dumy[j])>=ii[0] and abs(l_dumy[j])<ii[1]:
                        n=bl_dl[key]['segment'].index(ii)
                        LB_dumy.append(bl_dl[key]['info'][n][1])
                        dL_dumy.append(bl_dl[key]['info'][n][0])
            LB_dumy=np.array(LB_dumy)
            dL_dumy=np.array(dL_dumy)
            rough_dumy = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(l_dumy-LB_dumy)/dL_dumy)**2)**0.5
            f_dumy=rough_dumy*abs(sample.calculate_structure_factor(h_dumy,k_dumy,l_dumy,None,index=0,fit_mode=fit_mode,height_offset=height_offset))
            f_dumy=f_dumy*f_dumy
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis]),axis=1)
    Q_list_Fourier_synthesis=np.pi*2*sample.unit_cell.abs_hkl(np.array(HKL_list_raxr[0]),np.array(HKL_list_raxr[1]),np.array(HKL_list_raxr[2]))    
    #dump CTR data and profiles
    hkls=['00L']
    plot_data_list=[]
    for hkl in hkls:
        plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
    pickle.dump(plot_data_list,open(output_file_path+"temp_plot","wb"))
    #dump raxr data and profiles
    pickle.dump([plot_raxr_container_experiment,plot_raxr_container_model],open(output_file_path+"temp_plot_raxr","wb"))
    pickle.dump([[A_list_calculated,P_list_calculated,Q_list_calculated],[A_list_Fourier_synthesis,P_list_Fourier_synthesis,Q_list_Fourier_synthesis]],open(output_file_path+"temp_plot_raxr_A_P_Q","wb"))
    #dump electron density profiles
    #e density based on model fitting
    sample.plot_electron_density_muscovite(sample.domain,file_path=output_file_path,z_min=z_min,z_max=z_max,N_layered_water=20,height_offset=height_offset)#dumpt file name is "temp_plot_eden" 
    #e density based on Fourier synthesis
    z_plot,eden_plot,eden_domains=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_Fourier_synthesis).transpose(),np.array(A_list_Fourier_synthesis).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000)
    pickle.dump([z_plot,eden_plot,eden_domains],open(output_file_path+"temp_plot_eden_fourier_synthesis","wb"))  

#this function must be called within the shell of GenX gui and par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q' by default
#The purpose of this function is to append the errors of A and P extracted from the errors displaying inside the tab of GenX gui 
#copy and past this command line to the shell for action:
#model.script_module.create_plots.append_errors_for_A_P(par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs')   
def append_errors_for_A_P(par_instance,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs'):
    data_AP_Q=pickle.load(open(dump_file,"rb"))
    AP_calculated=data_AP_Q[0]
    A_model_fit,P_model_fit=data_AP_Q[1][0],data_AP_Q[1][1]
    A_error_model_fit,P_error_model_fit=[],[]
    table=np.array(par_instance.data)
    for i in range(len(A_model_fit)):
        A_error_model_fit_domain=[]
        for j in range(len(A_model_fit[i])):
            par_name=raxs_rgh+'.setA'+str(i+1)+'_D'+str(j+1)
            for k in range(len(table)):
                if table[k][0]==par_name:
                    if table[k][5][0]=='(' and table[k][5][-1]==')':
                        error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                        A_error_model_fit_domain.append(error)
                    else:
                        A_error_model_fit_domain.append(np.array([0.1,0.1]))
        A_error_model_fit.append(A_error_model_fit_domain)
    for i in range(len(P_model_fit)):
        P_error_model_fit_domain=[]
        for j in range(len(P_model_fit[i])):
            par_name=raxs_rgh+'.setP'+str(i+1)+'_D'+str(j+1)
            for k in range(len(table)):
                if table[k][0]==par_name:
                    if table[k][5][0]=='(' and table[k][5][-1]==')':
                        error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                        P_error_model_fit_domain.append(error)
                    else:
                        P_error_model_fit_domain.append(np.array([0.1,0.1]))
        P_error_model_fit.append(P_error_model_fit_domain)
    dump_data=[[AP_calculated[0],AP_calculated[1],AP_calculated[2]],[data_AP_Q[1][0],data_AP_Q[1][1],data_AP_Q[1][2],A_error_model_fit,P_error_model_fit]]
    pickle.dump(dump_data,open(dump_file,"wb"))

def plotting_raxr_new(data,savefile="D://raxr_temp.png",color=['b','r'],marker=['o']):
    experiment_data,model=data[0],data[1]
    labels=model.keys()
    labels.sort()
    fig=pyplot.figure(figsize=(15,len(labels)/3))
    for i in range(len(labels)):
        rows=None
        if len(labels)%4==0:
            rows=len(labels)/4
        else:
            rows=len(labels)/4+1
        ax=fig.add_subplot(rows,4,i+1)
        ax_pre=ax
        ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label="data points")
        ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],yerr=experiment_data[labels[i]][:,2],fmt=None,color=color[0])
        ax.plot(model[labels[i]][:,0],model[labels[i]][:,1],color=color[1],lw=3,label='model profile')
        if i!=len(labels)-1:
            ax.set_xticklabels([])
            pyplot.xlabel('')
        else:
            pyplot.xlabel('Energy (kev)',axes=ax,fontsize=12)
        pyplot.ylabel('|F|',axes=ax,fontsize=12)
        pyplot.title(labels[i])
    fig.tight_layout()
    fig.savefig(savefile,dpi=300)
    return fig
        
def plotting_modelB(object=[],fig=None,index=[2,3,1],color=['0.35','r','c','m','k'],l_dashes=[()],lw=3,label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1,data2,data3],multiple dataset with the first one of experimental data and the others model datasets
    
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    ax.scatter(object[0][:,0],object[0][:,1],marker='o',s=20,facecolors='none',edgecolors=color[0],label=label[0])
    ax.errorbar(object[0][:,0],object[0][:,1],yerr=object[0][:,2],fmt=None,ecolor=color[0])
    for i in range(len(object)-1):#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
        l,=ax.plot(object[i+1][:,0],object[i+1][:,1],color=color[i+1],lw=lw,label=label[i+1])
        l.set_dashes(l_dashes[i])
    if index[2] in [7,8,9]:
        pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=12)
    if index[2] in [1,4,7]:
        pyplot.ylabel(r'$|F_{HKL}|$',axes=ax,fontsize=12)
    #settings for demo showing
    pyplot.title('('+title[0]+')',position=(0.5,0.86),weight=4,size=10,clip_on=True)
    if title[0]=='0 0 L':
        pyplot.ylim((0,1000))
        #pyplot.xlim((0,20))
    elif title[0]=='3 0 L':
        pyplot.ylim((1,10000))
    else:pyplot.ylim((1,10000))
    #pyplot.ylim((1,1000))
    #settings for publication
    #pyplot.title('('+title[0]+')',position=(0.5,1.001),weight=4,size=10,clip_on=True)
    """##add arrows to antidote the misfits 
    if title[0]=='0 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.25,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
        ax.add_patch(mpt.patches.FancyArrow(0.83,0.5,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='1 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.68,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='3 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.375,0.8,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    """    
    if legend==True:
        #ax.legend()
        ax.legend(bbox_to_anchor=(0.2,1.03,3.,1.202),mode='expand',loc=3,ncol=5,borderaxespad=0.,prop={'size':9})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)

    #ax.set_ylim([1,10000])
#object files are returned from genx when switch the plot on
def plotting_many_modelB(save_file='D://pic.png',head='C:\\Users\\jackey\\Google Drive\\useful codes\\plotting\\',object_files=['temp_plot_O1O2','temp_plot_O5O6','temp_plot_O1O3','temp_plot_O5O7','temp_plot_O1O4','temp_plot_O5O8'],index=[3,3],color=['0.6','b','b','g','g','r','r'],lw=1.5,l_dashes=[(2,2,2,2),(None,None),(2,2,2,2),(None,None),(2,2,2,2),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    #setting for demo show
    #fig=pyplot.figure(figsize=(10,9))
    #settings for publication
    #fig=pyplot.figure(figsize=(10,7))
    fig=pyplot.figure(figsize=(8.5,7))
    object_sets=[pickle.load(open(head+file)) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
    object=[]
    for i in range(len(object_sets[0])):
        object.append([])
        for j in range(len(object_sets)):
            if j==0:
                object[-1].append(object_sets[j][i][0])
            object[-1].append(object_sets[j][i][1])
    if len(object_sets[0])==1:
        index=[2,1]

    for i in range(len(object)):
    #for i in range(1):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        #plotting_modelB(object=ob,fig=fig,index=[1,1,i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig
    
def plotting_single_rod(save_file='D://pic.png',head='C:\\Users\\jackey\\Google Drive\\useful codes\\plotting\\',object_files=['temp_plot_O1O2','temp_plot_O5O6','temp_plot_O1O3','temp_plot_O5O7','temp_plot_O1O4','temp_plot_O5O8'],index=[1,1],color=['0.6','b','b','g','g','r','r'],lw=1.5,l_dashes=[(2,2,2,2),(None,None),(2,2,2,2),(None,None),(2,2,2,2),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10,rod_index=0):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    #setting for demo show
    #fig=pyplot.figure(figsize=(10,9))
    #settings for publication
    #fig=pyplot.figure(figsize=(10,7))
    fig=pyplot.figure(figsize=(8.5,7))
    object_sets=[pickle.load(open(head+file)) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
    object=[]
    for i in range(9):
        object.append([])
        for j in range(len(object_sets)):
            if j==0:
                object[-1].append(object_sets[j][i][0])
            object[-1].append(object_sets[j][i][1])

    for i in [rod_index]:
    #for i in range(1):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=None,fontsize=fontsize)
        #plotting_modelB(object=ob,fig=fig,index=[1,1,i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig
    
#overplotting experimental datas formated with UAF_CTR_RAXS_2 loader in GenX            
def plot_many_experiment_data(data_files=['D:\\Google Drive\\data\\400uM_Sb_hematite_rcut.datnew_formate','D:\\Google Drive\\data\\1000uM_Pb_hematite_rcut.datnew_formate'],labels=['Sb 400uM on hematite','Pb 1000uM on hematite'],HKs=[[0,0],[0,2],[1,0],[1,1],[2,0],[2,1],[2,-1],[2,2],[3,0]],index_subplot=[3,3],colors=['b','g','r','c','m','y','w'],markers=['.','*','o','v','^','<','>'],fontsize=10):
    data_container={}
    for i in range(len(labels)):
        temp_data=np.loadtxt(data_files[i])
        sub_set={}
        for HK in HKs:
            label=str(int(HK[0]))+'_'+str(int(HK[1]))
            sub_set[label]=np.array(filter(lambda x:x[1]==HK[0] and x[2]==HK[1],temp_data))
        data_container[labels[i]]=sub_set
    fig=pyplot.figure()
    for i in range(len(HKs)):
        title=str(int(HKs[i][0]))+str(int(HKs[i][1]))+'L'
        ax=fig.add_subplot(index_subplot[0],index_subplot[1],i+1)
        ax.set_yscale('log')
        for label in labels:
            data_temp=data_container[label][str(int(HKs[i][0]))+'_'+str(int(HKs[i][1]))]
            ax.errorbar(data_temp[:,0],data_temp[:,4],data_temp[:,5],label=label,marker=markers[labels.index(label)],ecolor=colors[labels.index(label)],color=colors[labels.index(label)],markerfacecolor=colors[labels.index(label)],linestyle='None',markersize=8)
        pyplot.title(title,position=(0.5,0.85),weight='bold',clip_on=True)
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines(): 
            l.set_markersize(5) 
            l.set_markeredgewidth(2)
        if i==0:
            ax.legend(bbox_to_anchor=(0.5,0.92,0,3),bbox_transform=fig.transFigure,loc='lower center',ncol=4,borderaxespad=0.,prop={'size':14})
        if (i+1)>index_subplot[1]*(index_subplot[0]-1):
            pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        if i%index_subplot[1]==0:
            pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
    return True
    
if __name__=="__main__":    

    #which plots do you want to create
    plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q=1,0,1,0,0

    #specify file paths (files are dumped files when setting running_mode=False in GenX script)
    e_file="D:\\temp_plot_eden"#e density from model
    e_file_FS="D:\\temp_plot_eden_fourier_synthesis" #e density from Fourier synthesis
    ctr_file_folder="D:\\"
    ctr_file_names=["temp_plot"]#you may want to overplot differnt ctr profiles based on differnt models
    raxr_file="D:\\temp_plot_raxr"
    AP_Q_file="D:\\temp_plot_raxr_A_P_Q"
    #plot electron density profile
    if plot_e_model: 
        data_eden=pickle.load(open(e_file,"rb"))
        edata,labels=data_eden[0],data_eden[1]
        N=len(labels)
        fig=pyplot.figure(figsize=(15,6))
        if plot_e_FS:
            data_eden_FS=pickle.load(open(e_file_FS,"rb"))
        for i in range(N):
            ax=fig.add_subplot(N,1,i+1)
            ax.plot(np.array(edata[i][0,:]),edata[i][1,:],color='b',label="model dependent")
            pyplot.title(labels[i])
            if plot_e_FS:
                if i!=N-1:
                    ax.plot(data_eden_FS[0],list(np.array(data_eden_FS[2])[:,i]),color='r',label="RAXR imaging")
                else:
                    ax.plot(data_eden_FS[0],data_eden_FS[1],color='r',label="RAXR imaging")
            if i==N-1:pyplot.xlabel('Z(Angstrom)',axes=ax,fontsize=12)
            pyplot.ylabel('E_density',axes=ax,fontsize=12)
            pyplot.ylim(ymin=0)
            pyplot.legend()
        fig.tight_layout()
        fig.savefig(e_file+".png",dpi=300)
    if plot_ctr:
        #plot ctr profiles
        #plotting_single_rod(save_file=ctr_file_folder+"temp_plot_ctr.png",head=ctr_file_folder,object_files=ctr_file_names,color=['w','r'],l_dashes=[(None,None)],lw=2,rod_index=0)
        plotting_many_modelB(save_file=ctr_file_folder+"temp_plot_ctr.png",head=ctr_file_folder,object_files=ctr_file_names,color=['b','r'],l_dashes=[(None,None)],lw=2)
    if plot_raxr:
        #plot raxr profiles
        data_raxr=pickle.load(open(raxr_file,"rb"))
        plotting_raxr_new(data_raxr,savefile=raxr_file+".png",color=['b','r'],marker=['o'])
    if plot_AP_Q:
        #plot Q dependence of Foriour components A and P
        colors=['black','r','blue','green','yellow']
        labels=['Domain1','Domain2','Domain3','Domain4']
        data_AP_Q=pickle.load(open(AP_Q_file,"rb"))
        fig1=pyplot.figure(figsize=(15,6))
        ax1=fig1.add_subplot(1,1,1)
        shape=data_AP_Q[0][0].shape
        #A over Q
        for i in range(shape[1]):
            ax1.plot(data_AP_Q[0][2],np.array(data_AP_Q[0][0])[:,i],color=colors[i])
            #print shape
            #print np.array(data_AP_Q[1][3])
            ax1.errorbar(data_AP_Q[1][2],np.array(data_AP_Q[1][0])[:,i],yerr=[np.array(data_AP_Q[1][3])[:,i,:][:,0],np.array(data_AP_Q[1][3])[:,i,:][:,1]],color=colors[i],fmt='-o')
        pyplot.ylabel("A",axes=ax1)
        pyplot.xlabel("Q",axes=ax1)
        pyplot.legend()
        #P over Q
        fig2=pyplot.figure(figsize=(15,6))
        ax2=fig2.add_subplot(1,1,1)
        for i in range(shape[1]):
            ax2.plot(data_AP_Q[0][2],np.array(data_AP_Q[0][1])[:,i]/data_AP_Q[0][2]*np.pi*2,color=colors[i])
            ax2.errorbar(data_AP_Q[1][2],np.array(data_AP_Q[1][1])[:,i]/data_AP_Q[1][2]*np.pi*2,yerr=[np.array(data_AP_Q[1][4])[:,i,:][:,0]/data_AP_Q[1][2]*np.pi*2,np.array(data_AP_Q[1][4])[:,i,:][:,1]/data_AP_Q[1][2]*np.pi*2],color=colors[i],fmt='-o')
        pyplot.ylabel("P/Q(2pi)",axes=ax2)
        pyplot.xlabel("Q",axes=ax2)
        pyplot.legend()
    pyplot.show()
    
    
    
    
    
    
    
import numpy as num
import numpy as np
from matplotlib import pyplot
import matplotlib as mpt
import pickle
import sys,os,inspect
from matplotlib import pyplot
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def local_func():
    return None
    
def module_path_locator(func=local_func):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(func)))),'dump_files')  

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
            A_list_Fourier_synthesis.append(sample.domain['raxs_vars'][A_key_list[0]])
            P_list_Fourier_synthesis.append(sample.domain['raxs_vars'][P_key_list[0]])
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5
            f=rough*abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=fit_mode,height_offset=height_offset))
            f=f*f
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
            q_dumy=np.pi*2*sample.unit_cell.abs_hkl(h_dumy,k_dumy,l_dumy)
            q_data=np.pi*2*sample.unit_cell.abs_hkl(h,k,l)
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
            f_ctr=lambda q:(np.sin(q*20.003509882813105/4))**2
            #f_ctr=lambda q:(np.sin(q*19.96/4))**2
            f_dumy_norm=f_dumy*f_ctr(q_dumy)
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis],(I*f_ctr(q_data))[:,np.newaxis],(eI*f_ctr(q_data))[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis],f_dumy_norm[:,np.newaxis]),axis=1)
    Q_list_Fourier_synthesis=np.pi*2*sample.unit_cell.abs_hkl(np.array(HKL_list_raxr[0]),np.array(HKL_list_raxr[1]),np.array(HKL_list_raxr[2]))    
    
    A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=list(HKL_list_raxr[0]),k=list(HKL_list_raxr[1]),l=list(HKL_list_raxr[2]))
    #A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=HKL_list_raxr[0][0],k=HKL_list_raxr[1][0],l=HKL_list_raxr[2][-1])

    
    #dump CTR data and profiles
    hkls=['00L']
    plot_data_list=[]
    for hkl in hkls:
        plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
    pickle.dump(plot_data_list,open(os.path.join(output_file_path,"temp_plot"),"wb"))
    #dump raxr data and profiles
    pickle.dump([plot_raxr_container_experiment,plot_raxr_container_model],open(os.path.join(output_file_path,"temp_plot_raxr"),"wb"))
    pickle.dump([[A_list_calculated,P_list_calculated,Q_list_calculated],[A_list_Fourier_synthesis,P_list_Fourier_synthesis,Q_list_Fourier_synthesis]],open(os.path.join(output_file_path,"temp_plot_raxr_A_P_Q"),"wb"))
    #dump electron density profiles
    #e density based on model fitting
    sample.plot_electron_density_muscovite(sample.domain,file_path=output_file_path,z_min=z_min,z_max=z_max,N_layered_water=100,height_offset=height_offset)#dumpt file name is "temp_plot_eden" 
    #e density based on Fourier synthesis
    z_plot,eden_plot,eden_domains=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_Fourier_synthesis).transpose(),np.array(A_list_Fourier_synthesis).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000)
    z_plot_sub,eden_plot_sub,eden_domains_sub=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_calculated_sub).transpose(),np.array(A_list_calculated_sub).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000)
    #z_plot_sub,eden_plot_sub,eden_domains_sub=sample.fourier_synthesis(np.array([[HKL_list_raxr[0][0]]*100,[HKL_list_raxr[1][0]]*100,np.arange(0,HKL_list_raxr[2][-1],HKL_list_raxr[2][-1]/100.)]),np.array(P_list_calculated_sub).transpose(),np.array(A_list_calculated_sub).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000)

    pickle.dump([z_plot,eden_plot,eden_domains],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis"),"wb"))
    pickle.dump([z_plot_sub,eden_plot_sub,eden_domains_sub],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis_sub"),"wb")) 

#a function to make files to generate vtk files    
def generate_plot_files_2(output_file_path,sample,rgh,data,fit_mode, z_min=0,z_max=29,RAXR_HKL=[0,0,20],bl_dl=bl_dl_muscovite,height_offset=0,tag=1):
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
            A_list_Fourier_synthesis.append(sample.domain['raxs_vars'][A_key_list[0]])
            P_list_Fourier_synthesis.append(sample.domain['raxs_vars'][P_key_list[0]])
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5
            f=rough*abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=fit_mode,height_offset=height_offset))
            f=f*f
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
            q_dumy=np.pi*2*sample.unit_cell.abs_hkl(h_dumy,k_dumy,l_dumy)
            q_data=np.pi*2*sample.unit_cell.abs_hkl(h,k,l)
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
            f_ctr=lambda q:(np.sin(q*20.003509882813105/4))**2
            #f_ctr=lambda q:(np.sin(q*19.96/4))**2
            f_dumy_norm=f_dumy*f_ctr(q_dumy)
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis],(I*f_ctr(q_data))[:,np.newaxis],(eI*f_ctr(q_data))[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis],f_dumy_norm[:,np.newaxis]),axis=1)
    Q_list_Fourier_synthesis=np.pi*2*sample.unit_cell.abs_hkl(np.array(HKL_list_raxr[0]),np.array(HKL_list_raxr[1]),np.array(HKL_list_raxr[2]))    
    
    A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=list(HKL_list_raxr[0]),k=list(HKL_list_raxr[1]),l=list(HKL_list_raxr[2]))
    #A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=HKL_list_raxr[0][0],k=HKL_list_raxr[1][0],l=HKL_list_raxr[2][-1])

    #output files
    #CTR
    np.savetxt('D://temp_CTR'+str(tag),plot_data_container_model['00L'])
    #RAXR
    keys=plot_raxr_container_model.keys()
    keys.sort()
    np.savetxt('D://temp_RAXR'+str(tag),plot_raxr_container_model[keys[0]])
    #Fourier components
    #print A_list_calculated
    ap_data=np.concatenate((A_list_calculated[:,np.newaxis],P_list_calculated[:,np.newaxis],Q_list_calculated[:,np.newaxis]),axis=1)
    np.savetxt('D://temp_APQ'+str(tag),ap_data)
       
#this function must be called within the shell of GenX gui and par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q' by default
#The purpose of this function is to append the errors of A and P extracted from the errors displaying inside the tab of GenX gui 
#copy and past this command line to the shell for action:
#model.script_module.create_plots.append_errors_for_A_P(par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs')   
def append_errors_for_A_P_original(par_instance,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs'):
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
    
def append_errors_for_A_P(par_instance,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs'):
    data_AP_Q=pickle.load(open(dump_file,"rb"))
    AP_calculated=data_AP_Q[0]
    A_model_fit,P_model_fit=data_AP_Q[1][0],data_AP_Q[1][1]
    A_error_model_fit,P_error_model_fit=[],[]
    table=np.array(par_instance.data)
    for i in range(len(A_model_fit)):
        par_name=raxs_rgh+'.setA'+str(i+1)+'_D'+str(1)
        for k in range(len(table)):
            if table[k][0]==par_name:
                if table[k][5][0]=='(' and table[k][5][-1]==')':
                    error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                    A_error_model_fit.append(error)
                else:
                    A_error_model_fit.append(np.array([0.1,0.1]))
    for i in range(len(P_model_fit)):
        par_name=raxs_rgh+'.setP'+str(i+1)+'_D'+str(1)
        for k in range(len(table)):
            if table[k][0]==par_name:
                if table[k][5][0]=='(' and table[k][5][-1]==')':
                    error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                    P_error_model_fit.append(error)
                else:
                    P_error_model_fit.append(np.array([0.1,0.1]))
    dump_data=[[AP_calculated[0],AP_calculated[1],AP_calculated[2]],[data_AP_Q[1][0],data_AP_Q[1][1],data_AP_Q[1][2],A_error_model_fit,P_error_model_fit]]
    pickle.dump(dump_data,open(dump_file,"wb"))

def plotting_raxr_new(data,savefile="D://raxr_temp.png",color=['b','r'],marker=['o']):
    experiment_data,model=data[0],data[1]
    labels=model.keys()
    label_tag=map(lambda x:float(x.split("_")[-1]),labels)
    label_tag.sort()
    labels=map(lambda x:"0_0_"+str(x),label_tag)
    #labels.sort()
    fig=pyplot.figure(figsize=(15,len(labels)/3))
    for i in range(len(labels)):
        rows=None
        if len(labels)%3==0:
            rows=len(labels)/3
        else:
            rows=len(labels)/3+1
        ax=fig.add_subplot(rows,3,i+1)
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
    #plot normalized data now    
    ax=fig.add_subplot(index[0],index[1],index[2]+1)
    ax.set_yscale('log')
    ax.scatter(object[0][:,0],object[0][:,3],marker='o',s=20,facecolors='none',edgecolors=color[0],label=label[0])
    ax.errorbar(object[0][:,0],object[0][:,3],yerr=object[0][:,4],fmt=None,ecolor=color[0])
    for i in range(len(object)-1):#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
        l,=ax.plot(object[i+1][:,0],object[i+1][:,2],color=color[i+1],lw=lw,label=label[i+1])
        l.set_dashes(l_dashes[i])
    if index[2] in [7,8,9]:
        pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=12)
    if index[2] in [1,4,7]:
        pyplot.ylabel(r'$|normalized F_{HKL}|$',axes=ax,fontsize=12)
    #settings for demo showing
    pyplot.title('('+title[0]+')',position=(0.5,0.86),weight=4,size=10,clip_on=True)
    if title[0]=='0 0 L':
        pass
        #pyplot.ylim((0,1000))
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
    fig=pyplot.figure(figsize=(8,8))
    object_sets=[pickle.load(open(os.path.join(head,file))) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
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
    
def plot_all(path=module_path_locator()):
    PATH=path
    #which plots do you want to create
    plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q=1,1,1,1,1

    #specify file paths (files are dumped files when setting running_mode=False in GenX script)
    e_file=os.path.join(PATH,"temp_plot_eden")#e density from model
    e_file_FS=os.path.join(PATH,"temp_plot_eden_fourier_synthesis") #e density from Fourier synthesis
    ctr_file_folder=PATH
    ctr_file_names=["temp_plot"]#you may want to overplot differnt ctr profiles based on differnt models
    raxr_file=os.path.join(PATH,"temp_plot_raxr")
    AP_Q_file=os.path.join(PATH,"temp_plot_raxr_A_P_Q")
    e_den_subtracted=None
    e_den_raxr_MI=None
    #plot electron density profile
    if plot_e_model: 
        data_eden=pickle.load(open(e_file,"rb"))
        edata,labels=data_eden[0],data_eden[1]
        N=len(labels)
        fig=pyplot.figure(figsize=(15,6))
        if plot_e_FS:
            data_eden_FS=pickle.load(open(e_file_FS,"rb"))
            data_eden_FS_sub=pickle.load(open(e_file_FS+"_sub","rb"))
        for i in range(N):
            if i==N-1:
                ax=fig.add_subplot(1,2,2)
            else:
                ax=fig.add_subplot(N/2+1,2,i*2+1)
            ax.plot(np.array(edata[i][0,:]),edata[i][1,:],color='b',label="Total e density")
            try:#some domain may have no raxr element
                ax.plot(np.array(edata[i][0,:]),edata[i][2,:],color='g',label="RAXS element e profile (MD)")
            except:
                pass
            pyplot.title(labels[i],fontsize=11)
            if plot_e_FS:
                if i==0:
                    ax.plot(data_eden_FS[0],list(np.array(data_eden_FS[2])[:,i]),color='r',label="RAXR imaging (MI)")
                    ax.fill_between(data_eden_FS[0],list(np.array(data_eden_FS[2])[:,i]),color='m',alpha=0.6)
                    #clip off negative part of the e density through Fourier thynthesis
                    ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[2])[:,i]*(np.array(data_eden_FS[2])[:,i]>0.01)),color='black',alpha=0.6,label="Total e - LayerWater - RAXR")
                    ax.fill_between(data_eden_FS[0],edata[i][3,:],color='blue',alpha=0.6,label="LayerWater")

                    ax.plot(data_eden_FS_sub[0],list(np.array(data_eden_FS_sub[2])[:,i]),color='black',label="RAXR imaging (MD)")
                    ax.fill_between(data_eden_FS_sub[0],list(np.array(data_eden_FS_sub[2])[:,i]),color='c',alpha=0.6)
                elif i==N-1:
                    ax.plot(data_eden_FS[0],data_eden_FS[1],color='r',label="RAXR imaging (MI)")
                    ax.fill_between(data_eden_FS[0],data_eden_FS[1],color='m',alpha=0.6)
                    #ax.fill_between(data_eden_FS[0],edata[i][1,:]-data_eden_FS[1],color='black',alpha=0.6,label="Total e - RAXR(MI)")
                    ax.fill_between(data_eden_FS[0],edata[i][3,:],color='blue',alpha=0.6,label="LayerWater")
                    ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01)),color='black',alpha=0.6,label="Total e - LayerWater - RAXR")
                    eden_temp=list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01))
                    eden_temp=(np.array(eden_temp)*(np.array(eden_temp)>0.01))[:,np.newaxis]
                    z_temp=np.array(data_eden_FS[0])[:,np.newaxis]
                    e_den_subtracted=np.append(z_temp,eden_temp,axis=1)
                    e_den_raxr_MI=np.append(np.array(data_eden_FS[0])[:,np.newaxis],(np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01))[:,np.newaxis],axis=1)
                    ax.plot(data_eden_FS_sub[0],data_eden_FS_sub[1],color='black',label="RAXR imaging (MD)")
                    ax.fill_between(data_eden_FS_sub[0],data_eden_FS_sub[1],color='c',alpha=0.6)
            if i==N-1:pyplot.xlabel('Z(Angstrom)',axes=ax,fontsize=12)
            pyplot.ylabel('E_density',axes=ax,fontsize=12)
            pyplot.ylim(ymin=0)
            pyplot.legend(fontsize=11,ncol=1)
        fig.tight_layout()
        fig.savefig(e_file+".png",dpi=300)
    if plot_ctr:
        #plot ctr profiles
        #plotting_single_rod(save_file=ctr_file_folder+"temp_plot_ctr.png",head=ctr_file_folder,object_files=ctr_file_names,color=['w','r'],l_dashes=[(None,None)],lw=2,rod_index=0)
        plotting_many_modelB(save_file=os.path.join(ctr_file_folder,"temp_plot_ctr.png"),head=ctr_file_folder,object_files=ctr_file_names,color=['b','r'],l_dashes=[(None,None)],lw=2)
    if plot_raxr:
        #plot raxr profiles
        data_raxr=pickle.load(open(raxr_file,"rb"))
        plotting_raxr_new(data_raxr,savefile=raxr_file+".png",color=['b','r'],marker=['o'])
    if plot_AP_Q:
        #plot Q dependence of Foriour components A and P
        colors=['black','r','blue','green','yellow']
        labels=['Domain1','Domain2','Domain3','Domain4']
        data_AP_Q=pickle.load(open(AP_Q_file,"rb"))
        fig1=pyplot.figure(figsize=(9,9))
        ax1=fig1.add_subplot(2,1,1)
        #A over Q
        ax1.plot(data_AP_Q[0][2],data_AP_Q[0][0],color='r')
        ax1.errorbar(data_AP_Q[1][2],data_AP_Q[1][0],yerr=np.transpose(data_AP_Q[1][3]),color='g',fmt='o')
        pyplot.ylabel("A",axes=ax1)
        pyplot.xlabel("Q",axes=ax1)
        pyplot.legend()
        #P over Q
        ax2=fig1.add_subplot(2,1,2)
        ax2.plot(data_AP_Q[0][2],np.array(data_AP_Q[0][1])/np.array(data_AP_Q[0][2])*np.pi*2,color='r')
        ax2.errorbar(data_AP_Q[1][2],np.array(data_AP_Q[1][1])/np.array(data_AP_Q[1][2])*np.pi*2,yerr=np.transpose(data_AP_Q[1][4])*np.pi*2/[data_AP_Q[1][2],data_AP_Q[1][2]],color='g',fmt='o')
        pyplot.ylabel("P/Q(2pi)",axes=ax2)
        pyplot.xlabel("Q",axes=ax2)
        pyplot.legend()
        fig1.savefig(os.path.join(PATH,'temp_APQ_profile.png'),dpi=300)
    #now plot the subtracted e density and print out the gaussian fit results
    
    pyplot.figure()
    print '##############Total e - raxr -layer water#################'
    gaussian_fit(e_den_subtracted)
    pyplot.title('Total e - raxr -layer water')
    pyplot.figure()
    print '#########################RAXR (MI)########################'
    gaussian_fit(e_den_raxr_MI,zs=None,N=40)
    pyplot.title('RAXR (MI)')
    pyplot.figure()
    print '#########################RAXR (MD)########################'
    gaussian_fit(np.append([data_eden_FS_sub[0]],[data_eden_FS_sub[1]*(np.array(data_eden_FS_sub[1])>0)],axis=0).transpose(),zs=None,N=40)
    pyplot.title('RAXR (MD)')
    pyplot.show()
    #return e_den_subtracted,data_eden_FS

def gaussian_fit(data,fit_range=[1,40],zs=None,N=8):
    x,y=[],[]
    for i in range(len(data)):
        if data[i,0]>fit_range[0] and data[i,0]<fit_range[1]:
            x.append(data[i,0]),y.append(data[i,1])
    x,y=np.array(x),np.array(y)
    plt.plot(x,y)
    plt.show()

    def func(x_ctrs,*params):
        y = np.zeros_like(x_ctrs[0])
        x=x_ctrs[0]
        ctrs=x_ctrs[1]
        for i in range(0, len(params), 2):
            amp = abs(params[i])
            wid = abs(params[i+1])
            ctr=ctrs[int(i/2)]
            y = y + amp * np.exp( -((x - ctr)/wid)**2/2)
        return y

    guess = []
    ctrs=[]
    if zs==None:
        for i in range(1,len(x)-1):
            if y[i-1]<y[i] and y[i+1]<y[i]:
                ctrs.append(x[i])
    elif type(zs)==int:
        ctrs=[fit_range[0]+(fit_range[1]-fit_range[0])/zs*i for i in range(zs)]+[fit_range[1]]
    else:
        ctrs=np.array(zs)
    for i in range(len(ctrs)):
        guess += [0.5, 1]   

    popt, pcov = curve_fit(func, [x,ctrs], y, p0=guess)
    combinded_set=[]
    #print 'z occupancy*4 U(sigma**2)'
    for i in range(0,len(popt),2):
        combinded_set=combinded_set+[ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2]
        #print '%3.3f\t%3.3f\t%3.3f'%(ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2)
    combinded_set=np.reshape(np.array(combinded_set),(len(combinded_set)/3,3)).transpose()
    #combinded_set=combinded_set.transpose()
    print 'total_occupancy=',np.sum(combinded_set[1,:]/4)
    print 'OC_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])'
    print 'U_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])'
    print 'X_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Y_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Z_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])'
    
    
    fit = func([x,ctrs], *popt)

    plt.plot(x, y)
    plt.plot(x, fit , 'r-')
    plt.show()
    
    
if __name__=="__main__":    
    plot_all()
    
    
    
    
    
    
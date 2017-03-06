import numpy as np

def qsi_correction(data_path='M:\\fwog\\members\\qiu05\\mica\\nQc_zr_mica_CTR_May19_GenX_formate.dat',L_column=0,I_column=4,correction_factor=0.3125):
    #correction_factor=2pi/c_project, where c-project=volume/a*b
    data=np.loadtxt(data_path)
    I=data[:,I_column]*(correction_factor*data[:,L_column])**2
    data[:,I_column]=I
    data[:,I_column+1]=data[:,I_column+1]*(correction_factor*data[:,L_column])**2
    np.savetxt(data_path.replace('.dat','_Q_corrected.dat'),data,fmt='%.5e')
    return True

def l_correction(data_path='P:\\apps\\genx_pc_qiu\\dump_files\\temp_full_dataset.dat',L_column=[0,3],I_column=4,correction_factor=0.3125,l_shift=0):
    data=np.loadtxt(data_path)
    raxr_first=None
    for i in range(len(data)):
        if data[i,0]>100:
            raxr_first=i
            break
    data[0:raxr_first,I_column]=(data[0:raxr_first,0]/(data[0:raxr_first,0]+l_shift))**2*data[0:raxr_first,I_column]
    data[raxr_first:len(data),I_column]=(data[raxr_first:len(data),3]/(data[raxr_first:len(data),3]+l_shift))**2*data[raxr_first:len(data),I_column]
    np.savetxt(data_path.replace('.dat','_l_corrected.dat'),data,fmt='%.5e')
    return True

bl_dl_muscovite={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '0_0':{'segment':[[0,20]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
    '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
    '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
def formate_CTR_data(file='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\nQc_S0_Zr_0mM_NaCl_Dry_CTR_1st_spot1_R',bragg_peaks=bl_dl_muscovite):
    data_formated=None
    f_original=np.loadtxt(file,skiprows=1,comments='#')
    data_points=len(f_original)-1#the first row is not data but some q corr information
    LB=np.array([2]*data_points)[:,np.newaxis]
    dL=np.array([2]*data_points)[:,np.newaxis]
    #print f_original[1:,2]
    #print f_original[1:,2][:,np.newaxis]*f_original[0,0]/2./np.pi
    data_formated=f_original[1:,2][:,np.newaxis]*f_original[0,0]/2./np.pi#recaculate L based on q data (matlab script correct only q column but not update L column as well)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,f_original[1:,3][:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,f_original[1:,4][:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,LB,axis=1)
    data_formated=np.append(data_formated,dL,axis=1)
    np.savetxt(file+'_GenX_formate.dat',data_formated,fmt='%.5e')
    qsi_correction(file+'_GenX_formate.dat',L_column=0,I_column=4,correction_factor=np.pi*2/f_original[0,0])
    return None

def formate_RAXR_data(file_path='M:\\fwog\\members\\qiu05\\mica\\zr_mica_RAXR_L',E_range=[17934,18119]):
    full_data=np.zeros((1,8))
    L_list=['0041','0053','0061','0075','0088','0115','0145','0171','0231','0264','0285','0321','0355','0424','0455','0561','0625','0731','0915','1031','1115']

    segment_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #L_list=[L_list[0]]
    def _find_segment_index(data,segment_index):
        current_segment=0
        index_container=[]
        for i in range(len(data)):
            if i!=len(data)-1:
                if data[i+1,0]<data[i,0]:
                    current_segment+=1
            if current_segment==segment_index:
                index_container.append(i)
            else:
                pass
        return index_container

    for i in range(len(L_list)):
        file=file_path+L_list[i]+'a_RAXR_R.ipg'
        data=np.loadtxt(file,comments='%')
        index_segment_all=_find_segment_index(data,segment_list[i])
        if E_range==None:
            index_segment=index_segment_all
        else:
            index_segment=[i for i in index_segment_all if data[i,3]*1000<E_range[1] and data[i,3]*1000>E_range[0]]
        E=data[index_segment,3][:,np.newaxis]
        I=(data[index_segment,5]/data[index_segment,4])[:,np.newaxis]
        Ierr=(data[index_segment,6]/data[index_segment,4])[:,np.newaxis]
        BL=np.array([2]*len(index_segment))[:,np.newaxis]
        dL=np.array([2]*len(index_segment))[:,np.newaxis]
        H=np.array([0]*len(index_segment))[:,np.newaxis]
        K=np.array([0]*len(index_segment))[:,np.newaxis]
        L=np.around(data[index_segment,1][:,np.newaxis],2)
        temp_data=E*1000
        temp_data=np.append(temp_data,H,axis=1)
        temp_data=np.append(temp_data,K,axis=1)
        temp_data=np.append(temp_data,L,axis=1)
        temp_data=np.append(temp_data,I,axis=1)
        temp_data=np.append(temp_data,Ierr,axis=1)
        temp_data=np.append(temp_data,BL,axis=1)
        temp_data=np.append(temp_data,dL,axis=1)
        full_data=np.append(full_data,temp_data,axis=0)
    np.savetxt(file_path+'_GenX_formate.dat',full_data[1:],fmt='%.5e')

def formate_RAXR_data_APS(file_path='M:\\fwog\\members\\qiu05\\1608 - 13-IDC\\schmidt\\mica\\mica-zr_s2_shortt_1_RAXR_1st_spot1.ipg',E_range=[17934,18119]):
    full_data=np.zeros((1,8))
    L_list=[]
    data=np.loadtxt(file_path,comments='%')
    for i in range(len(data)):
        temp_data=[]
        if i!=(len(data)-1) and data[i,0]>data[i+1,0]:
            L_list.append(np.around(data[i,1],2))
            temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
        else:
            if np.around(data[i,1],2) not in L_list:
                temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
        if temp_data!=[] and np.around(data[i,3]*1000,0)>E_range[0] and np.around(data[i,3]*1000,0)<E_range[1]:
            full_data=np.append(full_data,temp_data,axis=0)
    print L_list
    np.savetxt(file_path.replace('.ipg','_GenX_formate.dat'),full_data[1:],fmt='%.5e')

def formate_RAXR_data_ESRF(file_path='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\S0_Zr_0mM_NaCl_Dry_RAXR_1st_spot1_R.ipg',E_range=[17934,18119],L_shift=0):
    #L_shift:after q correction, L should be corrected somehow. For example, it was L=0.3 while it is now L=0.255 after Q correction, then L_shift=-0.045
    full_data=np.zeros((1,8))
    L_list=[]
    data=np.loadtxt(file_path,comments='%')
    for i in range(len(data)):
        temp_data=[]
        if i!=(len(data)-1) and data[i,0]>data[i+1,0]:
            L_list.append(np.around(data[i,1],2))
            temp_data=[[np.around(data[i,3],0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4],data[i,6]/data[i,4],2,2]]
        else:
            if np.around(data[i,1],2) not in L_list:
                temp_data=[[np.around(data[i,3],0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4],data[i,6]/data[i,4],2,2]]
        if temp_data!=[] and np.around(data[i,3],0)>E_range[0] and np.around(data[i,3],0)<E_range[1]:
            full_data=np.append(full_data,temp_data,axis=0)
    print L_list
    full_data[:,0]=full_data[:,0]+L_shift
    np.savetxt(file_path.replace('.ipg','_GenX_formate.dat'),full_data[1:],fmt='%.5e')

def scale_RAXS_data_to_CTR(file_ctr='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\nQc_S0_Zr_0mM_NaCl_Dry_CTR_1st_spot1_R_GenX_formate_Q_corrected.dat',file_raxs='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\S0_Zr_0mM_NaCl_Dry_RAXR_1st_spot1_R_GenX_formate.dat',E_col=0,L_col_raxs=3,E_ctr=16000):
    f_ctr=np.loadtxt(file_ctr)
    f_raxs=np.loadtxt(file_raxs)
    current_L=f_raxs[0,L_col_raxs]
    I_new=f_raxs[0,4]
    scaling=None
    for i in range(len(f_raxs)):
        if abs(f_raxs[i,L_col_raxs]-current_L)>0.01:#a sign move to next segment
            current_L=f_raxs[i,L_col_raxs]
            I_new=f_raxs[i,4]
            scaling=f_ctr[np.argmin(abs(f_ctr[:,0]-current_L)),4]/I_new
        elif abs(f_raxs[i,L_col_raxs]-current_L)<0.01 and scaling==None:#a sign of first segment
            current_L=f_raxs[i,L_col_raxs]
            I_new=f_raxs[0,4]
            scaling=f_ctr[np.argmin(abs(f_ctr[:,0]-current_L)),4]/I_new
        else:
            pass
        f_raxs[i,4]=f_raxs[i,4]*scaling#scaling I
        f_raxs[i,5]=f_raxs[i,5]*scaling#scaling error
    np.savetxt(file_raxs.replace('.dat','_scaled_to_CTR.data'),f_raxs,fmt='%.5e')

def formate_F1F2_data(f1f2_file='M:\\fwog\\members\\qiu05\\mica\\axd_Zr_k.002.nor',ipg_file='M:\\fwog\\members\\qiu05\\mica\\zr_mica_RAXR_L321.ipg'):
    f1f2=np.loadtxt(f1f2_file)
    ipg=np.loadtxt(ipg_file,comments='%')
    E_list=[]
    for i in range(len(ipg)):
        if ipg[i,0]<ipg[i-1,0] and i!=0:
            break
        E_list.append(round(ipg[i,3]*1000,0))

    f1f2_new=np.zeros((1,3))
    print E_list
    for i in range(len(f1f2)):
        if round(f1f2[i,0]*1000,0) in E_list:
            f1f2_new=np.append(f1f2_new,(f1f2[i]*[1000,1,1])[np.newaxis,:],axis=0)
    f1f2_new=f1f2_new[1:]
    np.savetxt(f1f2_file+'.formated',f1f2_new[:,[1,2,0]])
    return None

def formate_F1F2_data_ESRF(f1f2_file='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\axd_Zr_k.002.nor',ipg_file='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\S0_Zr_0mM_NaCl_Dry_RAXR_1st_spot1_R.ipg'):
    f1f2=np.loadtxt(f1f2_file)
    ipg=np.loadtxt(ipg_file,comments='%')
    E_list=[]
    for i in range(len(ipg)):
        if ipg[i,0]<ipg[i-1,0] and i!=0:
            break
        E_list.append(round(ipg[i,3],0))

    f1f2_new=np.zeros((1,3))
    print E_list
    for i in range(len(f1f2)):
        if round(f1f2[i,0],0) in E_list:
            f1f2_new=np.append(f1f2_new,f1f2[i][np.newaxis,:],axis=0)
    f1f2_new=f1f2_new[1:]
    np.savetxt(f1f2_file+'.formated',f1f2_new[:,[1,2,0]])
    return None

if __name__=='__main__':
    formate_CTR_data()
    formate_RAXR_data(E_range=[17920,18160])
    formate_F1F2_data()
import matplotlib.pyplot as pyplot
import numpy as np
import os

file_path='M:\\fwog\\members\\qiu05\\1607 - BM20\\ipg files'#ESRF
file_path='M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica'#APS
file_names=['mica_s2_a_CTR_1st_spot_R.ipg','mica_s2b_a_CTR_1st_spot_R.ipg','mica_s2c_c_CTR_1st_spot_R.ipg']
file_names=['mica_s2_a_RAXR_1st_spot2_R.ipg','mica_s2b_a_RAXR_1st_spot2_R.ipg','mica_s2c_c_RAXR_3rd_spot_R.ipg']
file_names=['mica-zr_s2_longt_1_RAXR_1st_spot1.ipg','mica-zr_s2b_1_RAXR_2nd_spot1.ipg','mica-zr_s2c_longt_1_RAXR_1st_spot1.ipg']
file_path='M:\\fwog\\members\\qiu05\\1611 - ROBL20'
file_names=['S0_Zr_0mM_NaCl_Dry_CTR_1st_spot1_R.ipg']
file_path='M:\\fwog\\members\\qiu05\\1611 - ROBL20'
file_names=['S3_100mM_NH4Cl_new_RAXR_1st_spot1_R.ipg']
file_path='M:\\fwog\\members\\qiu05\\1608 - 13-IDC\\schmidt\mica'
file_names=['mica-zr_s2_longt_1_RAXR_1st_spot1.ipg']
#file_path='M:\\fwog\\members\\qiu05\\1607 - BM20\\ipg files'
#file_names=['mica_s2_a_RAXR_1st_spot2_R.ipg']

file_path=['C:\\Users\\qiu05\\Downloads']
file_names=['HfZr-S1-12h_RAXR2_R.ipg']


labels=['Zr_100mM_NaCl','Zr_100mM_LiCl']

plot_type='RAXR'
x_column=3
y_column=5
L_column=1
y_error_column=6
y_norm_column=4
num_data_point_raxr=[43,43]
fig=pyplot.figure()
ax=[]
if plot_type=='CTR':
    ax.append(fig.add_subplot(1,1,1))
    ax[0].set_yscale('log')
else:
    for i in range(9):
        ax.append(fig.add_subplot(3,3,i+1))
    
for i in range(len(file_names)):
    file=os.path.join(file_path[i],file_names[i])
    data=np.loadtxt(file,comments='%')
    x,y,y_error,y_norm=data[:,x_column],data[:,y_column],data[:,y_error_column],data[:,y_norm_column]
    if plot_type=='CTR':
        ax[0].errorbar(x,y/y_norm,y_error/y_norm,label=labels[i],fmt='.')
        if i==len(file_names)-1:
            pyplot.xlabel('L',axes=ax[0])
            pyplot.ylabel('|F|',axes=ax[0])
    else:
        if x[0]<100:#energy in KeV
            x=x*1000
        for j in range(9):
            shift=y[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)][10]/y_norm[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)][10]-i*0.1
            ax[j].errorbar(x[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)]/1000,y[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)]/y_norm[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)]-shift,y_error[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)]/y_norm[num_data_point_raxr[i]*j:num_data_point_raxr[i]*(j+1)],fmt='*',label=labels[i])
            if i==len(file_names)-1:
                ax[j].set_title('L='+str(round(data[:,L_column][num_data_point_raxr[i]*j],2)),fontsize=10)
                if j==0 or j==3 or j==6:
                    ax[j].set_ylabel('|F|')
                if j in [6,7,8]:
                    ax[j].set_xlabel('Energy (KeV)')
#pyplot.legend()
pyplot.show()


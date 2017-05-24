import os,sys
sys.path.append('P://apps//genx_pc_qiu')
sys.path.append('P://apps//genx_pc_qiu//models')
sys.path.append('P://apps//genx_pc_qiu//lib')
import numpy as np

import model,diffev
import filehandling as io

path='P:\\My stuff\\Models\\CTR models\\Zr models for final publication\\refit results\\GenX'
file_path_raxr_mi=os.path.join(path,'MD_RAXR_refit_Zr_0mM_NaCl_6O_run1_Apr3_best_1_bin_R1_weighted_2_kr0.90_km0.90_pf0.80_run1_ran_merged.gx')
scan_number=50
error_bar_level=1.05

mod_raxr_mi = model.Model()
config_raxr_mi = io.Config()
opt_raxr_mi = diffev.DiffEv()

io.load_gx(file_path_raxr_mi,mod_raxr_mi,opt_raxr_mi,config_raxr_mi)

first_grid_mi,first_grid_md=None,None

print 'scan number is ',scan_number
print 'error bar level is ',error_bar_level

mod_raxr_mi.simulate()
best_fom=mod_raxr_mi.fom
for i in range(len(mod_raxr_mi.parameters.data)):
    if mod_raxr_mi.parameters.data[i][2]==True:
        print 'scan ',mod_raxr_mi.parameters.data[i][0],' now'

        left,right=float(mod_raxr_mi.parameters.data[i][3]),float(mod_raxr_mi.parameters.data[i][4])
        values=np.arange(left,right,(right-left)/scan_number)
        best=float(mod_raxr_mi.parameters.data[i][1])
        fom_container=[]
        for value in values:
            mod_raxr_mi.parameters.data[i][1]=value
            mod_raxr_mi.simulate()
            fom_container.append(mod_raxr_mi.fom)
            if not np.where(values==value)[0][0]%10:
                print 'doing scan ',np.where(values==value)[0][0],'FOM=',mod_raxr_mi.fom
        values_sub=np.compress(np.array(fom_container) < best_fom*error_bar_level,np.array(values))
        if len(values_sub)==0:
            mod_raxr_mi.parameters.data[i][1]=best
            print 'the range of this parameter is too large, either make a finer scan or shrink the range'
        else:
            left_error,right_error=np.min(values_sub)-best,np.max(values_sub)-best
            if right_error<0:
                right_error=right-best
            mod_raxr_mi.parameters.data[i][1]=best
            mod_raxr_mi.parameters.data[i][5]='(%3.6f,%3.6f)'%(left_error,right_error)
            #mod_raxr_mi.simulate()
            print 'after scan fom is ',mod_raxr_mi.fom
            print 'the error bar based on FOM scan is: %3.6f(%3.6f,%3.6f)'%(best,left_error,right_error)

        io.save_gx(file_path_raxr_mi.replace('.gx','_error_calculated.gx'),mod_raxr_mi,opt_raxr_mi,config_raxr_mi)
import os
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import models.sxrd_new1 as model
from models.utils import UserVars
import batchfile.locate_path as batch_path
import models.domain_creator as domain_creator
import supportive_functions.create_plots as create_plots
import models.domain_creator_sorbate as domain_creator_sorbate
import supportive_functions.make_parameter_table_GenX_5_beta as make_grid

##==========================================<program begins from here>=========================================##
COUNT_TIME=False
if COUNT_TIME:t_0=datetime.now()

##<global handles>##
RUN=False
BATCH_PATH_HEAD,OUTPUT_FILE_PATH=batch_path.module_path_locator(),'D:\\'
F1F2=np.loadtxt(os.path.join(BATCH_PATH_HEAD,'Zr_K_edge.f1f2'))
RAXR_EL,E0,NUMBER_RAXS_SPECTRA,RAXR_FIT_MODE='Zr',18007,21,'MD'
NUMBER_DOMAIN,COHERENCE=2,True
HEIGHT_OFFSET=-2.6685#if set to 0, the top atomic layer is at 2.6685 in fractional unit before relaxation
GROUP_SCHEME=[[1,0]]#means group Domain1 and Domain2 for inplane and out of plane movement, set Domain2=Domain1 in side sim func

##<setting slabs>##
unitcell = model.UnitCell(5.1988, 9.0266, 20.1058, 90, 95.782, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk, Domain1, Domain2 = model.Slab(T_factor='u'), model.Slab(c = 1.0,T_factor='u'), model.Slab(c = 1.0,T_factor='u')
domain_creator.add_atom_in_slab(bulk,os.path.join(BATCH_PATH_HEAD,'muscovite_001_bulk.str'),height_offset=HEIGHT_OFFSET)
domain_creator.add_atom_in_slab(Domain1,os.path.join(BATCH_PATH_HEAD,'muscovite_001_surface_Al.str'),attach='_D1',height_offset=HEIGHT_OFFSET)
domain_creator.add_atom_in_slab(Domain2,os.path.join(BATCH_PATH_HEAD,'muscovite_001_surface_Si.str'),attach='_D2',height_offset=HEIGHT_OFFSET)

##<coordination system definition>##
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
BASIS=np.array([5.1988, 9.0266, 20.1058])
BASIS_SET=[[1,0,0],[0,1,0],[0.10126,0,1.0051136]]
T=inv(np.transpose(f1(x0_v,y0_v,z0_v,*BASIS_SET)))
T_INV=inv(T)

##<Adding sorbates>##to be set##
#domain1
LEVEL,CAP,SYMMETRY=13,[],False
NUMBER_SORBATE_LAYER,NUMBER_EL_MOTIF=4,1+len(CAP)*2#1 if monomer, 2 if dimmer and so on
INFO_LIB={'basis':BASIS,'sorbate_el':'Zr','coordinate_el':'O','T':T,'T_INV':T_INV,'oligomer_type':'monomer'}

for i in range(NUMBER_SORBATE_LAYER):
    vars()['rgh_domain1_set'+str(i+1)]=UserVars() 
    geo_lib_domain1={'cent_point_offset_x':0,'cent_point_offset_y':0,'cent_point_offset_z':0,'r':2.2,'theta':59.2641329,'rot_x':0,'rot_y':0,'rot_z':0,'shift_btop':0,'shift_mid':0,'shift_cap':0}
    Domain1,vars()['rgh_domain1_set'+str(i+1)]=domain_creator.add_sorbate(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,geo_lib=geo_lib_domain1,info_lib=INFO_LIB,domain_tag='_D1',rgh=vars()['rgh_domain1_set'+str(i+1)],index_offset=[i*2*NUMBER_EL_MOTIF,NUMBER_EL_MOTIF+i*2*NUMBER_EL_MOTIF],height_offset=HEIGHT_OFFSET,level=LEVEL,symmetry_couple=SYMMETRY,cap=CAP)

##<Adding Gaussian peaks>##
NUMBER_GAUSSIAN_PEAK, EL_GAUSSIAN_PEAK, FIRST_PEAK_HEIGHT=0,'O',4
GAUSSIAN_OCC_INIT, GAUSSIAN_LAYER_SPACING, GAUSSIAN_U_INIT=1,2,0.1
Domain1, Gaussian_groups,Gaussian_group_names=domain_creator.add_gaussian(domain=Domain1,el=EL_GAUSSIAN_PEAK,number=NUMBER_GAUSSIAN_PEAK,first_peak_height=FIRST_PEAK_HEIGHT,spacing=GAUSSIAN_LAYER_SPACING,u_init=GAUSSIAN_U_INIT,occ_init=GAUSSIAN_OCC_INIT,height_offset=HEIGHT_OFFSET,c=unitcell.c,domain_tag='_D1')
for i in range(len(Gaussian_groups)):vars()[Gaussian_group_names[i]]=Gaussian_groups[i]

##<Adding absorbed water>##to be set## (no adsorbed water at the moment)
#Domain1,absorbed_water_pair1_D1=domain_creator.add_oxygen_pair_muscovite(domain=Domain1,ids=['O1a_W_D1','O1b_W_D1'],coors=np.array([[0,0,2.2+HEIGHT_OFFSET],[0.5,0.5,2.2+HEIGHT_OFFSET]]))

##<Define atom groups>##
#surface atoms
group_number=5##to be set##(number of groups to be considered for model fit)
groups,group_names,atom_group_info=domain_creator.setup_atom_group_muscovite(domain=[Domain1,Domain2],group_number=group_number)
for i in range(len(groups)):vars()[group_names[i]]=groups[i]
#sorbate_atoms
sorbate_id_list_domain1,sorbate_group_names_domain1=domain_creator.generate_sorbate_ids(Domain1,NUMBER_SORBATE_LAYER,INFO_LIB['sorbate_el'],NUMBER_EL_MOTIF,symmetry=SYMMETRY,level=CAP)
sorbate_atom_group_info=[{'domain':Domain1,'ref_id_list':sorbate_id_list_domain1,'ref_group_names':sorbate_group_names_domain1,'ref_sym_list':[],'domain_tag':''}]
sorbate_groups,sorbate_group_names=domain_creator.setup_atom_group(gp_info=sorbate_atom_group_info)
for i in range(len(sorbate_groups)):vars()[sorbate_group_names[i]]=sorbate_groups[i]
    
##<Define other pars>##
rgh=domain_creator.define_global_vars(rgh=UserVars(),domain_number=NUMBER_DOMAIN)#global vars
rgh_raxs=domain_creator.define_raxs_vars(rgh=UserVars(),number_spectra=NUMBER_RAXS_SPECTRA,number_domain=1)#RAXR spectra pars
rgh_dlw=domain_creator.define_diffused_layer_water_vars(rgh=UserVars())#Diffused Layered water pars
rgh_dls=domain_creator.define_diffused_layer_sorbate_vars(rgh=UserVars())#Diffused Layered sorbate pars

##<make fit table file>##
if not RUN:
    table_container=[]
    rgh_instance_list=[rgh]+groups+sorbate_groups+Gaussian_groups+[vars()['rgh_domain1_set'+str(i+1)] for i in range(NUMBER_SORBATE_LAYER)]+[rgh_dlw,rgh_dls]
    rgh_instance_name_list=['rgh']+group_names+sorbate_group_names+Gaussian_group_names+['rgh_domain1_set'+str(i+1) for i in range(NUMBER_SORBATE_LAYER)]+['rgh_dlw','rgh_dls']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(BATCH_PATH_HEAD,'pars_ranges.txt'))
    #raxs pars
    table_container=make_grid.set_table_input_raxs(container=table_container,rgh_group_instance=rgh_raxs,rgh_group_instance_name='rgh_raxs',par_range={'a':[0,2],'b':[0,2],'c':[0,1],'A':[0,2],'P':[0,1]},number_spectra=NUMBER_RAXS_SPECTRA,number_domain=1)
    #build up the tab file
    make_grid.make_table(container=table_container,file_path=OUTPUT_FILE_PATH+'par_table.tab')

##<fitting function part>##
if COUNT_TIME:t_1=datetime.now()
VARS=vars()
def Sim(data,VARS=VARS):

    ##<Extract pars>##
    layered_water_pars=vars(rgh_dlw)
    layered_sorbate_pars=vars(rgh_dls)
    raxs_vars=vars(rgh_raxs)
    
    ##<update sorbates>##
    [domain_creator.update_sorbate(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,info_lib=INFO_LIB,domain_tag='_D1',rgh=VARS['rgh_domain1_set'+str(i+1)],index_offset=[i*2*NUMBER_EL_MOTIF,NUMBER_EL_MOTIF+i*2*NUMBER_EL_MOTIF],height_offset=HEIGHT_OFFSET,level=LEVEL,symmetry_couple=SYMMETRY,cap=CAP) for i in range(NUMBER_SORBATE_LAYER)]#domain1
    
    ##<link groups>##
    [eval(each_command) for each_command in domain_creator.link_atom_group(gp_info=atom_group_info,gp_scheme=GROUP_SCHEME)]
    
    ##<format domains>##
    domain={'domains':[Domain1,Domain2],'layered_water_pars':layered_water_pars,'layered_sorbate_pars':layered_sorbate_pars,\
            'global_vars':rgh,'raxs_vars':raxs_vars,'F1F2':F1F2,'E0':E0,'el':RAXR_EL}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.})
    
    ##<calculate structure factor>##
    F,fom_scaler=[],[]
    i=0
    for data_set in data:
        f=np.array([])   
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        if x[0]>100:
            i+=1
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5
        else:
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(x-LB)/dL)**2)**0.5
        f=rough*abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=RAXR_FIT_MODE,height_offset=HEIGHT_OFFSET*BASIS[2]))
        F.append(f*f)
        fom_scaler.append(1)
        
    if COUNT_TIME:
        t_2=datetime.now()

    ##<print structure/plotting files>##
    if not RUN:
        domain_creator.print_structure_files_muscovite(domain_list=[Domain1,Domain2],z_shift=0.8+HEIGHT_OFFSET,matrix_info=INFO_LIB,save_file='D://')
        create_plots.generate_plot_files(output_file_path=OUTPUT_FILE_PATH,sample=sample,rgh=rgh,data=data,fit_mode=RAXR_FIT_MODE,z_min=0,z_max=50,RAXR_HKL=[0,0,20],height_offset=HEIGHT_OFFSET*BASIS[2])
        #then do this command inside shell to extract the errors for A and P: model.script_module.create_plots.append_errors_for_A_P(par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs') 
  
    if COUNT_TIME:
        t_3=datetime.now()
        print "It took "+str(t_1-t_0)+" seconds to setup"
        print "It took "+str(t_2-t_1)+" seconds to do calculation for one generation"
        print "It took "+str(t_3-t_2)+" seconds to generate output files"
    return F,1,fom_scaler
    ##========================================<program ends here>========================================================##
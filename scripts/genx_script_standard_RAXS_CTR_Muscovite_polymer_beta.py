import models.sxrd_new1 as model
import models.raxs as model2
from models.utils import UserVars
from datetime import datetime
import numpy as np
from numpy.linalg import inv
import sys,pickle,__main__
import models.domain_creator as domain_creator
import models.domain_creator_sorbate as domain_creator_sorbate
import supportive_functions.make_parameter_table_GenX_5_beta as make_grid
import supportive_functions.formate_xyz_to_vtk as xyz
import supportive_functions.create_plots as create_plots
from copy import deepcopy

##************************************<program begins from here>**********************************************##
COUNT_TIME=False
if COUNT_TIME:t_0=datetime.now()

##<coordination system definition>##
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
BASIS=np.array([5.1988, 9.0266, 20.1058])
BASIS_SET=[[1,0,0],[0,1,0],[0.10126,0,1.0051136]]
T=inv(np.transpose(f1(x0_v,y0_v,z0_v,*BASIS_SET)))
T_INV=inv(T)

##<global handles>##
RUN=False##to be set##
BATCH_PATH_HEAD='P:\\apps\\genx_pc_qiu\\batchfile\\'##to be set##
OUTPUT_FILE_PATH='D:\\'
F1F2=np.loadtxt(BATCH_PATH_HEAD+'f1f2_temp.f1f2')
E0=18007##to be set##(absorbtion edge in eV)
NUMBER_RAXS_SPECTRA=28##to be set##
NUMBER_DOMAIN=2##to be set##
COHERENCE=True
RAXR_EL='Zr'##to be set##
RAXR_FIT_MODE='MD'##to be set##
HEIGHT_OFFSET=-2.6685#if set to 0, the top atomic layer is at 2.6685 in fractional unit before relaxation
INFO_LIB={'basis':BASIS,'sorbate_el':'Zr','coordinate_el':'O','T':T,'T_INV':T_INV,'oligomer_type':'tetramer'}##to be set##

##<setting slabs>##
unitcell = model.UnitCell(5.1988, 9.0266, 20.1058, 90, 95.782, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='u')#bulk
domain_creator.add_atom_in_slab(bulk,BATCH_PATH_HEAD+'muscovite_001_bulk.str',height_offset=HEIGHT_OFFSET)
Domain1 =  model.Slab(c = 1.0,T_factor='u')#surface slabs-Domain1
domain_creator.add_atom_in_slab(Domain1,BATCH_PATH_HEAD+'muscovite_001_surface_Al.str',attach='_D1',height_offset=HEIGHT_OFFSET)
Domain2 =  model.Slab(c = 1.0,T_factor='u')#surface slabs-Domain2
domain_creator.add_atom_in_slab(Domain2,BATCH_PATH_HEAD+'muscovite_001_surface_Si.str',attach='_D2',height_offset=HEIGHT_OFFSET)
#You can add more domains

##<Adding sorbates>##to be set##
#domain1
rgh_domain1=UserVars()
geo_lib_domain1={'cent_point_offset_x':0,'cent_point_offset_y':0,'cent_point_offset_z':0,'r':2.2,'theta':59.2641329,'rot_x':0,'rot_y':0,'rot_z':0}
Domain1,rgh_domain1=domain_creator.add_sorbate(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,geo_lib=geo_lib_domain1,info_lib=INFO_LIB,domain_tag='_D1',rgh=rgh_domain1,index_offset=[0,1],height_offset=HEIGHT_OFFSET)
#domain2
rgh_domain2=UserVars()
geo_lib_domain2={'cent_point_offset_x':0,'cent_point_offset_y':0,'cent_point_offset_z':0,'r':2.2,'theta':59.2641329,'rot_x':0,'rot_y':0,'rot_z':0}
Domain2,rgh_domain2=domain_creator.add_sorbate(domain=Domain2,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,geo_lib=geo_lib_domain2,info_lib=INFO_LIB,domain_tag='_D2',rgh=rgh_domain2,index_offset=[0,1],height_offset=HEIGHT_OFFSET)
#You can add more domains

##<Adding absorbed water>##to be set##
#domain1
Domain1,absorbed_water_pair1_D1=domain_creator.add_oxygen_pair_muscovite(domain=Domain1,ids=['O1a_W_D1','O1b_W_D1'],coors=np.array([[0,0,2.2+HEIGHT_OFFSET],[0.5,0.5,2.2+HEIGHT_OFFSET]]))
Domain1,absorbed_water_pair2_D1=domain_creator.add_oxygen_pair_muscovite(domain=Domain1,ids=['O2a_W_D1','O2b_W_D1'],coors=np.array([[0,0,2.3+HEIGHT_OFFSET],[0.5,0.5,2.3+HEIGHT_OFFSET]]))
#domain2
Domain2,absorbed_water_pair1_D2=domain_creator.add_oxygen_pair_muscovite(domain=Domain2,ids=['O1a_W_D2','O1b_W_D2'],coors=np.array([[0,0,2.2+HEIGHT_OFFSET],[0.5,0.5,2.2+HEIGHT_OFFSET]]))
Domain2,absorbed_water_pair2_D2=domain_creator.add_oxygen_pair_muscovite(domain=Domain2,ids=['O2a_W_D2','O2b_W_D2'],coors=np.array([[0,0,2.3+HEIGHT_OFFSET],[0.5,0.5,2.3+HEIGHT_OFFSET]]))

##<Define atom groups>##
#surface atoms
group_number=5##to be set##(number of groups to be considered for model fit)
ref_id_list_Al=[['O4_3_0','O4_4_0'],['O3_3_0','O3_4_0'],['O5_3_0','O5_4_0'],['Al1_3_0','Al1_4_0'],['Al2_3_0','Al2_4_0'],['O1_3_0','O1_4_0'],['O2_3_0','O2_4_0'],['O6_3_0','O6_4_0'],\
             ['Al3_3_0','Al3_4_0'],['Al3_5_0','Al3_6_0'],['O6_5_0','O6_6_0'],['O2_5_0','O2_6_0'],['O1_5_0','O1_6_0'],['Al2_5_0','Al2_6_0'],['Al1_5_0','Al1_6_0'],['O5_5_0','O5_6_0'],['O4_5_0','O4_6_0'],['O3_5_0','O3_6_0']]
ref_id_list_Si=[['O4_3_0','O4_4_0'],['O3_3_0','O3_4_0'],['O5_3_0','O5_4_0'],['Si1_3_0','Si1_4_0'],['Si2_3_0','Si2_4_0'],['O1_3_0','O1_4_0'],['O2_3_0','O2_4_0'],['O6_3_0','O6_4_0'],\
             ['Al3_3_0','Al3_4_0'],['Al3_5_0','Al3_6_0'],['O6_5_0','O6_6_0'],['O2_5_0','O2_6_0'],['O1_5_0','O1_6_0'],['Si2_5_0','Si2_6_0'],['Si1_5_0','Si1_6_0'],['O5_5_0','O5_6_0'],['O4_5_0','O4_6_0'],['O3_5_0','O3_6_0']]
ref_sym_list=[[[1,0,0,0,1,0,0,0,1]]*2]*18
ref_group_names_Al=['gp_O4_O4O3','gp_O3_O4O3','gp_O5_O3O4','gp_Al1_Al4Al3','gp_Al2_Al3Al4','gp_O1_O4O3','gp_O2_O3O4','gp_O6_O3O4','gp_Al3_Al4Al3','gp_Al3_Al6Al5','gp_O6_O5O6','gp_O2_O5O6','gp_O1_O6O5','gp_Al2_Al5Al6','gp_Al1_Al6Al5','gp_O5_O5O6','gp_O4_O6O5','gp_O3_O6O5']
ref_group_names_Si=['gp_O4_O4O3','gp_O3_O4O3','gp_O5_O3O4','gp_Si1_Al4Al3','gp_Si2_Al3Al4','gp_O1_O4O3','gp_O2_O3O4','gp_O6_O3O4','gp_Al3_Al4Al3','gp_Al3_Al6Al5','gp_O6_O5O6','gp_O2_O5O6','gp_O1_O6O5','gp_Si2_Al5Al6','gp_Si1_Al6Al5','gp_O5_O5O6','gp_O4_O6O5','gp_O3_O6O5']
atom_group_info=[{'domain':Domain1,'ref_id_list':ref_id_list_Al[0:group_number],'ref_group_names':ref_group_names_Al[0:group_number],'ref_sym_list':ref_sym_list[0:group_number],'domain_tag':'_D1'},
                 {'domain':Domain2,'ref_id_list':ref_id_list_Si[0:group_number],'ref_group_names':ref_group_names_Si[0:group_number],'ref_sym_list':ref_sym_list[0:group_number],'domain_tag':'_D2'}]
groups,group_names=domain_creator.setup_atom_group(gp_info=atom_group_info)
for i in range(len(groups)):
    vars()[group_names[i]]=groups[i]
    
#sorbate_atoms
sorbate_id_list_domain1=[[id for id in Domain1.id if INFO_LIB['sorbate_el'] in id],[id for id in Domain1.id if INFO_LIB['sorbate_el'] in id and 'O' not in id],[id for id in Domain1.id if INFO_LIB['sorbate_el'] in id and 'O' in id]]
sorbate_id_list_domain2=[[id for id in Domain2.id if INFO_LIB['sorbate_el'] in id],[id for id in Domain2.id if INFO_LIB['sorbate_el'] in id and 'O' not in id],[id for id in Domain2.id if INFO_LIB['sorbate_el'] in id and 'O' in id]]

sorbate_sym_list_domain1=[]
sorbate_sym_list_domain2=[]

sorbate_group_names_domain1=['sorbate_D1',INFO_LIB['sorbate_el']+'_D1','HO_D1']
sorbate_group_names_domain2=['sorbate_D2',INFO_LIB['sorbate_el']+'_D2','HO_D2']

sorbate_atom_group_info=[{'domain':Domain1,'ref_id_list':sorbate_id_list_domain1,'ref_group_names':sorbate_group_names_domain1,'ref_sym_list':sorbate_sym_list_domain1,'domain_tag':''},
                         {'domain':Domain2,'ref_id_list':sorbate_id_list_domain2,'ref_group_names':sorbate_group_names_domain2,'ref_sym_list':sorbate_sym_list_domain2,'domain_tag':''}]
sorbate_groups,sorbate_group_names=domain_creator.setup_atom_group(gp_info=sorbate_atom_group_info)
for i in range(len(sorbate_groups)):
    vars()[sorbate_group_names[i]]=sorbate_groups[i]
    
##<Define other pars>##
rgh=domain_creator.define_global_vars(rgh=UserVars(),domain_number=NUMBER_DOMAIN)#global vars
rgh_raxs=domain_creator.define_raxs_vars(rgh=UserVars(),number_spectra=NUMBER_RAXS_SPECTRA,number_domain=NUMBER_DOMAIN)#RAXR spectra pars
rgh_dlw=domain_creator.define_diffused_layer_water_vars(rgh=UserVars())#Diffused Layered water pars
rgh_dls=domain_creator.define_diffused_layer_sorbate_vars(rgh=UserVars())#Diffused Layered sorbate pars

##<make fit table file>##
if not RUN:
    table_container=[]
    rgh_instance_list=[rgh]+groups+[absorbed_water_pair1_D1,absorbed_water_pair2_D1,absorbed_water_pair1_D2,absorbed_water_pair2_D2]+\
                      sorbate_groups+[rgh_domain1,rgh_domain2,rgh_dlw,rgh_dls]
    rgh_instance_name_list=['rgh']+group_names+['absorbed_water_pair1_D1','absorbed_water_pair2_D1','absorbed_water_pair1_D2','absorbed_water_pair2_D2']+\
                           sorbate_group_names+['rgh_domain1','rgh_domain2','rgh_dlw','rgh_dls']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=BATCH_PATH_HEAD+'pars_ranges.txt')
    #raxs pars
    table_container=make_grid.set_table_input_raxs(container=table_container,rgh_group_instance=rgh_raxs,rgh_group_instance_name='rgh_raxs',par_range={'a':[0,1],'b':[0,1],'c':[0,1],'A':[0,1],'P':[0,1]},number_spectra=NUMBER_RAXS_SPECTRA,number_domain=NUMBER_DOMAIN)
    #build up the tab file
    make_grid.make_table(container=table_container,file_path=OUTPUT_FILE_PATH+'par_table.tab')

##<fitting function part>##
if COUNT_TIME:t_1=datetime.now()
def Sim(data):

    ##<Extract pars>##
    layered_water_pars=vars(rgh_dlw)
    layered_sorbate_pars=vars(rgh_dls)
    raxs_vars=vars(rgh_raxs)
    
    ##<update sorbates>##
    domain_creator.update_sorbate(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,info_lib=INFO_LIB,domain_tag='_D1',rgh=rgh_domain1,index_offset=[0,1],height_offset=HEIGHT_OFFSET)#domain1
    domain_creator.update_sorbate(domain=Domain2,anchored_atoms=[],func=domain_creator_sorbate.OS_sqr_antiprism_oligomer,info_lib=INFO_LIB,domain_tag='_D2',rgh=rgh_domain2,index_offset=[0,1],height_offset=HEIGHT_OFFSET)#domain2
    #You can add more domains
    
    ##<format domains>##
    domain={'domains':[Domain1,Domain2],'layered_water_pars':layered_water_pars,'layered_sorbate_pars':layered_sorbate_pars,\
            'global_vars':rgh,'raxs_vars':raxs_vars,'F1F2':F1F2,'E0':E0,'el':RAXR_EL}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.})
    
    ##<calculate structure factor for each dataset>##
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
        F.append(abs(f))
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
    ##******************************<program ends here>****************************************************##
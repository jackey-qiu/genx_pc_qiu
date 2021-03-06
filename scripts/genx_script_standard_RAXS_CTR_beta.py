import models.sxrd_new1 as model
import models.raxs as model2
from models.utils import UserVars
from datetime import datetime
import numpy as np
import sys,pickle,__main__,os
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
import supportive_functions.make_parameter_table_GenX_beta4 as make_grid
import supportive_functions.formate_xyz_to_vtk as xyz
from copy import deepcopy

#************************************program begins from here **********************************************
###############################################global vars##################################################
COUNT_TIME=False

if COUNT_TIME:t_0=datetime.now()

##matching index##
"""
*********************************************************************************************************************************
* Note the pickup index supposed to be used in a ternary complex structure have no concept of domain type, so you can           *
* freely mix using both. For example, the pickup_index item of [15,26,26] has no difference from [15,11,11] and also [15,11,26].*
*********************************************************************************************************************************

HL-->0          1           2           3             4              5(Face-sharing)     6           7
CS(O1O2)        CS(O2O3)    ES(O1O3)    ES(O1O4)      TD(O1O2O3)     TD(O1O3O4)          OS          Clean

TERNARY HL-->     8         9           10            11             12                  13          14
                  M(HO1)    M(HO2)      M(HO3)        B(HO1_HO2)     B(HO1_HO3)          B(HO2_HO3)  T(HO1_HO2_HO3)

FL-->15         16          17          18            19             20(Face-sharing)    21          22
CS(O5O6)        ES(O5O7)    ES(O5O8)    CS(O6O7)      TD(O5O6O7)     TD(O5O7O8)          OS          Clean

TERNARY FL-->   23        24          25            26             27                  28          29
                M(HO1)    M(HO2)      M(HO3)        B(HO1_HO2)     B(HO1_HO3)          B(HO2_HO3)  T(HO1_HO2_HO3)
"""
##############################################main setup zone###############################################
running_mode=1
USE_BV=False
COVALENT_HYDROGEN_RANDOM=False
COUNT_DISTAL_OXYGEN=False
ADD_DISTAL_LIGAND_WILD=[[False]*10]*10
BOND_VALENCE_WAIVER=[]
CONSIDER_WATER_IN_BV=False
if not CONSIDER_WATER_IN_BV:BOND_VALENCE_WAIVER=BOND_VALENCE_WAIVER+['Os'+str(ii+1) for ii in range(10)]

SORBATE=[['Sb','Pb'],['Pb'],['As','As','As']]
BASAL_EL=[[None]+each_domain[:-1] for each_domain in SORBATE]
pickup_index=[[4,14],[2],[15,26,26]]
sym_site_index=[[[0,1]]* len(each) for each in pickup_index]
half_layer=[3,3]#2 for short slab and 3 for long slab
full_layer=[1]#0 for short slab and 1 for long slab
half_layer_pick=half_layer+[None]*len(full_layer)
full_layer_pick=[None]*len(half_layer)+full_layer
OS_X_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[0]
OS_Y_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[1]
OS_Z_REF=domain_creator.init_OS_auto(pickup_index,half_layer+full_layer,OS_index=[6,21])[2]
DOMAINS_BV=range(len(pickup_index))
TABLE_DOMAINS=[1]*len(pickup_index)

RAXR_EL='Pb'
NUMBER_SPECTRA=0
RESONANT_EL_LIST=[1,0,0]
E0=13035
F1F2_FILE="f1f2_temp.f1f2"
F1F2=None

BV_OFFSET_SORBATE=[[0.2]*8]*len(pickup_index)
SEARCH_RANGE_OFFSET=0.3

USE_COORS=[[0,0,0,0]*10]*len(pickup_index)
COORS={(0,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]},\
       (2,0):{'sorbate':[[0,0,0]],'oxygen':[[0,0,0],[0,0,0]]}}

water_pars={'use_default':True,'number':[0,2,0],'ref_point':[[[]],[['O1_3_0','O1_4_0']],[[]]]}
layered_water_pars={'yes_OR_no':[0]*len(pickup_index),'ref_layer_height':[]}
layered_sorbate_pars={'yes_OR_no':[0]*len(pickup_index),'ref_layer_height':[],'el':''}

O_NUMBER_HL=[[4,4],[0,0],[1,1],[0,0],[3,3],[0,0],[3,3],[0,0]]
O_NUMBER_HL_EXTRA=[[0,0],[0,0],[0,0],[1,1],[1,1],[0,0],[0,0]]
O_NUMBER_FL=[[2,2],[0,0],[0,0],[0,0],[0,0],[0,0],[4,4],[0,0]]
O_NUMBER_FL_EXTRA=[[0,0],[0,0],[0,0],[2,2],[0,0],[0,0],[4,4]]

MIRROR=[[True for each_item in each] for each in pickup_index]

SORBATE_NUMBER_HL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_HL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]
SORBATE_NUMBER_FL=[[2],[2],[2],[2],[2],[2],[2],[0]]
SORBATE_NUMBER_FL_EXTRA=[[2],[2],[2],[2],[2],[2],[2]]

GROUPING_SCHEMES=[[1,0]]#domain tag of first domain is 0
GROUPING_DEPTH=[[0,10]]#means I will group top 10 (range(0,10)) layers of domain1 and domain2 together
commands_surface=domain_creator.generate_commands_for_surface_atom_grouping_new(np.array(GROUPING_SCHEMES)+1,domain_creator.translate_domain_type(GROUPING_SCHEMES,half_layer+full_layer),GROUPING_DEPTH)
commands_other=\
   [

   ]
commands=commands_other+commands_surface
#depository path for output files(structure model files(.xyz,.cif), optimized values (CTR,RAXR,E_Density) for plotting
output_file_path=output_path.module_path_locator()
##############################################end of main setup zone############################################

##file paths and wt factors##
WT_BV=1#weighting for bond valence constrain (1 recommended)
BV_TOLERANCE=[-0.2,0.2]#ideal bv value + or - this value is acceptable, negative side is over-saturation and positive site is under-saturated
USE_TOP_ANGLE=True#fit top angle if true otherwise fit the Pb-O bond length (used in bidentate case)
INCLUDE_HYDROGEN=0

##make a pick index list specifying the type of full layer (0 for short and 1 for long slab)
##this function will return a list of indexes list to be passed to pick_full_layer
def make_pick_index(full_layer_pick,pick,half_layer_cases=8,full_layer_cases=8):
    pick_index_all=[]
    for i in range(len(full_layer_pick)):
        pick_index=[1]*full_layer_cases
        if full_layer_pick[i]!=None:
            #pick_index[pick[i][0]-half_layer_cases]=full_layer_pick[i]
            for j in pick[i]:
                pick_index[j-half_layer_cases]=full_layer_pick[i]
            pick_index_all.append(pick_index)
        else:
            pass
    return pick_index_all

def make_pick_index_half_layer(half_layer_pick,pick,half_layer_cases=8):
    pick_index_all=[]
    for i in range(len(half_layer_pick)):
        pick_index=[2]*half_layer_cases
        if half_layer_pick[i]!=None:
            #pick_index[pick[i][0]]=half_layer_pick[i]
            for j in pick[i]:
                pick_index[j]=half_layer_pick[i]
            pick_index_all.append(pick_index)
        else:
            pass
    return pick_index_all
##pick the full layer cases according to the type of full layers(pick_index is a list of list created from make_pick_index)
def pick_full_layer(LFL=[],SFL=[],pick_index=[]):
    FL_all=[]
    for pick in pick_index:
        FL=[]
        for i in range(len(pick)):
            if pick[i]==0:
                FL.append(SFL[i])
            elif pick[i]==1:
                FL.append(LFL[i])
        FL_all.append(FL)
    return FL_all

def pick_half_layer(LHL=[],SHL=[],pick_index=[]):
    HL_all=[]
    for pick in pick_index:
        HL=[]
        for i in range(len(pick)):
            if pick[i]==2:
                HL.append(SHL[i])
            elif pick[i]==3:
                HL.append(LHL[i])
        HL_all.append(HL)
    return HL_all
##pick functions
def pick(pick_list,pick_index=pickup_index):
    picked_box=[]
    for each in pick_index:
        picked_box.append(pick_list[each[0]])
    return picked_box

def pick_act(pick_list,pick_index=pickup_index):
    picked_box=[]
    for each in pick_index:
        temp_box=[]
        for i in range(len(each)):
            temp_box=temp_box+pick_list[each[i]]
        picked_box.append(temp_box)
    return picked_box

def deep_pick(pick_list,sym_site_index=sym_site_index,pick_index=pickup_index):
    picked_box=[]
    for i in range(len(pick_index)):
        temp_box_j=[]
        for j in range(len(pick_index[i])):
            for k in sym_site_index[i][j]:
                temp_box_j.append(pick_list[pick_index[i][j]][k])
        picked_box.append(temp_box_j)
    return picked_box

#sorbate_el_list is a unique list of sorbate elements being considered in the model system
SORBATE_EL_LIST=[]
[SORBATE_EL_LIST.append(each_el) for each_el in sum(SORBATE,[]) if each_el not in SORBATE_EL_LIST]

FULL_LAYER_PICK_INDEX=make_pick_index(full_layer_pick=full_layer_pick,pick=pickup_index,half_layer_cases=15,full_layer_cases=15)
HALF_LAYER_PICK_INDEX=make_pick_index_half_layer(half_layer_pick=half_layer_pick,pick=pickup_index,half_layer_cases=15)
N_FL=len([i for i in full_layer_pick if i!=None])
N_HL=len(pickup_index)-N_FL
COHERENCE=[{True:range(len(pickup_index))}] #want to add up in coherence? items inside list corresponding to each domain
##cal bond valence switch##
SEARCH_MODE_FOR_SURFACE_ATOMS=True#If true then cal bond valence of surface atoms based on searching within a spherical region
METAL_VALENCE={'Pb':(2.,3.),'Sb':(5.,6.),'As':(5.,4.),'P':(5.,4.),'Cr':(6.,4.),'Cd':(2.,6.),'Cu':(2.,6.),'Zn':(2.,6.)}#for each value (valence charge,coordination number)
R0_BV={('As','O'):1.767,('Cr','O'):1.794,('Cd','O'):1.904,('Cu','O'):1.679,('Zn','O'):1.704,('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973,('P','O'):1.617}#r0 for different couples
IDEAL_BOND_LENGTH={('As','O'):1.68,('Cr','O'):1.64,('Cd','O'):2.31,('Cu','O'):2.09,('Zn','O'):2.11,('Fe','O'):2.02,('Pb','O'):2.19,('Sb','O'):2.04,('P','O'):1.534}#ideal bond length for each case
LOCAL_STRUCTURE_MATCH_LIB={'trigonal_pyramid':['Pb'],'octahedral':['Sb','Fe','Cd','Cu','Zn'],'tetrahedral':['As','P','Cr']}
debug_bv=not running_mode
##want to output the data for plotting?##
PLOT=not running_mode
##want to print out the protonation status?##
PRINT_PROTONATION=not running_mode
##want to print bond valence?##
PRINT_BV=not running_mode
##want to print the xyz files to build a 3D structure?##
PRINT_MODEL_FILES=not running_mode
##pars for sorbates##
LOCAL_STRUCTURE=deepcopy(SORBATE)
METAL_BV_EACH=deepcopy(SORBATE)
BOND_LENGTH_EACH=deepcopy(SORBATE)
for i in range(len(LOCAL_STRUCTURE)):
    for j in range(len(LOCAL_STRUCTURE[i])):
        METAL_BV_EACH[i][j]=METAL_VALENCE[SORBATE[i][j]][0]/METAL_VALENCE[SORBATE[i][j]][1]#valence for each bond
        BOND_LENGTH_EACH[i][j]=R0_BV[(SORBATE[i][j],'O')]-np.log(METAL_BV_EACH[i][j])*0.37#ideal bond length using bond valence equation
        for key in LOCAL_STRUCTURE_MATCH_LIB.keys():
            if LOCAL_STRUCTURE[i][j] in LOCAL_STRUCTURE_MATCH_LIB[key]:
                LOCAL_STRUCTURE[i][j]=key
                break
            else:pass

UPDATE_SORBATE_IN_SIM=True#you may not want to update the sorbate in sim function based on the frame of geometry, then turn this off
SORBATE_ATTACHE_ATOM_EXTRA=[[['HO1_'],['HO1_']],[['HO2_'],['HO2_']],[['HO3_'],['HO3_']],[['HO1_','HO2_'],['HO1_','HO2_']],[['HO1_','HO3_'],['HO1_','HO3_']],[['HO2_','HO3_'],['HO2_','HO3_']],[['HO1_','HO2_','HO3_'],['HO1_','HO2_','HO3_']]]
SORBATE_ATTACH_ATOM_HL_L=[[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']],[['O1_1_0','O1_4_0'],['O1_3_0','O1_2_0']],[['O1_1_0','O1_3_0'],['O1_4_0','O1_2_0']],[['O1_1_0','O1_4_0'],['O1_3_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0'],['O1_1_0','O1_2_0','O1_4_0']],[['O1_1_0','O1_3_0','O1_4_0'],['O1_2_0','O1_3_0','O1_4_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
SORBATE_ATTACH_ATOM_HL_S=[[['O1_7_0','O1_8_0'],['O1_7_0','O1_8_0']],[['O1_7_0','O1_10_0'],['O1_9_0','O1_8_0']],[['O1_7_0','O1_9_0'],['O1_10_0','O1_8_0']],[['O1_7_0','O1_10_0'],['O1_9_0','O1_8_0']],[['O1_8_0','O1_7_0','O1_9_0'],['O1_8_0','O1_7_0','O1_10_0']],[['O1_7_0','O1_9_0','O1_10_0'],['O1_8_0','O1_9_0','O1_10_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
SORBATE_ATTACH_ATOM_FL_L=[[['O1_11_t','O1_12_t'],['O1_11_t','O1_12_t']],[['O1_11_t','O1_1_0'],['O1_2_0','O1_12_t']],[['O1_11_t','O1_2_0'],['O1_1_0','O1_12_t']],[['O1_11_t','O1_1_0'],['O1_2_0','O1_12_t']],[['O1_11_t','O1_12_t','O1_2_0'],['O1_11_t','O1_12_t','O1_1_0']],[['O1_11_t','O1_2_0','O1_1_0'],['O1_12_t','O1_2_0','O1_1_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
SORBATE_ATTACH_ATOM_FL_S=[[['O1_5_0','O1_6_0'],['O1_5_0','O1_6_0']],[['O1_5_0','O1_7_0'],['O1_8_0','O1_6_0']],[['O1_5_0','O1_8_0'],['O1_7_0','O1_6_0']],[['O1_7_0','O1_5_0'],['O1_6_0','O1_8_0']],[['O1_6_0','O1_5_0','O1_8_0'],['O1_6_0','O1_5_0','O1_7_0']],[['O1_5_0','O1_7_0','O1_8_0'],['O1_6_0','O1_7_0','O1_8_0']],[[],[]],[[],[]]]+SORBATE_ATTACHE_ATOM_EXTRA
SORBATE_ATTACH_ATOM_FL=pick_full_layer(LFL=SORBATE_ATTACH_ATOM_FL_L,SFL=SORBATE_ATTACH_ATOM_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
SORBATE_ATTACH_ATOM_HL=pick_half_layer(LHL=SORBATE_ATTACH_ATOM_HL_L,SHL=SORBATE_ATTACH_ATOM_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
SORBATE_ATTACH_ATOM_SEPERATED=[deep_pick(SORBATE_ATTACH_ATOM_HL_L+each_FL) for each_FL in SORBATE_ATTACH_ATOM_FL]
SORBATE_ATTACH_ATOM_SEPERATED_HL=[deep_pick(each_HL+SORBATE_ATTACH_ATOM_FL_L) for each_HL in SORBATE_ATTACH_ATOM_HL]
SORBATE_ATTACH_ATOM=[SORBATE_ATTACH_ATOM_SEPERATED_HL[i][i] for i in range(N_HL)]+[SORBATE_ATTACH_ATOM_SEPERATED[i][N_HL+i] for i in range(N_FL)]

SORBATE_ATTACH_ATOM_OFFSET_EXTRA=[[[None],[None]],[[None],[None]],[[None],[None]],[[None,None],[None,None]],[[None,None],[None,None]],[[None,None],[None,None]],[[None,None,None],[None,None,None]]]
SORBATE_ATTACH_ATOM_OFFSET_HL_L=[[[None,None],[None,'+y']],[['-y','+x'],[None,None]],[[None,None],['+x',None]],[[None,'+y'],['+x',None]],[[None,None,None],['-y',None,'+x']],[[None,None,'+y'],['-x',None,None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
SORBATE_ATTACH_ATOM_OFFSET_HL_S=[[[None,None],[None,'+y']],[['-y','-x'],[None,None]],[[None,None],['-x',None]],[[None,'+y'],[None,'+x']],[[None,None,None],[None,'-y','-x']],[[None,None,'+y'],['+x',None,None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
SORBATE_ATTACH_ATOM_OFFSET_FL_L=[[[None,None],[None,'+y']],[[None,'-x'],[None,None]],[[None,'-x'],['-y',None]],[[None,None],['-x',None]],[[None,None,'-x'],[None,'+y',None]],[['+x',None,None],[None,None,'-y']],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
SORBATE_ATTACH_ATOM_OFFSET_FL_S=[[[None,None],[None,'+y']],[[None,'+x'],[None,None]],[[None,'+x'],['-y',None]],[[None,None],[None,'+x']],[[None,None,'+x'],['+y',None,None]],[['-x',None,None],[None,'-y',None]],[[],[]],[[],[]]]+SORBATE_ATTACH_ATOM_OFFSET_EXTRA
SORBATE_ATTACH_ATOM_OFFSET_FL=pick_full_layer(LFL=SORBATE_ATTACH_ATOM_OFFSET_FL_L,SFL=SORBATE_ATTACH_ATOM_OFFSET_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
SORBATE_ATTACH_ATOM_OFFSET_HL=pick_half_layer(LHL=SORBATE_ATTACH_ATOM_OFFSET_HL_L,SHL=SORBATE_ATTACH_ATOM_OFFSET_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
SORBATE_ATTACH_ATOM_OFFSET_SEPERATED=[deep_pick(SORBATE_ATTACH_ATOM_OFFSET_HL_L+each_FL) for each_FL in SORBATE_ATTACH_ATOM_OFFSET_FL]
SORBATE_ATTACH_ATOM_OFFSET_SEPERATED_HL=[deep_pick(each_HL+SORBATE_ATTACH_ATOM_OFFSET_FL_L) for each_HL in SORBATE_ATTACH_ATOM_OFFSET_HL]
SORBATE_ATTACH_ATOM_OFFSET=[SORBATE_ATTACH_ATOM_OFFSET_SEPERATED_HL[i][i] for i in range(N_HL)]+[SORBATE_ATTACH_ATOM_OFFSET_SEPERATED[i][N_HL+i] for i in range(N_FL)]

ANCHOR_REFERENCE_EXTRA=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]
ANCHOR_REFERENCE_HL_L=[[None,None],['Fe1_4_0','Fe1_6_0'],['Fe1_4_0','Fe1_6_0'],['Fe1_4_0','Fe1_6_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
ANCHOR_REFERENCE_HL_S=[[None,None],['Fe1_10_0','Fe1_12_0'],['Fe1_10_0','Fe1_12_0'],['Fe1_10_0','Fe1_12_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA
ANCHOR_REFERENCE_FL_L=[[None,None],['Fe1_2_0','Fe1_3_0'],['Fe1_2_0','Fe1_3_0'],['Fe1_2_0','Fe1_3_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
ANCHOR_REFERENCE_FL_S=[[None,None],['Fe1_8_0','Fe1_9_0'],['Fe1_8_0','Fe1_9_0'],['Fe1_8_0','Fe1_9_0'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_EXTRA#ref point for anchors
ANCHOR_REFERENCE_FL=pick_full_layer(LFL=ANCHOR_REFERENCE_FL_L,SFL=ANCHOR_REFERENCE_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
ANCHOR_REFERENCE_HL=pick_half_layer(LHL=ANCHOR_REFERENCE_HL_L,SHL=ANCHOR_REFERENCE_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
ANCHOR_REFERENCE_SEPERATED=[deep_pick(ANCHOR_REFERENCE_HL_L+each_FL) for each_FL in ANCHOR_REFERENCE_FL]
ANCHOR_REFERENCE_SEPERATED_HL=[deep_pick(each_HL+ANCHOR_REFERENCE_FL_L) for each_HL in ANCHOR_REFERENCE_HL]
ANCHOR_REFERENCE=[ANCHOR_REFERENCE_SEPERATED_HL[i][i] for i in range(N_HL)]+[ANCHOR_REFERENCE_SEPERATED[i][N_HL+i] for i in range(N_FL)]

ANCHOR_REFERENCE_OFFSET_EXTRA=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]
ANCHOR_REFERENCE_OFFSET_HL_L=[[None,None],['-y','+x'],[None,'+x'],[None,'+x'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
ANCHOR_REFERENCE_OFFSET_HL_S=[[None,None],['-y',None],[None,None],[None,'+x'],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
ANCHOR_REFERENCE_OFFSET_FL_L=[[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
ANCHOR_REFERENCE_OFFSET_FL_S=[[None,None],['+x',None],['+x',None],['+x',None],[None,None],[None,None],[None,None],[None,None]]+ANCHOR_REFERENCE_OFFSET_EXTRA
ANCHOR_REFERENCE_OFFSET_FL=pick_full_layer(LFL=ANCHOR_REFERENCE_OFFSET_FL_L,SFL=ANCHOR_REFERENCE_OFFSET_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
ANCHOR_REFERENCE_OFFSET_HL=pick_half_layer(LHL=ANCHOR_REFERENCE_OFFSET_HL_L,SHL=ANCHOR_REFERENCE_OFFSET_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
ANCHOR_REFERENCE_OFFSET_SEPERATED=[deep_pick(ANCHOR_REFERENCE_OFFSET_HL_L+each_FL) for each_FL in ANCHOR_REFERENCE_OFFSET_FL]
ANCHOR_REFERENCE_OFFSET_SEPERATED_HL=[deep_pick(each_HL+ANCHOR_REFERENCE_OFFSET_FL_L) for each_HL in ANCHOR_REFERENCE_OFFSET_HL]
ANCHOR_REFERENCE_OFFSET=[ANCHOR_REFERENCE_OFFSET_SEPERATED_HL[i][i] for i in range(N_HL)]+[ANCHOR_REFERENCE_OFFSET_SEPERATED[i][N_HL+i] for i in range(N_FL)]

#if consider hydrogen bonds#
#Arbitrary number of distal oxygens(6 here) will be helpful and handy if you want to consider the distal oxygen for bond valence constrain in a random mode, sine you wont need extra edition for that.
#It wont hurt even if the distal oxygen in the list doesn't actually exist for your model. Same for the potential hydrogen acceptor below
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA=[['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*7
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_EXTRA#Will be considered only when COVALENT_HYDROGEN_RANDOM=True
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL=pick_full_layer(LFL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L,SFL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL=pick_half_layer(LHL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L,SHL=POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED=[pick(POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL_L+each_FL) for each_FL in POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL]
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL=[pick(each_HL+POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_FL_L) for each_HL in POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_HL]
POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR=[POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]

COVALENT_HYDROGEN_ACCEPTOR_EXTRA=[[None]]*7
COVALENT_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
COVALENT_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA
COVALENT_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
COVALENT_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']]*8+COVALENT_HYDROGEN_ACCEPTOR_EXTRA#will be considered only when COVALENT_HYDROGEN_RANDOM=False
COVALENT_HYDROGEN_ACCEPTOR_FL=pick_full_layer(LFL=COVALENT_HYDROGEN_ACCEPTOR_FL_L,SFL=COVALENT_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
COVALENT_HYDROGEN_ACCEPTOR_HL=pick_half_layer(LHL=COVALENT_HYDROGEN_ACCEPTOR_HL_L,SHL=COVALENT_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
COVALENT_HYDROGEN_ACCEPTOR_SEPERATED=[pick(COVALENT_HYDROGEN_ACCEPTOR_HL_L+each_FL) for each_FL in COVALENT_HYDROGEN_ACCEPTOR_FL]
COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL=[pick(each_HL+COVALENT_HYDROGEN_ACCEPTOR_FL_L) for each_HL in COVALENT_HYDROGEN_ACCEPTOR_HL]
COVALENT_HYDROGEN_ACCEPTOR=[COVALENT_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[COVALENT_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]

COVALENT_HYDROGEN_NUMBER_EXTRA=[[None]]*7
COVALENT_HYDROGEN_NUMBER_HL=[[1,1,1,1],[2,1,0,1],[2,1,1,0],[2,1,0,1],[1,1,1,0],[2,1,0,0],[2,2,1,1],[2,2,1,1]]+COVALENT_HYDROGEN_NUMBER_EXTRA
COVALENT_HYDROGEN_NUMBER_FL=[[1,1,1,1],[2,1,1,0],[2,1,0,1],[2,1,1,0],[1,1,0,1],[2,1,0,0],[2,2,1,1],[2,2,1,1]]+COVALENT_HYDROGEN_NUMBER_EXTRA
COVALENT_HYDROGEN_NUMBER=pick(COVALENT_HYDROGEN_NUMBER_HL+COVALENT_HYDROGEN_NUMBER_FL)

POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA=[['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*7
POTENTIAL_HYDROGEN_ACCEPTOR_HL_L=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
POTENTIAL_HYDROGEN_ACCEPTOR_HL_S=[['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA
POTENTIAL_HYDROGEN_ACCEPTOR_FL_L=[['O1_11_t','O1_12_t','O1_1_0','O1_2_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
POTENTIAL_HYDROGEN_ACCEPTOR_FL_S=[['O1_5_0','O1_6_0','O1_7_0','O1_8_0']+['HO'+str(i+1)+'_'+sorbate+str(j+1) for i in range(6) for j in range(6) for sorbate in SORBATE_EL_LIST]]*8+POTENTIAL_HYDROGEN_ACCEPTOR_EXTRA#they can accept one hydrogen bond or not
POTENTIAL_HYDROGEN_ACCEPTOR_FL=pick_full_layer(LFL=POTENTIAL_HYDROGEN_ACCEPTOR_FL_L,SFL=POTENTIAL_HYDROGEN_ACCEPTOR_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
POTENTIAL_HYDROGEN_ACCEPTOR_HL=pick_half_layer(LHL=POTENTIAL_HYDROGEN_ACCEPTOR_HL_L,SHL=POTENTIAL_HYDROGEN_ACCEPTOR_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED=[pick(POTENTIAL_HYDROGEN_ACCEPTOR_HL_L+each_FL) for each_FL in POTENTIAL_HYDROGEN_ACCEPTOR_FL]
POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED_HL=[pick(each_HL+POTENTIAL_HYDROGEN_ACCEPTOR_FL_L) for each_HL in POTENTIAL_HYDROGEN_ACCEPTOR_HL]
POTENTIAL_HYDROGEN_ACCEPTOR=[POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED_HL[i][i] for i in range(N_HL)]+[POTENTIAL_HYDROGEN_ACCEPTOR_SEPERATED[i][N_HL+i] for i in range(N_FL)]

##pars for interfacial waters##
WATER_NUMBER=None
REF_POINTS=None
WATER_PAIR=True#add water pair each time if True, otherwise only add single water each time (only needed par is V_SHIFT)
if not water_pars['use_default']:
    WATER_NUMBER=water_pars['number']
    REF_POINTS=water_pars['ref_point']
else:
    WATER_NUMBER=pick([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    REF_POINTS_HL_L=[[['O1_1_0','O1_2_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
    REF_POINTS_HL_S=[[['O1_7_0','O1_8_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
    REF_POINTS_FL_L=[[['O1_11_t','O1_12_t']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
    REF_POINTS_FL_S=[[['O1_5_0','O1_6_0']]]*15#each item inside is a list of one or couple items, and each water set has its own ref point
    REF_POINTS_FL=pick_full_layer(LFL=REF_POINTS_FL_L,SFL=REF_POINTS_FL_S,pick_index=FULL_LAYER_PICK_INDEX)
    REF_POINTS_HL=pick_half_layer(LHL=REF_POINTS_HL_L,SHL=REF_POINTS_HL_S,pick_index=HALF_LAYER_PICK_INDEX)
    REF_POINTS_SEPERATED=[pick(REF_POINTS_HL_L+each_FL) for each_FL in REF_POINTS_FL]
    REF_POINTS_SEPERATED_HL=[pick(each_HL+REF_POINTS_FL_L) for each_HL in REF_POINTS_HL]
    REF_POINTS=[REF_POINTS_SEPERATED_HL[i][i] for i in range(N_HL)]+[REF_POINTS_SEPERATED[i][N_HL+i] for i in range(N_FL)]+[[None]]*7#each item inside is a list of one or couple items, and each water set has its own ref point

##chemically different domain type##
DOMAIN=pick([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
DOMAIN_NUMBER=len(DOMAIN)

SORBATE_NUMBER=pick_act(SORBATE_NUMBER_HL+SORBATE_NUMBER_HL_EXTRA+SORBATE_NUMBER_FL+SORBATE_NUMBER_FL_EXTRA)
O_NUMBER=pick_act(O_NUMBER_HL+O_NUMBER_HL_EXTRA+O_NUMBER_FL+O_NUMBER_FL_EXTRA)

#specify the METAL_BV based on the metal valence charge and the coordinated local structure
N_BOND=[]
METAL_BV=[]
for i in range(len(O_NUMBER)):
    temp_box=[]
    for j in range(len(O_NUMBER[i])):
        temp_box.append(O_NUMBER[i][j]+len(SORBATE_ATTACH_ATOM[i][j]))
    N_BOND.append(temp_box)

for i in range(len(N_BOND)):
    temp_box=[]
    for j in range(len(N_BOND[i])):
        if len(METAL_BV_EACH[i])!=len(N_BOND[i]):
            temp_box.append([METAL_BV_EACH[i][j/2]*N_BOND[i][j]-BV_OFFSET_SORBATE[i][j],METAL_BV_EACH[i][j/2]*N_BOND[i][j]])
        else:
            temp_box.append([METAL_BV_EACH[i][j]*N_BOND[i][j]-BV_OFFSET_SORBATE[i][j],METAL_BV_EACH[i][j]*N_BOND[i][j]])
    METAL_BV.append(temp_box)
#specify the searching range and penalty factor for surface atoms and sorbates
SEARCHING_PARS={'surface':[2.5,50],'sorbate':[[np.array(each)+SEARCH_RANGE_OFFSET for each in BOND_LENGTH_EACH],50]}#The value for each item [searching radius(A),scaling factor]

PROTONATION_DISTAL_OXYGEN=[[0,0]]*len(pickup_index)#Protonation of distal oxygens, any number in [0,1,2], where 1 means singly protonated, two means doubly protonated
SORBATE_LIST=domain_creator.create_sorbate_el_list2(SORBATE,SORBATE_NUMBER)
##want to make parameter table?##
TABLE=not running_mode
if TABLE:
    O_N=[]
    binding_mode=[]
    for i in O_NUMBER:
        temp=[]
        for j in range(0,len(i),2):
            temp.append(i[j])
        O_N.append(temp)
    for i in range(DOMAIN_NUMBER):
        temp_binding_mode=[]
        for j in range(0,len(SORBATE_ATTACH_ATOM[i]),2):
            if SORBATE_ATTACH_ATOM[i][j]==[]:
                temp_binding_mode.append('OS')
            else:
                if len(SORBATE_ATTACH_ATOM[i][j])==1:
                    temp_binding_mode.append('MD')
                elif len(SORBATE_ATTACH_ATOM[i][j])==2:
                    temp_binding_mode.append('BD')
                elif len(SORBATE_ATTACH_ATOM[i][j])==3:
                    temp_binding_mode.append('TD')
        binding_mode.append(temp_binding_mode)
    make_grid.make_structure(map(sum,SORBATE_NUMBER),O_N,WATER_NUMBER,DOMAIN,Metal=SORBATE,binding_mode=binding_mode,long_slab=full_layer_pick,long_slab_HL=half_layer_pick,local_structure=LOCAL_STRUCTURE,add_distal_wild=ADD_DISTAL_LIGAND_WILD,use_domains=TABLE_DOMAINS,N_raxr=NUMBER_SPECTRA,domain_raxr_el=RESONANT_EL_LIST,layered_water=layered_water_pars['yes_OR_no'],layered_sorbate=layered_sorbate_pars['yes_OR_no'],tab_path=os.path.join(output_file_path,'table.tab'))

#function to group the Fourier components (FC) from different domains in each RAXR spectra
#domain_index=[0,1] means setting the FC for domain2 (1+1) same as domain1 (0+1)
#domain_index=3 means setting the FC for domain2 and domain3 same as domain1, in this case the number indicate the number of total domains
def set_RAXR(domain_index=[],number_spectra=NUMBER_SPECTRA):
    domains=None
    if type(domain_index)!=type([]):
        domains=range(domain_index)
    else:
        domains=domain_index
    for i in range(number_spectra):
        for j in domains[1:]:
            eval('rgh_raxr'+'.setA_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getA_D'+str(domains[0]+1)+'_'+str(i+1)+'())')
            eval('rgh_raxr'+'.setP_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getP_D'+str(domains[0]+1)+'_'+str(i+1)+'())')

#freeze A and B in the process of model fitting
def set_RAXR_AB(number_spectra=NUMBER_SPECTRA):
    spectra=None
    if type(number_spectra)!=type([]):
        spectra=range(number_spectra)
    else:
        spectra=number_spectra
    for i in spectra:
        eval('rgh_raxr'+'.setA'+str(i+1)+'(1.)')
        eval('rgh_raxr'+'.setB'+str(i+1)+'(0.)')

#function to group outer-sphere pars from different domains (to be placed inside sim function)
def set_OS(domain_names=['domain5','domain4']):
    eval('rgh_'+domain_names[0]+'.setCt_offset_dx_OS(rgh_'+domain_names[1]+'.getCt_offset_dx_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dy_OS(rgh_'+domain_names[1]+'.getCt_offset_dy_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dz_OS(rgh_'+domain_names[1]+'.getCt_offset_dz_OS())')
    eval('rgh_'+domain_names[0]+'.setTop_angle_OS(rgh_'+domain_names[1]+'.getTop_angle_OS())')
    eval('rgh_'+domain_names[0]+'.setR0_OS(rgh_'+domain_names[1]+'.getR0_OS())')
    eval('rgh_'+domain_names[0]+'.setPhi_OS(rgh_'+domain_names[1]+'.getPhi_OS())')

#function to group bidentate pars from different domains (to be placed inside sim function)
def set_BD(domain_names=[2,1],sorbate_sets=1,distal_oxygen_number=1,sorbate='Pb'):
    for i in range(sorbate_sets):
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset2_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset2_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAngle_offset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getAngle_offset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setR_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getR_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setPhi_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getPhi_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setTop_angle_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getTop_angle_BD_'+str(i*2)+'())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        for j in range(distal_oxygen_number):
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')

#function to group water pairs togeter from different domains
#domain_names is list of index of domains counting from 0 and number sets is the number of water pair counting from 1
def set_water_pair(domain_names=[3,2],number_sets=2):
    for i in range(number_sets):
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setdy(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getdy())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setV_shift_W_'+str(i+1)+'(rgh_domain'+str(domain_names[1]+1)+'.getV_shift_W_'+str(i+1)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAlpha_W_'+str(i+1)+'(180-rgh_domain'+str(domain_names[1]+1)+'.getAlpha_W_'+str(i+1)+'())')

#function to group tridentate pars specifically for distal oxygens from different domains (to be placed inside sim function)
def set_TD(domain_names=['domain2','domain1']):
    eval('rgh_'+domain_names[0]+'.setTheta1_1_TD(rgh_'+domain_names[1]+'.getTheta1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_2_TD(rgh_'+domain_names[1]+'.getTheta1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_3_TD(rgh_'+domain_names[1]+'.getTheta1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_1_TD(rgh_'+domain_names[1]+'.getPhi1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_2_TD(rgh_'+domain_names[1]+'.getPhi1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_3_TD(rgh_'+domain_names[1]+'.getPhi1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_1_TD(rgh_'+domain_names[1]+'.getR1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_2_TD(rgh_'+domain_names[1]+'.getR1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_3_TD(rgh_'+domain_names[1]+'.getR1_3_TD())')

#function to group Hydrogen pars from the same domain (to be placed inside sim function)
def set_H(domain_name='domain1',tag=['W_1_2_1','W_1_1_1']):
    eval('rgh_'+domain_name+'.setPhi_H_'+tag[0]+'(180-rgh_'+domain_name+'.getPhi_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setR_H_'+tag[0]+'(rgh_'+domain_name+'.getR_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setTheta_H_'+tag[0]+'(rgh_'+domain_name+'.getTheta_H_'+tag[1]+'())')

#function to group distal oxygens based on adding in wild, N is the number of distal oxygens (to be placed inside sim function)
def set_distal_wild(domain_name=['domain2','domain1'],tag='BD',N=2):
    for i in range(N):
        eval('rgh_'+domain_name[0]+'.setPhi1_'+str(i)+'_'+tag+'(180-rgh_'+domain_name[1]+'.getPhi1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setR1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getR1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setTheta1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getTheta1_'+str(i)+'_'+tag+'())')

#function to run commands in sim funtion, command_list is a list of string representing the commands (mostly set get function)
def eval_in_sim(command_list=commands):
    if command_list!=[]:
        for command in command_list:
            eval(command)
    else:
        pass

##############################################set up atm ids###############################################

for i in range(DOMAIN_NUMBER):
    ##user defined variables
    vars()['rgh_domain'+str(int(i+1))]=UserVars()
    vars()['rgh_domain'+str(int(i+1))].new_var('wt', 1.)
    vars()['rgh_domain'+str(int(i+1))].new_var('wt_domainA', 0.5)

    ##sorbate list (HO is oxygen binded to pb and Os is water molecule)
    vars()['SORBATE_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids2(el=SORBATE[i],N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['SORBATE_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids2(el=SORBATE[i],N=SORBATE_NUMBER[i],tag='_D'+str(int(i+1))+'B')

    vars()['HO_list_domain'+str(int(i+1))+'a']=domain_creator.create_HO_ids3(anchor_els=SORBATE_LIST[i],O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['HO_list_domain'+str(int(i+1))+'b']=domain_creator.create_HO_ids3(anchor_els=SORBATE_LIST[i],O_N=O_NUMBER[i],tag='_D'+str(int(i+1))+'B')

    vars()['Os_list_domain'+str(int(i+1))+'a']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'A')
    vars()['Os_list_domain'+str(int(i+1))+'b']=domain_creator.create_sorbate_ids(el='Os',N=WATER_NUMBER[i],tag='_D'+str(int(i+1))+'B')

    vars()['sorbate_ids_domain'+str(int(i+1))+'a']=vars()['SORBATE_list_domain'+str(int(i+1))+'a']+vars()['HO_list_domain'+str(int(i+1))+'a']+vars()['Os_list_domain'+str(int(i+1))+'a']
    vars()['sorbate_ids_domain'+str(int(i+1))+'b']=vars()['SORBATE_list_domain'+str(int(i+1))+'b']+vars()['HO_list_domain'+str(int(i+1))+'b']+vars()['Os_list_domain'+str(int(i+1))+'b']
    vars()['sorbate_els_domain'+str(int(i+1))]=SORBATE_LIST[i]+['O']*(sum([np.sum(N_list) for N_list in O_NUMBER[i]])+WATER_NUMBER[i])

    ##set up group name container(discrete:single atom from each domain, sequence:double atoms at same layer from each domain)
    #atom ids for grouping(containerB must be the associated chemically equivalent atoms)
    equivalent_atm_list_A_L_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0"]
    equivalent_atm_list_A_S_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1"]

    equivalent_atm_list_A_S_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1"]
    equivalent_atm_list_A_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0"]

    equivalent_atm_list_B_L_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1"]
    equivalent_atm_list_B_S_1=["O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1","O1_7_1","O1_8_1","Fe1_8_1","Fe1_9_1","O1_9_1","O1_10_1","Fe1_10_1","Fe1_12_1","O1_11_1","O1_12_1"]
    equivalent_atm_list_B_S_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1","O1_5_1","O1_6_1","O1_7_1","O1_8_1","Fe1_8_1","Fe1_9_1","O1_9_1","O1_10_1","Fe1_10_1","Fe1_12_1"]
    equivalent_atm_list_B_L_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0","O1_1_1","O1_2_1","Fe1_2_1","Fe1_3_1","O1_3_1","O1_4_1","Fe1_4_1","Fe1_6_1"]

    atm_sequence_gp_names_L_1=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    atm_sequence_gp_names_S_1=['O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12']
    atm_sequence_gp_names_S_2=['O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12']
    atm_sequence_gp_names_L_2=['O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6']

    atm_list_A_L_1=['O1_1_0','O1_2_0','O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0']
    atm_list_A_S_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_10_0','Fe1_12_0','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1',]
    atm_list_A_S_2=['O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0','Fe1_4_1','Fe1_6_1']
    atm_list_A_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0",'O1_3_0','O1_4_0','O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','Fe1_2_0','Fe1_3_0','Fe1_4_0','Fe1_6_0','Fe1_8_0','Fe1_9_0']

    atm_list_B_L_1=['O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_11_0','O1_12_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_10_0','Fe1_12_0','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1']
    atm_list_B_S_1=['O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','O1_11_1','O1_12_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1','Fe1_10_1','Fe1_12_1']
    atm_list_B_S_2=["O1_11_0","O1_12_0","O1_1_1","O1_2_1",'O1_3_1','O1_4_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','O1_9_1','O1_10_1','Fe1_2_1','Fe1_3_1','Fe1_4_1','Fe1_6_1','Fe1_8_1','Fe1_9_1']
    atm_list_B_L_2=['O1_5_0','O1_6_0','O1_7_0','O1_8_0','O1_9_0','O1_10_0','O1_1_1','O1_2_1','O1_3_1','O1_4_1','O1_5_1','O1_6_1','Fe1_8_0','Fe1_9_0','Fe1_10_0','Fe1_12_0','Fe1_4_1','Fe1_6_1']

    if int(DOMAIN[i])==1:
        tag=None
        if half_layer_pick[i]==2:
            tag='S'
        elif half_layer_pick[i]==3:
            tag='L'
        vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['discrete_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a'])+\
                                                     map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
        vars()['sequence_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['atm_list_'+str(int(i+1))+'A']=map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['atm_list_'+str(int(i+1))+'B']=map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])
    elif int(DOMAIN[i])==2:
        tag=None
        if full_layer_pick[i]==0:
            tag='S'
        elif full_layer_pick[i]==1:
            tag='L'
        vars()['ids_domain'+str(int(i+1))+'A']=vars()['sorbate_ids_domain'+str(int(i+1))+'a']+map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['ids_domain'+str(int(i+1))+'B']=vars()['sorbate_ids_domain'+str(int(i+1))+'b']+map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['discrete_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x.rsplit('_')[0]+'_D'+str(int(i+1)),vars()['sorbate_ids_domain'+str(int(i+1))+'a'])+\
                                                     map(lambda x:'gp_'+x[0].rsplit('_')[0][:-1]+x[0].rsplit('_')[1]+x[1].rsplit('_')[0][:-1]+x[1].rsplit('_')[1]+'_D'+str(int(i+1)),zip(vars()['equivalent_atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))],vars()['equivalent_atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))]))
        vars()['sequence_gp_names_domain'+str(int(i+1))]=map(lambda x:'gp_'+x+'_D'+str(int(i+1)),vars()['atm_sequence_gp_names_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['atm_list_'+str(int(i+1))+'A']=map(lambda x:x+'_D'+str(int(i+1))+'A',vars()['atm_list_A_'+tag+'_'+str(int(DOMAIN[i]))])
        vars()['atm_list_'+str(int(i+1))+'B']=map(lambda x:x+'_D'+str(int(i+1))+'B',vars()['atm_list_B_'+tag+'_'+str(int(DOMAIN[i]))])

##id list according to the order in the reference domain (used to set up ref domain)
ref_id_list_L_1=["O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_S_1=["O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_S_2=["O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
ref_id_list_L_2=["O1_11_t","O1_12_t","O1_1_0","O1_2_0","Fe1_2_0","Fe1_3_0","O1_3_0","O1_4_0","Fe1_4_0","Fe1_6_0","O1_5_0","O1_6_0","O1_7_0","O1_8_0","Fe1_8_0","Fe1_9_0","O1_9_0","O1_10_0","Fe1_10_0","Fe1_12_0","O1_11_0","O1_12_0",\
'O1_1_1','O1_2_1','Fe1_2_1','Fe1_3_1','O1_3_1','O1_4_1','Fe1_4_1','Fe1_6_1','O1_5_1','O1_6_1','O1_7_1','O1_8_1','Fe1_8_1','Fe1_9_1','O1_9_1','O1_10_1','Fe1_10_1','Fe1_12_1','O1_11_1','O1_12_1']
###############################################setting slabs##################################################################
unitcell = model.UnitCell(5.038, 5.434, 7.3707, 90, 90, 90)
inst = model.Instrument(wavel = .833, alpha = 2.0)
bulk = model.Slab(T_factor='B')
ref_S_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_L_domain1 =  model.Slab(c = 1.0,T_factor='B')
ref_S_domain2 =  model.Slab(c = 1.0,T_factor='B')
ref_L_domain2 =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)
scales=['scale_CTR']
for scale in scales:
    rgh.new_var(scale,1.)

################################################build up ref domains############################################
#add atoms for bulk and two ref domains (ref_domain1<half layer> and ref_domain2<full layer>)
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted

#only two possible path(one for runing in pacman, the other in local laptop)
#batch_path_head='//'.join(batch_path.module_path_locator().rsplit('/'))+'//'
batch_path_head=batch_path.module_path_locator()
#try:
#    domain_creator.add_atom_in_slab(bulk,batch_path_head+'bulk.str')
#except:
#    batch_path_head='/u1/uaf/cqiu/batchfile/'
domain_creator.add_atom_in_slab(bulk,os.path.join(batch_path_head,'bulk.str'))
domain_creator.add_atom_in_slab(ref_L_domain1,os.path.join(batch_path_head,'half_layer2.str'))
domain_creator.add_atom_in_slab(ref_S_domain1,os.path.join(batch_path_head,'half_layer3.str'))
domain_creator.add_atom_in_slab(ref_L_domain2,os.path.join(batch_path_head,'full_layer2.str'))
domain_creator.add_atom_in_slab(ref_S_domain2,os.path.join(batch_path_head,'full_layer3.str'))

##set up Fourier pars if there are RAXR datasets
#Fourier component looks like A_Dn0_n1, where n0, n1 are used to specify the index for domain, and spectra, respectively
#Each spectra will have its own set of A and P list, and each domain has its own set of P and A list
rgh_raxs=None
if NUMBER_SPECTRA!=0:
    F1F2=np.loadtxt(os.path.join(batch_path_head,F1F2_FILE))
    rgh_raxr=UserVars()
    for i in range(NUMBER_SPECTRA):
        rgh_raxr.new_var('a'+str(i+1),0.0)
        rgh_raxr.new_var('b'+str(i+1),0.0)
        for j in range(len(RESONANT_EL_LIST)):
            if RESONANT_EL_LIST[j]!=0:
                rgh_raxr.new_var('A_D'+str(j+1)+'_'+str(i+1),2.0)
                rgh_raxr.new_var('P_D'+str(j+1)+'_'+str(i+1),0.0)
###################create domain classes and initiate the chemical equivalent domains####################
#when change or create a new domain, make sure the terminated_layer (start from 0)set right

##setup domains
for i in range(DOMAIN_NUMBER):
    vars()['HB_MATCH_'+str(i+1)]={}
    HB_MATCH=vars()['HB_MATCH_'+str(i+1)]
    if int(DOMAIN[i])==1:
        if half_layer_pick[i]==2:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_S_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_S_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
        elif half_layer_pick[i]==3:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_L_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_L_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    elif int(DOMAIN[i])==2:
        if full_layer_pick[i]==0:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_S_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_S_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
        elif full_layer_pick[i]==1:
            vars()['domain_class_'+str(int(i+1))]=domain_creator.domain_creator(ref_domain=vars()['ref_L_domain'+str(int(DOMAIN[i]))],id_list=vars()['ref_id_list_L_'+str(int(DOMAIN[i]))],terminated_layer=0,domain_tag='_D'+str(int(i+1)),new_var_module=vars()['rgh_domain'+str(int(i+1))])
    vars()['domain'+str(int(i+1))+'A']=vars()['domain_class_'+str(int(i+1))].domain_A
    vars()['domain'+str(int(i+1))+'B']=vars()['domain_class_'+str(int(i+1))].domain_B
    vars(vars()['domain_class_'+str(int(i+1))])['domainA']=vars()['domain'+str(int(i+1))+'A']
    vars(vars()['domain_class_'+str(int(i+1))])['domainB']=vars()['domain'+str(int(i+1))+'B']

    #Adding hydrogen to pre-defined hydrogen acceptor of surface oxygens
    if INCLUDE_HYDROGEN:
        for i_H in range(len(COVALENT_HYDROGEN_ACCEPTOR[i])):
            for j_H in range(COVALENT_HYDROGEN_NUMBER[i][i_H]):
                vars()['rgh_domain'+str(int(i+1))].new_var('r_H_'+str(i_H+1)+'_'+str(j_H+1), 1.)
                vars()['rgh_domain'+str(int(i+1))].new_var('phi_H_'+str(i_H+1)+'_'+str(j_H+1), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('theta_H_'+str(i_H+1)+'_'+str(j_H+1), 0.)
                coor=vars()['domain_class_'+str(int(i+1))].adding_hydrogen(domain=vars()['domain'+str(int(i+1))+'A'],N_of_HB=j_H,ref_id=COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A',r=getattr(vars()['rgh_domain'+str(int(i+1))],'r_H_'+str(i_H+1)+'_'+str(j_H+1)),theta=getattr(vars()['rgh_domain'+str(int(i+1))],'theta_H_'+str(i_H+1)+'_'+str(j_H+1)),phi=getattr(vars()['rgh_domain'+str(int(i+1))],'phi_H_'+str(i_H+1)+'_'+str(j_H+1)))
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_H+1)+'_'+COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'B'],els=['H'])
                if COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A' in HB_MATCH.keys():
                    HB_MATCH[COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A'].append('HB'+str(j_H+1)+'_'+COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A')
                else:
                    HB_MATCH[COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A']=['HB'+str(j_H+1)+'_'+COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A']
                HB_MATCH['HB'+str(j_H+1)+'_'+COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A']=[COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A']
    #Adding sorbates to domainA and domainB
    for j in range(sum(SORBATE_NUMBER[i])):
        SORBATE_coors_a=[]
        O_coors_a=[]
        if len(SORBATE_ATTACH_ATOM[i][j])==1:#monodentate case
            if j%2==0:
                vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_MD_'+str(j), 71.)
                vars()['rgh_domain'+str(int(i+1))].new_var('phi_MD_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('r_MD_'+str(j), 2.)
            if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(2)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(2)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(2)]
            if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(3)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(3)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(3)]
            if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='octahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_MD_'+str(j), 2.27) for KK in range(5)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(5)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_MD_'+str(j), 0) for KK in range(5)]

            ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j][0]
            if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
                ids=SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
            else:
                ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']

            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
            O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
            sorbate_coors=[]
            if USE_COORS[i][j]:
                sorbate_coors=COORS[(i,j)]['sorbate'][0]+COORS[i]['oxygen'][0]
            else:
                if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=70,phi=0,r=2,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,O_id=O_id,mirror=MIRROR[i],sorbate_el=SORBATE_LIST[i][j])
                elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],phi=0,r=2,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
                elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_tetrahedral_monodentate(domain=vars()['domain'+str(int(i+1))+'A'],phi=0.,r=2.25,attach_atm_id=ids,offset=offset,sorbate_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
            SORBATE_coors_a.append(sorbate_coors[0])
            if O_id!=[]:
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
            #now put on sorbate on the symmetrically related domain
            sorbate_ids=[SORBATE_id_B]+O_id_B
            sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
            if USE_COORS[i][j]:
                SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                O_id_A=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id_A in HO_id]
                sorbate_ids_A=[SORBATE_id_A]+O_id_A
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
        elif len(SORBATE_ATTACH_ATOM[i][j])==2:#bidentate case
            if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset2_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('angle_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_BD_'+str(j), 70.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('r_BD_'+str(j), 2.27)
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(1)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(1)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(1)]
                else:
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset2_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('angle_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_BD_'+str(j), 70.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('r_BD_'+str(j), 2.27)
            elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='octahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(4)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(4)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(4)]
                else:
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)
            elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('anchor_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_offset_BD_'+str(j), 0.)
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_BD_'+str(j), 2.27) for KK in range(2)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(2)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_BD_'+str(j), 0) for KK in range(2)]
                else:
                    vars()['rgh_domain'+str(int(i+1))].new_var('anchor_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('offset2_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('angle_offset_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('angle_offset2_BD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('phi_BD_'+str(j), 0.)

            ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
            anchor,anchor_offset=None,None
            if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
                ids=[SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
                anchor=BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
                anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
            else:
                ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A']
                if ANCHOR_REFERENCE[i][j]!=None:
                    anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
                    anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]

            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
            O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
            sorbate_coors=[]
            if USE_COORS[i][j]:
                sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
            else:
                if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion_B(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=70,phi=0,edge_offset=[0,0],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=MIRROR[i][j/2])
                elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=vars()['domain'+str(int(i+1))+'A'],phi=90,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
                elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_tetrahedral(domain=vars()['domain'+str(int(i+1))+'A'],phi=0,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
            SORBATE_coors_a.append(sorbate_coors[0])
            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
            #now put on sorbate on the symmetrically related domain
            sorbate_ids=[SORBATE_id_B]+O_id_B
            sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
            if USE_COORS[i][j]:
                SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                O_id_A=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id_A in HO_id]
                sorbate_ids_A=[SORBATE_id_A]+O_id_A
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
        elif len(SORBATE_ATTACH_ATOM[i][j])==3:#tridentate case (no oxygen sorbate here considering it is a trigonal pyramid structure)
            if j%2==0 and LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_TD_'+str(j), 70.)
            elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='octahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr1_oct_TD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr2_oct_TD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr3_oct_TD_'+str(j), 0.)
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_TD_'+str(j), 2.27) for KK in range(3)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(3)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(3)]
                else:
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr1_oct_TD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr2_oct_TD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr3_oct_TD_'+str(j), 0.)
            elif j%2==0 and LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                if ADD_DISTAL_LIGAND_WILD[i][j]:
                    [vars()['rgh_domain'+str(int(i+1))].new_var('r1_'+str(KK+1)+'_TD_'+str(j), 2.27) for KK in range(1)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('theta1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(1)]
                    [vars()['rgh_domain'+str(int(i+1))].new_var('phi1_'+str(KK+1)+'_TD_'+str(j), 0.) for KK in range(1)]
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr_tetrahedral_TD_'+str(j), 0.)
                else:
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr_tetrahedral_TD_'+str(j), 0.)
                    vars()['rgh_domain'+str(int(i+1))].new_var('dr_bc_tetrahedral_TD_'+str(j), 0.)

            ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
            if "HO" in SORBATE_ATTACH_ATOM[i][j][0]:#a sign for ternary complex structure forming
                ids=[SORBATE_ATTACH_ATOM[i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][2]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
            else:
                ids=[SORBATE_ATTACH_ATOM[i][j][0]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][1]+'_D'+str(int(i+1))+'A',SORBATE_ATTACH_ATOM[i][j][2]+'_D'+str(int(i+1))+'A']
            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
            O_index,O_id,sorbate_coors,O_id_B,HO_set_ids,SORBATE_id_B,sorbate_ids,SORBATE_coors_a=[],[],[],[],[],[],[],[]
            if LOCAL_STRUCTURE[i][j/2]=='octahedral':
                O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                if USE_COORS[i][j]:
                    sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
                else:
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_share_triple_octahedra(domain=vars()['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id)
                SORBATE_coors_a.append(sorbate_coors[0])
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                sorbate_ids=[SORBATE_id_B]+O_id_B
                sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                if USE_COORS[i][j]:
                    SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    O_id_A=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id_A in HO_id]
                    sorbate_ids_A=[SORBATE_id_A]+O_id_A
                    domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            elif LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                if USE_COORS[i][j]:
                    sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
                else:
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_pb_share_triple4(domain=vars()['domain'+str(int(i+1))+'A'],top_angle=70,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j])
                SORBATE_coors_a.append(sorbate_coors[0])
                SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                #now put on sorbate on the symmetrically related domain
                sorbate_ids=[SORBATE_id_B]
                sorbate_els=[SORBATE_LIST[i][j]]
                if USE_COORS[i][j]:
                    SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    sorbate_ids_A=[SORBATE_id_A]
                    domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a),ids=sorbate_ids_A,els=sorbate_els)
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                if USE_COORS[i][j]:
                    sorbate_coors=COORS[(i,j)]['sorbate']+COORS[(i,j)]['oxygen']
                else:
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].adding_sorbate_tridentate_tetrahedral(domain=vars()['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id)
                SORBATE_coors_a.append(sorbate_coors[0])
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                sorbate_ids=[SORBATE_id_B]+O_id_B
                sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                if USE_COORS[i][j]:
                    SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    O_id_A=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id_A in HO_id]
                    sorbate_ids_A=[SORBATE_id_A]+O_id_A
                    domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)

            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            #if SORBATE_LIST[i][j]=='Sb':
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)
        else:#add an outer-sphere case here
            if j%2==0:
                vars()['rgh_domain'+str(int(i+1))].new_var('phi_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('r0_OS_'+str(j), 2.26)
                vars()['rgh_domain'+str(int(i+1))].new_var('top_angle_OS_'+str(j), 70.)
                vars()['rgh_domain'+str(int(i+1))].new_var('ct_offset_dx_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('ct_offset_dy_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('ct_offset_dz_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('rot_x_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('rot_y_OS_'+str(j), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('rot_z_OS_'+str(j), 0.)

            SORBATE_id=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
            O_id=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
            consider_distal=False
            if O_id!=[]:
                consider_distal=True
            sorbate_coors=[]
            if USE_COORS[i][j]:
                sorbate_coors=COORS[(i,j)]['sorbate'][0]+COORS[(i,j)]['oxygen'][0]
            else:
                if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].outer_sphere_complex_2(domain=vars()['domain'+str(int(i+1))+'A'],cent_point=[0.75,0.+j*0.5,2.1],r_Pb_O=2.28,O_Pb_O_ang=70,phi=j*np.pi-0,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
                elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].outer_sphere_complex_oct(domain=vars()['domain'+str(int(i+1))+'A'],cent_point=[0.75,0.+j*0.5,2.1],r0=1.62,phi=j*np.pi-0,Sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
                elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                    sorbate_coors=vars()['domain_class_'+str(int(i+1))].outer_sphere_tetrahedral2(domain=vars()['domain'+str(int(i+1))+'A'],cent_point=[0.75,0.+j*0.5,2.1],r_sorbate_O=1.65,phi=j*np.pi-0,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
            SORBATE_coors_a.append(sorbate_coors[0])
            if O_id!=[]:
                [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
            SORBATE_id_B=vars()['SORBATE_list_domain'+str(int(i+1))+'b'][j]
            O_id_B=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
            #now put on sorbate on the symmetrically related domain
            sorbate_ids=[SORBATE_id_B]+O_id_B
            sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
            if USE_COORS[i][j]:
                SORBATE_id_A=vars()['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                O_id_A=[HO_id for HO_id in vars()['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id_A in HO_id]
                sorbate_ids_A=[SORBATE_id_A]+O_id_A
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'A'],ref_coor=np.array(SORBATE_coors_a+O_coors_a),ids=sorbate_ids_A,els=sorbate_els)
            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
            #grouping sorbates (each set of Pb and HO, set the occupancy equivalent during fitting, looks like gp_sorbates_set1_D1)
            #also group the oxygen sorbate to set equivalent u during fitting (looks like gp_HO_set1_D1)
            sorbate_set_ids=[SORBATE_id]+O_id+[SORBATE_id_B]+O_id_B
            HO_set_ids=O_id+O_id_B
            N=len(sorbate_set_ids)/2
            M=len(O_id)
            vars()['gp_sorbates_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*N+[vars()['domain'+str(int(i+1))+'B']]*N,atom_ids=sorbate_set_ids)
            #if O_NUMBER[i][j]!=0:
            if M!=0:
                vars()['gp_HO_set'+str(j+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=HO_set_ids)

    if INCLUDE_HYDROGEN:
        for i_distal in range(len(vars()['HO_list_domain'+str(int(i+1))+'a'])):#add hydrogen to distal oxygens if any
            for j_distal in range(PROTONATION_DISTAL_OXYGEN[i][i_distal]):
                vars()['rgh_domain'+str(int(i+1))].new_var('r_H_D_'+str(i_distal+1)+'_'+str(j_distal+1), 1.)
                vars()['rgh_domain'+str(int(i+1))].new_var('phi_H_D_'+str(i_distal+1)+'_'+str(j_distal+1), 0.)
                vars()['rgh_domain'+str(int(i+1))].new_var('theta_H_D_'+str(i_distal+1)+'_'+str(j_distal+1), 0.)
                coor=vars()['domain_class_'+str(int(i+1))].adding_hydrogen(domain=vars()['domain'+str(int(i+1))+'A'],N_of_HB=j_distal,ref_id=vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal],r=getattr(vars()['rgh_domain'+str(int(i+1))],'r_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)),theta=getattr(vars()['rgh_domain'+str(int(i+1))],'theta_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)),phi=getattr(vars()['rgh_domain'+str(int(i+1))],'phi_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)))
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_distal+1)+'_'+vars()['HO_list_domain'+str(int(i+1))+'b'][i_distal]],els=['H'])
                if vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal] in HB_MATCH.keys():
                    HB_MATCH[vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal]].append('HB'+str(j_distal+1)+'_'+vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal])
                else:
                    HB_MATCH[vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal]]=['HB'+str(j_distal+1)+'_'+vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal]]
                HB_MATCH['HB'+str(j_distal+1)+'_'+vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal]]=[vars()['HO_list_domain'+str(int(i+1))+'a'][i_distal]]

    if layered_water_pars['yes_OR_no'][i]:
        vars()['rgh_domain'+str(int(i+1))].new_var('u0',0.4)
        vars()['rgh_domain'+str(int(i+1))].new_var('ubar',0.4)
        vars()['rgh_domain'+str(int(i+1))].new_var('first_layer_height',4.0)#relative height in A
        vars()['rgh_domain'+str(int(i+1))].new_var('d_w',1.9)#inter-layer water seperation in A
        vars()['rgh_domain'+str(int(i+1))].new_var('density_w',0.033)#number density in unit of # of waters per cubic A

    if layered_sorbate_pars['yes_OR_no'][i]:
        vars()['rgh_domain'+str(int(i+1))].new_var('u0_s',0.4)
        vars()['rgh_domain'+str(int(i+1))].new_var('ubar_s',0.4)
        vars()['rgh_domain'+str(int(i+1))].new_var('first_layer_height_s',4.0)#relative height in A
        vars()['rgh_domain'+str(int(i+1))].new_var('d_s',1.9)#inter-layer sorbate seperation in A
        vars()['rgh_domain'+str(int(i+1))].new_var('density_s',0.033)#number density in unit of # of sorbates per cubic A

    if WATER_NUMBER[i]!=0:#add water molecules if any
        if WATER_PAIR:
            for jj in range(WATER_NUMBER[i]/2):#note will add water pair (two oxygens) each time, and you can't add single water
                vars()['rgh_domain'+str(int(i+1))].new_var('alpha_W_'+str(jj+1),90.)
                #vars()['rgh_domain'+str(int(i+1))].new_var('R_W_'+str(jj+1),1)
                vars()['rgh_domain'+str(int(i+1))].new_var('v_shift_W_'+str(jj+1),1.)

                O_ids_a=vars()['Os_list_domain'+str(int(i+1))+'a'][jj*2:jj*2+2]
                O_ids_b=vars()['Os_list_domain'+str(int(i+1))+'b'][jj*2:jj*2+2]
                #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                H2O_coors_a=vars()['domain_class_'+str(int(i+1))].add_oxygen_pair2B(domain=vars()['domain'+str(int(i+1))+'A'],O_ids=O_ids_a,ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=1,r=2.717,alpha=90)
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
                #group water molecules at each layer (set equivalent the oc and u during fitting)
                M=len(O_ids_a)
                #group waters on a layer basis(every four, two from each domain)
                vars()['gp_waters_set'+str(jj+1)+'_D'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]*M+[vars()['domain'+str(int(i+1))+'B']]*M,atom_ids=O_ids_a+O_ids_b)
                #group each two waters on two symmetry domains together (to be used as constrain on inplane movements)
                #group names look like: gp_Os1_D1 which will group Os1_D1A and Os1_D1B together
                for O_id in O_ids_a:
                    index=O_ids_a.index(O_id)
                    gp_name='gp_'+O_id.rsplit('_')[0]+'_D'+str(int(i+1))
                    vars()[gp_name]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]+[vars()['domain'+str(int(i+1))+'B']],atom_ids=[O_ids_a[index],O_ids_b[index]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]])
                if INCLUDE_HYDROGEN:
                    #add two hydrogen for each oxygen
                    for i_water in [0,1]:#two waters considered here
                        for j_water in [0,1]:#doubly protonated for each water
                            vars()['rgh_domain'+str(int(i+1))].new_var('r_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 1.)
                            vars()['rgh_domain'+str(int(i+1))].new_var('phi_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 0.)
                            vars()['rgh_domain'+str(int(i+1))].new_var('theta_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 0.)
                            coor=vars()['domain_class_'+str(int(i+1))].adding_hydrogen(domain=vars()['domain'+str(int(i+1))+'A'],N_of_HB=j_water,ref_id=O_ids_a[i_water],r=getattr(vars()['rgh_domain'+str(int(i+1))],'r_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),theta=getattr(vars()['rgh_domain'+str(int(i+1))],'theta_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),phi=getattr(vars()['rgh_domain'+str(int(i+1))],'phi_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)))
                            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_water+1)+'_'+O_ids_b[i_water]],els=['H'])
                            if O_ids_a[i_water] in HB_MATCH.keys():
                                HB_MATCH[O_ids_a[i_water]].append('HB'+str(j_water+1)+'_'+O_ids_a[i_water])
                            else:
                                HB_MATCH[O_ids_a[i_water]]=['HB'+str(j_water+1)+'_'+O_ids_a[i_water]]
                            HB_MATCH['HB'+str(j_water+1)+'_'+O_ids_a[i_water]]=[O_ids_a[i_water]]

        else:
            for jj in range(WATER_NUMBER[i]):#note will add single water each time
                vars()['rgh_domain'+str(int(i+1))].new_var('v_shift_W_'+str(jj+1),1)

                O_ids_a=[vars()['Os_list_domain'+str(int(i+1))+'a'][jj]]
                O_ids_b=[vars()['Os_list_domain'+str(int(i+1))+'b'][jj]]
                #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                H2O_coors_a=vars()['domain_class_'+str(int(i+1))].add_single_oxygen(domain=vars()['domain'+str(int(i+1))+'A'],O_id=O_ids_a[0],ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=1)
                domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O'])
                #group each two waters on two symmetry domains together (to be used as constrain on inplane movements)
                #group names look like: gp_Os1_D1 which will group Os1_D1A and Os1_D1B together
                gp_name='gp_'+O_ids_a[0].rsplit('_')[0]+'_D'+str(int(i+1))
                vars()[gp_name]=vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A']]+[vars()['domain'+str(int(i+1))+'B']],atom_ids=[O_ids_a[0],O_ids_b[0]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]])

                if INCLUDE_HYDROGEN:
                    #add two hydrogen to each oxygen of one water
                    for i_water in [0]:#single waters considered
                        for j_water in [0,1]:#doubly protonated for each water
                            vars()['rgh_domain'+str(int(i+1))].new_var('r_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 1.)
                            vars()['rgh_domain'+str(int(i+1))].new_var('phi_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 0.)
                            vars()['rgh_domain'+str(int(i+1))].new_var('theta_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1), 0.)
                            coor=vars()['domain_class_'+str(int(i+1))].adding_hydrogen(domain=vars()['domain'+str(int(i+1))+'A'],N_of_HB=j_water,ref_id=O_ids_a[i_water],r=getattr(vars()['rgh_domain'+str(int(i+1))],'r_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),theta=getattr(vars()['rgh_domain'+str(int(i+1))],'theta_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),phi=getattr(vars()['rgh_domain'+str(int(i+1))],'phi_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)))
                            domain_creator.add_atom(domain=vars()['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_water+1)+'_'+O_ids_b[i_water]],els=['H'])
                            if O_ids_a[i_water] in HB_MATCH.keys():
                                HB_MATCH[O_ids_a[i_water]].append('HB'+str(j_water+1)+'_'+O_ids_a[i_water])
                            else:
                                HB_MATCH[O_ids_a[i_water]]=['HB'+str(j_water+1)+'_'+O_ids_a[i_water]]
                            HB_MATCH['HB'+str(j_water+1)+'_'+O_ids_a[i_water]]=[O_ids_a[i_water]]

######################################do grouping###############################################
for i in range(DOMAIN_NUMBER):
    #note the grouping here is on a layer basis, ie atoms of same layer are groupped together (4 atms grouped together in sequence grouping)
    #you may group in symmetry, then atoms of same layer are not independent. Know here the symmetry (equal opposite) is impressively defined in the function
    if DOMAIN[i]==1:
        if half_layer_pick[i]==3:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_1_0_D'+str(int(i+1))+'A','O1_7_0_D'+str(int(i+1))+'B']],layers_N=10)
        elif half_layer_pick[i]==2:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_7_0_D'+str(int(i+1))+'A','O1_1_1_D'+str(int(i+1))+'B']],layers_N=10)
    elif DOMAIN[i]==2:
        if full_layer_pick[i]==1:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_11_t_D'+str(int(i+1))+'A','O1_5_0_D'+str(int(i+1))+'B']],layers_N=10)
        elif full_layer_pick[i]==0:
            vars()['atm_gp_list_domain'+str(int(i+1))]=vars()['domain_class_'+str(int(i+1))].grouping_sequence_layer_new2(domain=[[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]], first_atom_id=[['O1_5_0_D'+str(int(i+1))+'A','O1_11_0_D'+str(int(i+1))+'B']],layers_N=10)

    #assign name to each group
    for j in range(len(vars()['sequence_gp_names_domain'+str(int(i+1))])):vars()[vars()['sequence_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_list_domain'+str(int(i+1))][j]

    #you may also only want to group each chemically equivalent atom from two domains (the use_sym is set to true here)
    vars()['atm_gp_discrete_list_domain'+str(int(i+1))]=[]
    for j in range(len(vars()['ids_domain'+str(int(i+1))+'A'])):
        vars()['atm_gp_discrete_list_domain'+str(int(i+1))].append(vars()['domain_class_'+str(int(i+1))].grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],\
                                                                   atom_ids=[vars()['ids_domain'+str(int(i+1))+'A'][j],vars()['ids_domain'+str(int(i+1))+'B'][j]],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]]))
    for j in range(len(vars()['discrete_gp_names_domain'+str(int(i+1))])):vars()[vars()['discrete_gp_names_domain'+str(int(i+1))][j]]=vars()['atm_gp_discrete_list_domain'+str(int(i+1))][j]

    try:#group sorbates in pairs
        for N in range(0,sum(SORBATE_NUMBER[i]),2):
            vars()['gp_'+SORBATE_LIST[i][N]+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']]*2,atom_ids=[SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'B',SORBATE_LIST[i][N]+str(N+2)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+2)+'_D'+str(i+1)+'B'],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.],[1.,0.,0.,0.,1.,0.,0.,0.,1.]])
            if vars()['HO_list_domain'+str(i+1)+'a']!=[]:
                HO_A1=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_B1=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_A2=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+2)+'_') in each]
                HO_B2=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+2)+'_') in each]
                for NN in range(len(HO_A1)):
                    vars()['gp_HO'+str(NN+1)+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B'],vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B']],\
                          atom_ids=[HO_A1[NN],HO_B1[NN],HO_A2[NN],HO_B2[NN]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1],[1,0,0,0,1,0,0,0,1]])
    except:#consider single site for each domain otherwise
        for N in range(0,sum(SORBATE_NUMBER[i]),1):
            vars()['gp_'+SORBATE_LIST[i][N]+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(int(i+1))+'A'],vars()['domain'+str(int(i+1))+'B']],atom_ids=[SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'A',SORBATE_LIST[i][N]+str(N+1)+'_D'+str(i+1)+'B'],sym_array=[[1.,0.,0.,0.,1.,0.,0.,0.,1.],[-1.,0.,0.,0.,1.,0.,0.,0.,1.]])
            if vars()['HO_list_domain'+str(i+1)+'a']!=[]:
                HO_A1=[each for each in vars()['HO_list_domain'+str(i+1)+'a'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                HO_B1=[each for each in vars()['HO_list_domain'+str(i+1)+'b'] if ('_'+SORBATE_LIST[i][N]+str(N+1)+'_') in each]
                for NN in range(len(HO_A1)):
                    vars()['gp_HO'+str(NN+1)+'_set'+str(N+1)+'_D'+str(i+1)]=domain_class_1.grouping_discrete_layer3(domain=[vars()['domain'+str(i+1)+'A'],vars()['domain'+str(i+1)+'B']],\
                          atom_ids=[HO_A1[NN],HO_B1[NN]],sym_array=[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]])
#####################################do bond valence matching###################################
if USE_BV:
    for i in range(DOMAIN_NUMBER):
        lib_sorbate={}
        if SORBATE_NUMBER[i]!=0:
            lib_sorbate=domain_creator.create_sorbate_match_lib4_test(metal=SORBATE_LIST[i],HO_list=vars()['HO_list_domain'+str(int(i+1))+'a'],anchors=SORBATE_ATTACH_ATOM[i],anchor_offsets=SORBATE_ATTACH_ATOM_OFFSET[i],domain_tag=i+1)
        if DOMAIN[i]==1:
            if half_layer_pick[i]==3:
                vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_2_0_D'+str(int(i+1))+'A','Fe1_3_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            elif half_layer_pick[i]==2:
                vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=['Fe1_8_0_D'+str(int(i+1))+'A','Fe1_9_0_D'+str(int(i+1))+'A']),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)
        elif DOMAIN[i]==2:
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.create_match_lib_before_fitting(domain_class=vars()['domain_class_'+str(int(i+1))],domain=vars()['domain_class_'+str(int(i+1))].build_super_cell(ref_domain=vars()['domain_class_'+str(int(i+1))].create_equivalent_domains_2()[0],rem_atom_ids=None),atm_list=vars()['atm_list_'+str(int(i+1))+'A'],search_range=2.3)
            vars()['match_lib_'+str(int(i+1))+'A']=domain_creator.merge_two_libs(vars()['match_lib_'+str(int(i+1))+'A'],lib_sorbate)
        if INCLUDE_HYDROGEN:
            #print HB_MATCH_1
            for key in vars()['HB_MATCH_'+str(i+1)].keys():
                if key not in vars()['match_lib_'+str(int(i+1))+'A'].keys():
                    vars()['match_lib_'+str(int(i+1))+'A'][key]=vars()['HB_MATCH_'+str(i+1)][key]
                else:
                    vars()['match_lib_'+str(int(i+1))+'A'][key]=vars()['match_lib_'+str(int(i+1))+'A'][key]+vars()['HB_MATCH_'+str(i+1)][key]

###################################fitting function part##########################################
VARS=vars()#pass local variables to sim function
if COUNT_TIME:t_1=datetime.now()

def Sim(data,VARS=VARS):
    eval_in_sim()
    VARS=VARS
    F =[]
    bv=0
    bv_container={}
    fom_scaler=[]
    beta=rgh.beta
    SCALES=[getattr(rgh,scale) for scale in scales]
    total_wt=0
    domain={}

    for i in range(DOMAIN_NUMBER):
        #extract the fitting par values in the associated attribute and then do the scaling(initiation+processing, actually update the fitting parameter values)
        #VARS['domain_class_'+str(int(i+1))].init_sim_batch(batch_path_head+VARS['sim_batch_file_domain'+str(int(i+1))])
        #VARS['domain_class_'+str(int(i+1))].scale_opt_batch(batch_path_head+VARS['scale_operation_file_domain'+str(int(i+1))])
        #grap wt for each domain and cal the total wt
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]

        if INCLUDE_HYDROGEN:
            #update hydrogen for surface oxygens
            for i_H in range(len(COVALENT_HYDROGEN_ACCEPTOR[i])):
                for j_H in range(COVALENT_HYDROGEN_NUMBER[i][i_H]):
                    coor=VARS['domain_class_'+str(int(i+1))].adding_hydrogen(domain=VARS['domain'+str(int(i+1))+'A'],N_of_HB=j_H,ref_id=COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'A',r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_H_'+str(i_H+1)+'_'+str(j_H+1)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta_H_'+str(i_H+1)+'_'+str(j_H+1)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_H_'+str(i_H+1)+'_'+str(j_H+1)))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_H+1)+'_'+COVALENT_HYDROGEN_ACCEPTOR[i][i_H]+'_D'+str(i+1)+'B'],els=['H'])

        #update sorbates
        if UPDATE_SORBATE_IN_SIM:

            for j in range(sum(VARS['SORBATE_NUMBER'][i])):
                SORBATE_coors_a=[]
                O_coors_a=[]

                if len(VARS['SORBATE_ATTACH_ATOM'][i][j])==1 and not USE_COORS[i][j]:#monodentate case
                    top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_MD_'+str(j/2*2))
                    phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_MD_'+str(j/2*2))
                    r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_MD_'+str(j/2*2))
                    ids=VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A'
                    if 'HO' in ids:
                        ids=VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
                    offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j][0]
                    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
                    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                    #O_id=VARS['HO_list_domain'+str(int(i+1))+'a'][O_index[j]:O_index[j+1]]#O_ide is a list of str
                    sorbate_coors=[]
                    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,r=r,attach_atm_ids=ids,offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=VARS['MIRROR'][i][j])
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_octahedral_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,r=r,attach_atm_id=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id)
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_tetrahedral_monodentate(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,r=r,attach_atm_id=ids,offset=offset,sorbate_id=SORBATE_id,O_id=O_id,sorbate_el=SORBATE_LIST[i][j])
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_MD'),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_MD'),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_MD'))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    SORBATE_coors_a.append(sorbate_coors[0])
                    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                    #now put on sorbate on the symmetrically related domain
                    sorbate_ids=[SORBATE_id_B]+O_id_B
                    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==2 and not USE_COORS[i][j]:#bidentate case
                    ids,offset=None,SORBATE_ATTACH_ATOM_OFFSET[i][j]
                    anchor,anchor_offset=None,None
                    if "HO" in VARS['SORBATE_ATTACH_ATOM'][i][j][0]:#a sign for ternary complex structure forming
                        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
                        anchor=BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A'
                        anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]
                    else:
                        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A']
                        if ANCHOR_REFERENCE[i][j]!=None:
                            anchor=ANCHOR_REFERENCE[i][j]+'_D'+str(int(i+1))+'A'
                            anchor_offset=ANCHOR_REFERENCE_OFFSET[i][j]

                    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                    sorbate_coors=[]
                    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                        if (i+j)%2==1:edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
                        else:edge_offset=-getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
                        edge_offset2=getattr(VARS['rgh_domain'+str(int(i+1))],'offset2_BD_'+str(j/2*2))
                        angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset_BD_'+str(j/2*2))
                        top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_BD_'+str(j/2*2))
                        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
                        if not USE_TOP_ANGLE:
                            r1=getattr(VARS['rgh_domain'+str(int(i+1))],'r_BD_'+str(j/2*2))
                            r2=r1+getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2))
                            l=domain_creator.extract_coor_offset(domain=VARS['domain'+str(int(i+1))+'A'],id=ids,offset=offset,basis=[5.038,5.434,7.3707])
                            top_angle=np.arccos((r1**2+r2**2-l**2)/2/r1/r2)/np.pi*180
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_pyramid_distortion_B2(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,phi=phi,edge_offset=[edge_offset,edge_offset2],attach_atm_ids=ids,offset=offset,anchor_ref=anchor,anchor_offset=anchor_offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,mirror=VARS['MIRROR'][i][j/2],angle_offset=angle_offset)
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=[],anchor_ref=anchor,anchor_offset=anchor_offset)
                            if (i+j)%2==1:
                                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                            else:
                                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                        else:
                            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_octahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,attach_atm_ids=ids,offset=offset,sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset)
                    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                        phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_BD_'+str(j/2*2))
                        if (i+j)%2==1:
                            ids=ids[::-1]
                            offset=offset[::-1]
                            phi=-phi
                        top_angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_offset_BD_'+str(j/2*2))
                        edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'anchor_offset_BD_'+str(j/2*2))

                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,distal_length_offset=[0,0],distal_angle_offset=[0,0],top_angle_offset=top_angle_offset,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=[],anchor_ref=anchor,anchor_offset=anchor_offset,edge_offset=edge_offset)
                            if (i+j)%2==1:
                                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                            else:
                                [sorbate_coors.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_BD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                        else:
                            angle_offsets=[getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset_BD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'angle_offset2_BD_'+str(j/2*2))]
                            distal_length_offset=[getattr(VARS['rgh_domain'+str(int(i+1))],'offset_BD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'offset2_BD_'+str(j/2*2))]
                            if (i+j)%2==1:
                                distal_length_offset=distal_length_offset[::-1]
                                angle_offsets=-np.array(angle_offsets[::-1])
                            sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_bidentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],phi=phi,distal_length_offset=distal_length_offset,distal_angle_offset=angle_offsets,top_angle_offset=top_angle_offset,attach_atm_ids=ids,offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_id=O_id,anchor_ref=anchor,anchor_offset=anchor_offset,edge_offset=edge_offset)
                    SORBATE_coors_a.append(sorbate_coors[0])
                    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                    #now put on sorbate on the symmetrically related domain
                    sorbate_ids=[SORBATE_id_B]+O_id_B
                    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                elif len(VARS['SORBATE_ATTACH_ATOM'][i][j])==3 and not USE_COORS[i][j]:#tridentate case (no oxygen sorbate here considering it is a trigonal pyramid structure)
                    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                        top_angle=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_TD_'+str(j/2*2))
                        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
                        if 'HO' in ids[0]:
                            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
                        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_pb_share_triple4(domain=VARS['domain'+str(int(i+1))+'A'],top_angle=top_angle,attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j])
                        SORBATE_coors_a.append(sorbate_coors[0])
                        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                        #now put on sorbate on the symmetrically related domain
                        sorbate_ids=[SORBATE_id_B]
                        sorbate_els=[SORBATE_LIST[i][j]]
                        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
                        if 'HO' in ids[0]:
                            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
                        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                        dr=[getattr(VARS['rgh_domain'+str(int(i+1))],'dr1_oct_TD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'dr2_oct_TD_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'dr3_oct_TD_'+str(j/2*2))]
                        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                        O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_share_triple_octahedra(domain=VARS['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id,dr=dr,mirror=MIRROR[i][j/2])
                        SORBATE_coors_a.append(sorbate_coors[0])
                        #sorbate_offset=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)-domain_creator.extract_coor2(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                        O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                        #now put on sorbate on the symmetrically related domain
                        sorbate_ids=[SORBATE_id_B]+O_id_B
                        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                        ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+'_D'+str(int(i+1))+'A']
                        if 'HO' in ids[0]:
                            ids=[VARS['SORBATE_ATTACH_ATOM'][i][j][0]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][1]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A',VARS['SORBATE_ATTACH_ATOM'][i][j][2]+BASAL_EL[i][j/2]+str(j-1)+'_D'+str(int(i+1))+'A']
                        offset=VARS['SORBATE_ATTACH_ATOM_OFFSET'][i][j]
                        SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]
                        edge_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'dr_tetrahedral_TD_'+str(j/2*2))
                        angle_offset=getattr(VARS['rgh_domain'+str(int(i+1))],'dr_bc_tetrahedral_TD_'+str(j/2*2))
                        O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].adding_sorbate_tridentate_tetrahedral(domain=VARS['domain'+str(int(i+1))+'A'],attach_atm_ids_ref=ids[0:2],attach_atm_id_third=[ids[-1]],offset=offset,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],sorbate_oxygen_ids=O_id,edge_offset=edge_offset,top_angle_offset=angle_offset)
                        SORBATE_coors_a.append(sorbate_coors[0])
                        #sorbate_offset=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)-domain_creator.extract_coor2(VARS['domain'+str(int(i+1))+'A'],SORBATE_id)
                        if ADD_DISTAL_LIGAND_WILD[i][j]:
                            if (i+j)%2==1:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                            else:
                                [O_coors_a.append(domain_class_1.adding_distal_ligand(domain=VARS['domain'+str(int(i+1))+'A'],id=O_id[ligand_id],ref=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],SORBATE_id),r=getattr(VARS['rgh_domain'+str(int(i+1))],'r1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)),phi=180-getattr(VARS['rgh_domain'+str(int(i+1))],'phi1_'+str(ligand_id+1)+'_TD_'+str(j/2*2)))) for ligand_id in range(len(O_id))]
                        else:
                            [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                        SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                        O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                        #now put on sorbate on the symmetrically related domain
                        sorbate_ids=[SORBATE_id_B]+O_id_B
                        sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                        domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)
                elif not USE_COORS[i][j]:#outer-sphere case
                    phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_OS_'+str(j/2*2))
                    r_Pb_O=getattr(VARS['rgh_domain'+str(int(i+1))],'r0_OS_'+str(j/2*2))
                    O_Pb_O_ang=getattr(VARS['rgh_domain'+str(int(i+1))],'top_angle_OS_'+str(j/2*2))
                    ct_offset_dx=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dx_OS_'+str(j/2*2))
                    ct_offset_dy=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dy_OS_'+str(j/2*2))
                    ct_offset_dz=getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dz_OS_'+str(j/2*2))
                    rot_x,rot_y,rot_z=getattr(VARS['rgh_domain'+str(int(i+1))],'rot_x_OS_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'rot_y_OS_'+str(j/2*2)),getattr(VARS['rgh_domain'+str(int(i+1))],'rot_z_OS_'+str(j/2*2))
                    ref_x,ref_y,ref_z=OS_X_REF[i][j],OS_Y_REF[i][j],OS_Z_REF[i][j]
                    if (j+i)%2==1:
                        phi=180-phi#note all angles in degree
                        ct_offset_dx=-getattr(VARS['rgh_domain'+str(int(i+1))],'ct_offset_dx_OS_'+str(j/2*2))
                        rot_y,rot_z=-rot_y,-rot_z
                    SORBATE_id=VARS['SORBATE_list_domain'+str(int(i+1))+'a'][j]#pb_id is a str NOT list
                    O_id=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'a'] if SORBATE_id in HO_id]
                    consider_distal=False
                    if O_id!=[]:
                        consider_distal=True
                    sorbate_coors=[]
                    if LOCAL_STRUCTURE[i][j/2]=='trigonal_pyramid':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_complex_2(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r_Pb_O=r_Pb_O,O_Pb_O_ang=O_Pb_O_ang,phi=phi,pb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
                    elif LOCAL_STRUCTURE[i][j/2]=='octahedral':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_complex_oct(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r0=r_Pb_O,phi=phi,Sb_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal)
                    elif LOCAL_STRUCTURE[i][j/2]=='tetrahedral':
                        sorbate_coors=VARS['domain_class_'+str(int(i+1))].outer_sphere_tetrahedral2(domain=VARS['domain'+str(int(i+1))+'A'],cent_point=[ref_x+ct_offset_dx,ref_y+ct_offset_dy,ref_z+ct_offset_dz],r_sorbate_O=r_Pb_O,phi=phi,sorbate_id=SORBATE_id,sorbate_el=SORBATE_LIST[i][j],O_ids=O_id,distal_oxygen=consider_distal,rotation_x=rot_x,rotation_y=rot_y,rotation_z=rot_z)

                    SORBATE_coors_a.append(sorbate_coors[0])
                    [O_coors_a.append(sorbate_coors[k]) for k in range(len(sorbate_coors))[1:]]
                    SORBATE_id_B=VARS['SORBATE_list_domain'+str(int(i+1))+'b'][j]
                    O_id_B=[HO_id for HO_id in VARS['HO_list_domain'+str(int(i+1))+'b'] if SORBATE_id_B in HO_id]
                    #now put on sorbate on the symmetrically related domain
                    sorbate_ids=[SORBATE_id_B]+O_id_B
                    sorbate_els=[SORBATE_LIST[i][j]]+['O']*(len(O_id_B))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=np.array(SORBATE_coors_a+O_coors_a)*[-1,1,1]-[-1.,0.06955,0.5],ids=sorbate_ids,els=sorbate_els)

        if INCLUDE_HYDROGEN:
            #update hydrogen for distal oxygens
            for i_distal in range(len(VARS['HO_list_domain'+str(int(i+1))+'a'])):
                for j_distal in range(PROTONATION_DISTAL_OXYGEN[i][i_distal]):
                    coor=VARS['domain_class_'+str(int(i+1))].adding_hydrogen(domain=VARS['domain'+str(int(i+1))+'A'],N_of_HB=j_distal,ref_id=VARS['HO_list_domain'+str(int(i+1))+'a'][i_distal],r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_H_D_'+str(i_distal+1)+'_'+str(j_distal+1)))
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_distal+1)+'_'+VARS['HO_list_domain'+str(int(i+1))+'b'][i_distal]],els=['H'])

        if WATER_NUMBER[i]!=0:#add water molecules if any
            if WATER_PAIR:
                for jj in range(WATER_NUMBER[i]/2):#note will add water pair (two oxygens) each time, and you can't add single water
                    O_ids_a=VARS['Os_list_domain'+str(int(i+1))+'a'][jj*2:jj*2+2]
                    O_ids_b=VARS['Os_list_domain'+str(int(i+1))+'b'][jj*2:jj*2+2]
                    alpha=getattr(VARS['rgh_domain'+str(int(i+1))],'alpha_W_'+str(jj+1))
                    r=0.5*5.434/2./np.sin(alpha/180.*np.pi)#here r is constrained by the condition of y1-y2=0.5
                    v_shift=getattr(VARS['rgh_domain'+str(int(i+1))],'v_shift_W_'+str(jj+1))
                    #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                    H2O_coors_a=VARS['domain_class_'+str(int(i+1))].add_oxygen_pair2B(domain=VARS['domain'+str(int(i+1))+'A'],O_ids=O_ids_a,ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=v_shift,r=r,alpha=alpha)
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O','O'])
                    if INCLUDE_HYDROGEN:
                        #update hydrogen for water molecules
                        for i_water in [0,1]:#two waters considered
                            for j_water in [0,1]:#doubly protonated for each water
                                coor=VARS['domain_class_'+str(int(i+1))].adding_hydrogen(domain=VARS['domain'+str(int(i+1))+'A'],N_of_HB=j_water,ref_id=O_ids_a[i_water],r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_H_W_'+str(jj+1)+'_'+str(i_water+1)+'_'+str(j_water+1)))
                                domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_water+1)+'_'+O_ids_b[i_water]],els=['H'])

            else:
                for jj in range(WATER_NUMBER[i]):#note will add single water each time
                    O_ids_a=[VARS['Os_list_domain'+str(int(i+1))+'a'][jj]]
                    O_ids_b=[VARS['Os_list_domain'+str(int(i+1))+'b'][jj]]
                    v_shift=getattr(VARS['rgh_domain'+str(int(i+1))],'v_shift_W_'+str(jj+1))
                    #set the first pb atm to be the ref atm(if you have two layers, same ref point but different height)
                    H2O_coors_a=VARS['domain_class_'+str(int(i+1))].add_single_oxygen(domain=VARS['domain'+str(int(i+1))+'A'],O_id=O_ids_a,ref_id=map(lambda x:x+'_D'+str(i+1)+'A',REF_POINTS[i][jj]),v_shift=v_shift)
                    domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=H2O_coors_a*[-1,1,1]-[-1.,0.06955,0.5],ids=O_ids_b,els=['O'])
                    if INCLUDE_HYDROGEN:
                        #update hydrogen for water molecules
                        for i_water in [0]:#two waters considered
                            for j_water in [0,1]:#doubly protonated for each water
                                coor=VARS['domain_class_'+str(int(i+1))].adding_hydrogen(domain=VARS['domain'+str(int(i+1))+'A'],N_of_HB=j_water,ref_id=O_ids_a[i_water],r=getattr(VARS['rgh_domain'+str(int(i+1))],'r_H_W_'+str(i_water+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),theta=getattr(VARS['rgh_domain'+str(int(i+1))],'theta_H_W_'+str(i_water+1)+'_'+str(i_water+1)+'_'+str(j_water+1)),phi=getattr(VARS['rgh_domain'+str(int(i+1))],'phi_H_W_'+str(i_water+1)+'_'+str(i_water+1)+'_'+str(j_water+1)))
                                domain_creator.add_atom(domain=VARS['domain'+str(int(i+1))+'B'],ref_coor=[np.array(coor)*[-1,1,1]-[-1.,0.06955,0.5]],ids=['HB'+str(j_water+1)+'_'+O_ids_b[i_water]],els=['H'])

        if USE_BV and i in DOMAINS_BV:
            #set up dynamic super cells,where water and sorbate is a library and surface is a domain instance
            def _widen_validness(value):#acceptable bond valence offset can be adjusted (here is 0.08)
                if value<BV_TOLERANCE[0]:return 100
                elif value>=BV_TOLERANCE[0] and value<BV_TOLERANCE[1]:return 0
                else:return value
            def _widen_validness_range(value_min,value_max):#consider a range of (ideal_bv-temp_bv)
                if (value_min<BV_TOLERANCE[0] and value_max>BV_TOLERANCE[1]) or (value_min>=BV_TOLERANCE[0] and value_min<=BV_TOLERANCE[1]) or (value_max>=BV_TOLERANCE[0] and value_max<=BV_TOLERANCE[1]):
                    return 0
                elif value_min>BV_TOLERANCE[1]:return value_min
                else:return 100
            def _widen_validness_hydrogen_acceptor(value,H_N=0):#here consider possible contribution of hydrogen bond (~0.2)
                if (value-H_N*0.2)<BV_TOLERANCE[0]:return 100
                elif (value-H_N*0.2)>=BV_TOLERANCE[0] and (value-H_N*0.2)<BV_TOLERANCE[1]:return 0
                else:return (value-H_N*0.2)
            def _widen_validness_potential_hydrogen_acceptor(value):#value=2-temp_bv(temp_bv include covalent hydrogen bond possibly)
                if value<0.2 and value>BV_TOLERANCE[0]: return 0
                elif value<BV_TOLERANCE[0]: return 100
                else:return value

            super_cell_sorbate,super_cell_surface,super_cell_water=None,None,None
            if WATER_NUMBER[i]!=0:
                if DOMAIN[i]==1:
                    super_cell_water=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+[4,5]+range(-(sum(SORBATE_NUMBER[i])+WATER_NUMBER[i]+sum([np.sum(N_list) for N_list in O_NUMBER[i]])),0))
                elif DOMAIN[i]==2:
                    super_cell_water=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1,2,3]+range(-(sum(SORBATE_NUMBER[i])+WATER_NUMBER[i]+sum([np.sum(N_list) for N_list in O_NUMBER[i]])),0))

            def _return_right_value(value):
                if value:return value
                else:return 1
            NN=_return_right_value(sum(SORBATE_NUMBER[i]))#number of sorbate sets(1 or 2)
            N_HB_SURFACE,N_HB_DISTAL=0,0
            if INCLUDE_HYDROGEN:
                N_HB_SURFACE=sum(COVALENT_HYDROGEN_NUMBER[i])#number of hydrogen for surface oxygens
                N_HB_DISTAL=sum(PROTONATION_DISTAL_OXYGEN[i])#number of hydrogen for distal oxygens
            O_N_for_this_domain=O_NUMBER[i]
            total_sorbate_number=sum(SORBATE_NUMBER[i])+sum(O_N_for_this_domain)
            #the idea is that we want to have only one set of sorbate and hydrogen within each domain (ie don't count symmetry counterpart twice)
            def _cal_segment2(O_N_list=[],water_N=0):
                cum_list=[-water_N]
                segment2_boundary=[]
                segment2=[]
                for O_N in O_N_list[::-1]:
                    cum_list.append(cum_list[-1]-(O_N+1))
                for i in range(0,len(cum_list)-1,2):
                    segment2_boundary.append([cum_list[i],cum_list[i+1]])
                for each in segment2_boundary:
                    segment2=segment2+range(each[1],each[0])
                return segment2

            segment1=range(-WATER_NUMBER[i],0)
            segment2=_cal_segment2(O_N_for_this_domain,WATER_NUMBER[i])
            segment3=range(-(WATER_NUMBER[i]+total_sorbate_number),-(WATER_NUMBER[i]+total_sorbate_number))

            if INCLUDE_HYDROGEN:
                segment1=range(-(N_HB_DISTAL/NN+WATER_NUMBER[i]*3),0)
                segment2=range(-(WATER_NUMBER[i]*3+N_HB_DISTAL+total_sorbate_number/NN),-(WATER_NUMBER[i]*3+N_HB_DISTAL))
                segment3=range(-(WATER_NUMBER[i]*3+N_HB_DISTAL+total_sorbate_number+N_HB_SURFACE),-(WATER_NUMBER[i]*3+N_HB_DISTAL+total_sorbate_number))
            if DOMAIN[i]==1:
                #note here if there are two symmetry pair, then only consider one of the couple for bv consideration, the other one will be skipped in the try except statement
                super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(4,8)+segment1+segment2+segment3)
                if SEARCH_MODE_FOR_SURFACE_ATOMS:
                    super_cell_surface=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],[0,1]+range(4,30)+segment1+segment2+segment3)
                else:
                    super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()
                    #delete the first iron layer atoms if considering a half layer
                    super_cell_surface.del_atom(super_cell_surface.id[2])
                    super_cell_surface.del_atom(super_cell_surface.id[2])
            elif DOMAIN[i]==2:
                super_cell_sorbate=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],range(0,6)+segment1+segment2+segment3)
                if SEARCH_MODE_FOR_SURFACE_ATOMS:
                    super_cell_surface=domain_class_1.build_super_cell2_simple(VARS['domain'+str(i+1)+'A'],range(0,30)+segment1+segment2+segment3)
                else:
                    super_cell_surface=VARS['domain'+str(i+1)+'A'].copy()

            #consider hydrogen bond only among interfacial water molecules and top surface Oxygen layer and Oxygen ligand
            if not INCLUDE_HYDROGEN:
                if WATER_NUMBER[i]!=0:
                    water_ids=VARS['Os_list_domain'+str(int(i+1))+'a']
                    for id in water_ids:
                        tmp_bv=domain_class_1.cal_hydrogen_bond_valence2B(super_cell_water,id,3.,2.5,BOND_VALENCE_WAIVER)*int(CONSIDER_WATER_IN_BV)
                        bv=bv+tmp_bv
                        if debug_bv:bv_container[id]=tmp_bv
            #cal bv for surface atoms and sorbates
            #only consdier domainA since domain B is symmetry related to domainA
            waiver_box=[]#the first set of anchored oxygens will be waived for being considered for bond valence constraints
            attach_atom_ids=VARS['SORBATE_ATTACH_ATOM'][i]
            if len(attach_atom_ids)==0:
                pass
            elif len(attach_atom_ids)!=0 and len(attach_atom_ids)%2==0:
                if len(attach_atom_ids[0])<3:#only for monodentate and bidentate binding mode
                    waiver_box=map(lambda x:x+'_D'+str(i+1)+'A',attach_atom_ids[0])
                else:
                    pass

            for key in [each_key for each_key in VARS['match_lib_'+str(i+1)+'A'].keys() if each_key not in waiver_box]:
                temp_bv=None
                if ([sorbate not in key for sorbate in SORBATE_EL_LIST]==[True]*len(SORBATE_EL_LIST)) and ("HO" not in key) and ("Os" not in key):#surface atoms
                    if SEARCH_MODE_FOR_SURFACE_ATOMS:#cal temp_bv based on searching within spherical region
                        el=None
                        if "Fe" in key: el="Fe"
                        elif "O" in key and "HB" not in key: el="O"
                        elif "HB" in key: el="H"
                        if el=="H":
                            temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_surface,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
                        else:
                            temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_surface,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                    else:
                        #no searching in this algorithem
                        temp_bv=domain_class_1.cal_bond_valence4B(super_cell_surface,key,VARS['match_lib_'+str(i+1)+'A'][key],2.5)
                else:#sorbates including water
                    #searching included in this algorithem
                    if "HO" in key and COUNT_DISTAL_OXYGEN:#distal oxygen and its associated hydrogen
                        el="O"
                        if "HB" in key:
                            el="H"
                        if el=="O":
                            try:
                                temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                            except:
                                temp_bv=2
                        else:
                            try:
                                temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_sorbate,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
                            except:
                                temp_bv=1
                    elif "HO" in key and not COUNT_DISTAL_OXYGEN:
                        temp_bv=2
                    elif "Os" in key:#water and the associated hydrogen
                        el="O"
                        if "HB" in key:
                            el="H"
                        if el=="O":
                            temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,el,SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],50,False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                        else:
                            temp_bv=domain_class_1.cal_bond_valence1_new2B_4(super_cell_sorbate,key,el,2.5,VARS['match_lib_'+str(i+1)+'A'][key],1,False,R0_BV,2.5)['total_valence']
                    else:#metals
                        try:
                            temp_bv=domain_class_1.cal_bond_valence1_new2B_7_2(super_cell_sorbate,key,SORBATE_LIST[i][j],SEARCH_RANGE_OFFSET,IDEAL_BOND_LENGTH,VARS['match_lib_'+str(i+1)+'A'][key],SEARCHING_PARS['sorbate'][1],False,R0_BV,2.5,BOND_VALENCE_WAIVER,check=not running_mode)['total_valence']
                        except:
                            temp_bv=METAL_BV[i][int(key.rsplit('_')[0][-1])-1][0]

                if PRINT_BV:print key, temp_bv
                #consider possible hydrogen bond and hydroxyl bond fro oxygen atoms
                if 'O' in key:
                    if INCLUDE_HYDROGEN:
                        #For O you may consider possible binding to proton (+0.8)
                        #And note the maximum coordination number for O is 4
                        if "HB" not in key:
                            bv=bv+_widen_validness(2-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                        else:
                            bv=bv+_widen_validness(1-temp_bv)
                            if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                    else:
                        #For O you may consider possible binding to proton (+0.8)
                        #And note the maximum coordination number for O is 4
                        case_tag=len(VARS['match_lib_'+str(i+1)+'A'][key])#current coordination number
                        if COVALENT_HYDROGEN_RANDOM==True:
                            if case_tag<4 and key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR[i]):
                                C_H_N=range(4-case_tag+1)#max CN allowed is 4,if case_tag=3, then range(4-case_tag+1)=[0,1]
                                bv_offset=[ _widen_validness_range(2-0.88*N-temp_bv,2-0.68*N-temp_bv) for N in C_H_N]
                                C_H_N=C_H_N[bv_offset.index(min(bv_offset))]
                                case_tag=case_tag+C_H_N#CN after considering the proton
                                if PRINT_PROTONATION:
                                    print key,C_H_N
                                if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider potential hydrogen bond (you can have or have not H-bonding)
                                    if _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==0 or _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==100:
                                    #if saturated already or over-saturated, then adding H-bonding wont help decrease the the total bv anyhow
                                    #or reach the maximum CN(4), the adding one hydrogen bond is not allowed
                                        bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                        if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                    else:#you can add one hydrogen bond at most
                                    #if undersaturation, then compare the cases of inclusion of H-bonding and exclusion of H-bonding. Whichever give rise to the lower bv will be used.
                                        bv=bv+min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                                        if debug_bv:bv_container[key]=min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                                else:#consider covalent hydrogen bond only
                                    bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                    if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            elif case_tag==4:#coordination saturation achieved, so neighter covalent hydrogen bond nor hydrogen bond
                                bv=bv+_widen_validness(2-temp_bv)
                                if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                        else:
                            if key in map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]):
                                #if consider convalent hydrogen bond (bv=0.68 to 0.88) while the hydrogen bond has bv from 0.13 to 0.25
                                C_H_N=COVALENT_HYDROGEN_NUMBER[i][map(lambda x:x+'_D'+str(i+1)+'A',COVALENT_HYDROGEN_ACCEPTOR[i]).index(key)]
                                case_tag=case_tag+C_H_N
                                if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider potential hydrogen bond (you can have or have not H-bonding)
                                    if _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==0 or _widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)==100 or case_tag>=4:
                                    #if saturated already or over-saturated, then adding H-bonding wont help decrease the the total bv anyhow
                                        bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                        if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                    else:
                                    #if undersaturation, then compare the cases of inclusion of H-bonding and exclusion of H-bonding. Whichever give rise to the lower bv will be used.
                                        bv=bv+min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                                        if debug_bv:bv_container[key]=min([_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv),_widen_validness_range(2-0.88*C_H_N-temp_bv-0.25,2-0.68*C_H_N-temp_bv)])
                                else:
                                    bv=bv+_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                                    if debug_bv:bv_container[key]=_widen_validness_range(2-0.88*C_H_N-temp_bv,2-0.68*C_H_N-temp_bv)
                            else:
                                if key in map(lambda x:x+'_D'+str(i+1)+'A',POTENTIAL_HYDROGEN_ACCEPTOR[i]):#consider hydrogen bond
                                    if _widen_validness(2-temp_bv)==0 or _widen_validness(2-temp_bv)==100 or case_tag>=4:
                                        bv=bv+_widen_validness(2-temp_bv)
                                        if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                                    else:
                                        bv=bv+min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                                        if debug_bv:bv_container[key]=min([_widen_validness(2-temp_bv),_widen_validness_range(2-temp_bv-0.25,2-temp_bv-0.13)])
                                else:
                                    bv=bv+_widen_validness(2-temp_bv)
                                    if debug_bv:bv_container[key]=_widen_validness(2-temp_bv)
                elif 'Fe' in key:
                    bv=bv+_widen_validness(3-temp_bv)
                    if debug_bv:bv_container[key]=_widen_validness(3-temp_bv)
                else:#do metal sorbates
                    metal_bv_range=[]
                    metal_bv_range=METAL_BV[i][int(key.rsplit('_')[0][-1])-1]
                    bv=bv+_widen_validness_range(metal_bv_range[0]-temp_bv,metal_bv_range[1]-temp_bv)
                    if debug_bv:bv_container[key]=_widen_validness_range(metal_bv_range[0]-temp_bv,metal_bv_range[1]-temp_bv)

    if debug_bv:
        print "Print out the species, which are not under bond valence saturation"
        for i in bv_container.keys():
            if bv_container[i]!=0:
                print i,"BV after considering penalty",bv_container[i]
    #set up multiple domains
    #note for each domain there are two sub domains which symmetrically related to each other, so have equivalent wt
    for i in range(DOMAIN_NUMBER):
        #extract layered water info
        u0,ubar,d_w,first_layer_height,density_w=0,0,0,0,0
        ref_height=None
        layered_water_A,layered_water_B=[],[]
        if layered_water_pars['yes_OR_no'][i]:
            u0=getattr(VARS['rgh_domain'+str(int(i+1))],'u0')
            ubar=getattr(VARS['rgh_domain'+str(int(i+1))],'ubar')
            d_w=getattr(VARS['rgh_domain'+str(int(i+1))],'d_w')
            first_layer_height=getattr(VARS['rgh_domain'+str(int(i+1))],'first_layer_height')
            density_w=getattr(VARS['rgh_domain'+str(int(i+1))],'density_w')
            ref_atom=layered_water_pars['ref_layer_height'][i]+'_D'+str(i+1)+'A'
            ref_height=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],ref_atom)[2]
            layered_water_A=[u0,ubar,d_w,first_layer_height/7.3707+ref_height,density_w]#7.3707 is specifically for hematite rcut
            layered_water_B=[u0,ubar,d_w,first_layer_height/7.3707+ref_height-0.5,density_w]#symmetry related domain has height offset of 0.5
        wt_DA=getattr(VARS['rgh_domain'+str(int(i+1))],'wt_domainA')
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':wt_DA*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_A}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':(1-wt_DA)*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_B}
        #extract layered sorbate info
        u0_s,ubar_s,d_s,first_layer_height_s,density_s=0,0,0,0,0
        ref_height_s=None
        layered_sorbate_A,layered_sorbate_B=[],[]
        if layered_sorbate_pars['yes_OR_no'][i]:
            u0_s=getattr(VARS['rgh_domain'+str(int(i+1))],'u0_s')
            ubar_s=getattr(VARS['rgh_domain'+str(int(i+1))],'ubar_s')
            d_s=getattr(VARS['rgh_domain'+str(int(i+1))],'d_s')
            first_layer_height_s=getattr(VARS['rgh_domain'+str(int(i+1))],'first_layer_height_s')
            density_s=getattr(VARS['rgh_domain'+str(int(i+1))],'density_s')
            ref_atom_s=layered_sorbate_pars['ref_layer_height'][i]+'_D'+str(i+1)+'A'
            ref_height_s=domain_creator.extract_coor(VARS['domain'+str(int(i+1))+'A'],ref_atom_s)[2]
            layered_sorbate_A=[layered_sorbate_pars['el'],u0_s,ubar_s,d_s,first_layer_height_s/7.3707+ref_height_s,density_s,F1F2]#7.3707 is specifically for hematite rcut
            layered_sorbate_B=[layered_sorbate_pars['el'],u0_s,ubar_s,d_s,first_layer_height_s/7.3707+ref_height_s-0.5,density_s,F1F2]#symmetry related domain has height offset of 0.5
        wt_DA=getattr(VARS['rgh_domain'+str(int(i+1))],'wt_domainA')
        domain['domain'+str(int(i+1))+'A']={'slab':VARS['domain'+str(int(i+1))+'A'],'wt':wt_DA*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_A,'layered_sorbate':layered_sorbate_A}
        domain['domain'+str(int(i+1))+'B']={'slab':VARS['domain'+str(int(i+1))+'B'],'wt':(1-wt_DA)*vars()['wt_domain'+str(int(i+1))]/total_wt,'layered_water':layered_water_B,'layered_sorbate':layered_sorbate_B}

    if COUNT_TIME:t_2=datetime.now()

    #cal structure factor for each dataset in this for loop
    i=0
    for data_set in data:
        if data_set.x[0]>15:#doing RAXR calculation(x is energy column typically in magnitude of 10000 ev)
            a=getattr(VARS['rgh_raxr'],'a'+str(i+1))
            b=getattr(VARS['rgh_raxr'],'b'+str(i+1))
            A_list,P_list=[],[]
            for index_resonant_el in range(len(RESONANT_EL_LIST)):
                A_list_domain=0
                P_list_domain=0
                if RESONANT_EL_LIST[index_resonant_el]!=0:
                    A_list_domain=getattr(VARS['rgh_raxr'],'A_D'+str(index_resonant_el+1)+'_'+str(i+1))
                    P_list_domain=getattr(VARS['rgh_raxr'],'P_D'+str(index_resonant_el+1)+'_'+str(i+1))
                A_list.append(A_list_domain)
                P_list.append(P_list_domain)
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            x = data_set.x
            y = data_set.extra_data['Y']
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                f = SCALES[0]*rough*sample.calc_f4_specular_RAXR(h, k, y, x, E0, F1F2, a, b, A_list, P_list, RESONANT_EL_LIST)
            else:
                f = SCALES[0]*rough*sample.calc_f4_nonspecular_RAXR(h, k, y, x, E0, F1F2, a, b, A_list, P_list, RESONANT_EL_LIST)
            F.append(abs(f))
            fom_scaler.append(1)
            i+=1
        else:#doing CTR calculation (x is perpendicular momentum transfer L typically smaller than 15)
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            x = data_set.x
            y = data_set.extra_data['Y']
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
            rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(x-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
            if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                f = SCALES[0]*rough*sample.calc_f4_specular(h, k, x)
            else:
                f = SCALES[0]*rough*sample.calc_f4(h, k, x)
            F.append(abs(f))
            fom_scaler.append(1)

    #domain_class_1.find_neighbors2(domain_class_1.build_super_cell(domain2A,['Fe1_2_0_D2A','Fe1_3_0_D2A','Pb2_D2A','HO1_Pb2_D2A']),'HO1_Pb1_D2A',3)
    #print 'As1_D1A',domain_creator.extract_coor(domain1A,'As1_D1A')
    #print domain_creator.extract_component(domain2A,'Pb1_D2A',['dx1','dy2','dz3'])
    #domain_creator.layer_spacing_calculator(domain1A,12,True)
    #domain_class_1.revert_coors_to_geometry_setting_tetrahedra_BD(domain5A,['O1_5_0_D5A','O1_8_0_D5A'],[None,'+x'],'As1_D5A','+y','Fe1_8_0_D5A','+x')
    #domain_creator.print_data_for_publication_B(N_sorbate=4,domain=domain1A,z_shift=1,half_layer=True,full_layer_long=0,save_file='D://model.xyz')

    if PRINT_MODEL_FILES:
        for i in range(DOMAIN_NUMBER):
            N_HB_SURFACE=sum(COVALENT_HYDROGEN_NUMBER[i])
            N_HB_DISTAL=sum(PROTONATION_DISTAL_OXYGEN[i])
            total_sorbate_number=sum(SORBATE_NUMBER[i])+sum([np.sum(N_list) for N_list in O_NUMBER[i]])
            N_sorbate,N_distal_old=np.array(SORBATE_NUMBER[i]),np.array([np.sum(N_list) for N_list in O_NUMBER[i]])
            N_distal=[np.sum(N_distal_old[j*2:j*2+2]) for j in range(len(N_distal_old)/2)]
            N_sorbate_and_distal=N_sorbate+N_distal
            first_item_index=[0]+[np.sum(N_sorbate_and_distal[0:j+1]) for j in range(len(N_sorbate_and_distal)-1)]
            length_of_each_segment=list(N_sorbate_and_distal/2)
            first_item_index.append(np.sum(N_sorbate_and_distal))
            length_of_each_segment.append(WATER_NUMBER[i])
            water_number=WATER_NUMBER[i]*3
            TOTAL_NUMBER=total_sorbate_number+water_number/3
            if INCLUDE_HYDROGEN:
                TOTAL_NUMBER=N_HB_SURFACE+N_HB_DISTAL+total_sorbate_number+water_number
            domain_creator.print_data2(N_sorbate=TOTAL_NUMBER,domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'_dsv.xyz'))
            domain_creator.print_data2C(domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'.xyz'),sorbate_index_list=first_item_index,each_segment_length=length_of_each_segment)
            domain_creator.make_cif_file(domain=VARS['domain'+str(i+1)+'A'],z_shift=1,half_layer=DOMAIN[i]-2,half_layer_long=half_layer_pick[i],full_layer_long=full_layer_pick[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'.cif'),sorbate_index_list=first_item_index,each_segment_length=length_of_each_segment)
            test=xyz.formate_vtk(os.path.join(output_file_path,'Model_domain'+str(i+1)+'.xyz'))
            test.all_in_all()
            #output for publication
            if water_pars['use_default']:
                domain_creator.print_data_for_publication_B2(N_sorbate=np.sum(SORBATE_NUMBER[i])+np.sum(O_NUMBER[i])+WATER_NUMBER[i],domain=VARS['domain'+str(int(i+1))+'A'],z_shift=1,layer_types=(half_layer+full_layer)[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'A_publication.dat'))
            else:
                domain_creator.print_data_for_publication_B2(N_sorbate=np.sum(SORBATE_NUMBER[i])+np.sum(O_NUMBER[i]),domain=VARS['domain'+str(int(i+1))+'A'],z_shift=1,layer_types=(half_layer+full_layer)[i],save_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'A_publication.dat'))
            try:#make sure you have the test.tab file in the specified folder
                domain_creator.make_publication_table2(model_file=os.path.join(output_file_path,'Model_domain'+str(i+1)+'A_publication.dat'),par_file=os.path.join(output_file_path,"test.tab"),el_substrate=['Fe','O'],el_sorbate=['Pb'],abc=[5.038,5.434,7.3707])
            except:
                pass
    #make dummy raxr dataset you will need to double check the LB,dL and the hkl
    DUMMY_RAXR_BUILT=False
    if DUMMY_RAXR_BUILT:
        LB=2
        dL=2
        h,k,l=np.zeros(28),np.zeros(28),np.arange(0.35,9.9,0.35)
        rough_temp = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l-LB)/dL)**2)**0.5
        f1f2_data_calculated=None
        try:
            f1f2_data_calculated=np.loadtxt('C:\\Users\\jackey\\Google Drive\\data\\Lead_CL_output.f1f2')
        except:
            f1f2_data_calculated=np.loadtxt('D:\\Google Drive\\data\\Lead_CL_output.f1f2')
        sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
        aa=rough_temp*sample.calc_f4_specular_RAXR_for_test_purpose(h,k,l,f1f2_data_calculated[:,(1,2)],res_el='Pb')
        try:
            pickle.dump(aa,open("C:\\Users\\jackey\\Google Drive\\useful codes\\plotting\\temp_plot_dummy_raxr","wb"))
        except:
            pickle.dump(aa,open("D:\\Google Drive\\useful codes\\plotting\\temp_plot_dummy_raxr","wb"))
        #after this step you should execute the "D:\Google Drive\useful codes\temp_make_dummy_raxr_data.py" in terminal

    #The A and P list returned is calculated based on the model dependent structure
    Print_AP=False
    if Print_AP:
        AP=sample.find_A_P(np.arange(0,10.38,0.35),'Pb',True)

    #export the model results for plotting if PLOT set to true
    if PLOT:
        sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.1391})
        bl_dl={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
            '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
            '0_0':{'segment':[[0,13]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
            '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
            '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}

        plot_data_container_experiment={}
        plot_data_container_model={}
        plot_raxr_container_experiment={}
        plot_raxr_container_model={}
        A_list_Fourier_synthesis=[]
        P_list_Fourier_synthesis=[]
        HKL_list_raxr=[[],[],[]]
        spectra_index=0
        for data_set in data:
            if data_set.x[0]<15:
                f=np.array([])
                h = data_set.extra_data['h']
                k = data_set.extra_data['k']
                l = data_set.x
                LB = data_set.extra_data['LB']
                dL = data_set.extra_data['dL']
                I=data_set.y
                eI=data_set.error
                #make dumy hkl and f to make the plot look smoother
                l_dumy=np.arange(l[0],l[-1]+0.1,0.1)
                N=len(l_dumy)
                h_dumy=np.array([h[0]]*N)
                k_dumy=np.array([k[0]]*N)
                LB_dumy=[]
                dL_dumy=[]
                f_dumy=[]

                for i in range(N):
                    key=None
                    if l_dumy[i]>=0:
                        key=str(int(h[0]))+'_'+str(int(k[0]))
                    else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                    for ii in bl_dl[key]['segment']:
                        if abs(l_dumy[i])>=ii[0] and abs(l_dumy[i])<ii[1]:
                            n=bl_dl[key]['segment'].index(ii)
                            LB_dumy.append(bl_dl[key]['info'][n][1])
                            dL_dumy.append(bl_dl[key]['info'][n][0])
                LB_dumy=np.array(LB_dumy)
                dL_dumy=np.array(dL_dumy)
                rough_dumy = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(l_dumy-LB_dumy)/dL_dumy)**2)**0.5
                if h_dumy[0]==0 and k_dumy[0]==0:
                    f_dumy = SCALES[0]*rough_dumy*sample.calc_f4_specular(h_dumy, k_dumy, l_dumy)
                else:
                    f_dumy = SCALES[0]*rough_dumy*sample.calc_f4(h_dumy, k_dumy, l_dumy)

                label=str(int(h[0]))+str(int(k[0]))+'L'
                plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
                plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis]),axis=1)
            else:#to be finished for plotting RAXR models here
                a=getattr(VARS['rgh_raxr'],'a'+str(spectra_index+1))
                b=getattr(VARS['rgh_raxr'],'b'+str(spectra_index+1))
                A_list,P_list=[],[]
                for index_resonant_el in range(len(RESONANT_EL_LIST)):
                    A_list_domain=0
                    P_list_domain=0
                    if RESONANT_EL_LIST[index_resonant_el]!=0:
                        A_list_domain=getattr(VARS['rgh_raxr'],'A_D'+str(index_resonant_el+1)+'_'+str(spectra_index+1))
                        P_list_domain=getattr(VARS['rgh_raxr'],'P_D'+str(index_resonant_el+1)+'_'+str(spectra_index+1))
                    A_list.append(A_list_domain)
                    P_list.append(P_list_domain)
                f=np.array([])
                h = data_set.extra_data['h']
                k = data_set.extra_data['k']
                x = data_set.x
                y = data_set.extra_data['Y']
                LB = data_set.extra_data['LB']
                dL = data_set.extra_data['dL']
                I=data_set.y
                eI=data_set.error
                A_list_Fourier_synthesis.append(A_list)
                P_list_Fourier_synthesis.append(P_list)
                HKL_list_raxr[0].append(h[0])
                HKL_list_raxr[1].append(k[0])
                HKL_list_raxr[2].append(y[0])
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                    f = SCALES[0]*rough*sample.calc_f4_specular_RAXR(h, k, y, x, E0, F1F2, a, b, A_list, P_list, RESONANT_EL_LIST)
                else:
                    f = SCALES[0]*rough*sample.calc_f4_nonspecular_RAXR(h, k, y, x, E0, F1F2, a, b, A_list, P_list, RESONANT_EL_LIST)
                label=str(int(h[0]))+'_'+str(int(k[0]))+'_'+str(y[0])
                plot_raxr_container_experiment[label]=np.concatenate((x[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
                plot_raxr_container_model[label]=np.concatenate((x[:,np.newaxis],f[:,np.newaxis]),axis=1)
                spectra_index+=1
        #dump CTR data and profiles
        hkls=['00L','02L','10L','11L','20L','22L','30L','2-1L','21L']
        plot_data_list=[]
        for hkl in hkls:
            plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
        pickle.dump(plot_data_list,open(os.path.join(output_file_path,"temp_plot"),"wb"))
        #dump raxr data and profiles
        pickle.dump([plot_raxr_container_experiment,plot_raxr_container_model],open(os.path.join(output_file_path,"temp_plot_raxr"),"wb"))
        #dump electron density profiles
        #e density based on model fitting
        sample.plot_electron_density(sample.domain,file_path=output_file_path,z_max=29)#dumpt file name is "temp_plot_eden"
        #e density based on Fourier synthesis
        z_plot,eden_plot,eden_domains=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_Fourier_synthesis).transpose(),np.array(A_list_Fourier_synthesis).transpose(),z_min=0.,z_max=29.,resonant_el=RAXR_EL,resolution=1000)
        pickle.dump([z_plot,eden_plot,eden_domains],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis"),"wb"))
    #you may play with the weighting rule by setting eg 2**bv, 5**bv for the wt factor, that way you are pushing the GenX to find a fit btween
    #good fit (low wt factor) and a reasonable fit (high wt factor)
    if COUNT_TIME:t_3=datetime.now()
    if COUNT_TIME:
        print "It took "+str(t_1-t_0)+" seconds to setup"
        print "It took "+str(t_2-t_1)+" seconds to calculate bv weighting"
        print "It took "+str(t_3-t_2)+" seconds to calculate structure factor"
    return F,1+WT_BV*bv,fom_scaler


#######################quick explanation for parameters in the main setup zone####################################
"""
running_mode(bool)
    if true then disable all the I/O function
SORBATE(list of list with each list item containing the sorbate element in each domain)
    element symbol for sorbate
    the shape of SORBATE is the same as pickup_index
BASAL_EL(a list of elements to specify the anchor reference to the ternary complex)
    only used in the domain containing ternary complex species
    The first item in each item list is alway None, since the first one is referenced to the substrate surface
    By default each item after the first one has a referenced element from the previous element)
pickup_index(a list of index list with items from the match index table above)
    representative of different binding configurations for different domains
    make sure the half layer indexes are in front of the full layer indexes
    In this new version, you can have multiple sites being assigned simultaneously on the same domain
    For example,in the case of [[0,6,6],[4],[10,14]] there are three sites assinged to domain1, i.e. bidentate site and the other two outer-sphere site
sym_site_index(a list of list of [0,1])
    a way to specify the symmetry site on each domain
    you may consider only site pairs in this version ([0,1])
    The shape is the same as pickup_index, except that the inner-most items are [0,1] instead of match index number
    It will be set up automatically
full_layer(a list of either 0 or 1 with 0 for short and 1 for long slab)
    used to specify the step for full layer termination, the items in this list must have a one to one corresponding to the items appearing in the pick_up_index for FL
half_layer(a list of either 2 or 3 with 2 for short and 3 for long slab)
    Analogous to full_layer but used for half layer termination case
full_layer_pick(a list of value of either None, or 0 or 1)
    used to specify the full layer type, which could be either long slab (1) or short slab (0)
    don't forget to set None for the half layer termination domain
    Again Nones if any must be in front of numbers (Half layer domains in front of full layer domains)
    concerns about None has been automatically setup in this new version
half_layer_pick(a list of value of either None, or 2 or 3)
    Analogous to full_layer_pick but used for half layer termination
OS_X(Y,Z)_REF(a list of None,or any number)
    set the reference coordinate xyz value for the outer-sphere configuration, which could be on either HL or FL domain
    these values are fractional coordinates of sorbates
    if N/A then set it to None
    such setting is based on the symmetry operation intrinsic for the hematite rcut surface, which have the following relationship
    x1+x2=0.5/1.5, y1-y2=0.5 or -0.5, z1=z2
    The shape is like [[],[]], each item corresponds to different domains
    The number of items within each domain is twice (considering symmetry site pair) the number of sorbate for that domain
DOMAIN_GP(a list of list of domain indexs)
    use this to group two domains with same surface termination (HL or FL) together
    the associated atom groups for both surface atoms and sorbates will be created (refer to manual)
    This feature is not necessary and so not supported anymore in this version.
water_pars(a lib to set the interfacial waters quickly)
    This water molecules are regarded as adsorbed water molecules with lateral and vertical ordering which will have effect on both the specular and offspecular rods
    you may use default which has no water or turn the switch off and set the number and anchor points
layered_water_pars(a lib to set layered water structure)
    layered water structure factor only have effect on the specular rod
    Based on the equation(29) in Reviews in Mineralogy and Geochemistry v. 49 no. 1 p. 149-221
    key of 'yes_OR_no':a list of 0 or 1 to specify whether or not considering the layered water structure
    key of 'ref_layer_height' is a list of atom ids (domain information not needed) to specify the reference height for the layered water heights
layered_sorbate_pars(a lib to set layered sorbate structure)
    pretty much the same as layered_water_pars
    key of 'el' is the symbol for the resonant element
USE_BV(bool)
    a switch to apply bond valence constrain during surface modelling
TABLE_DOMAINS(list of 0 or 1, the length should be higher than the total domain number)
    specify whether or not generate the associated pars for each domain
    [0,1,1] means only generate the pars for last two domains
RAXR_EL(resonant element)
NUMBER_SPECTRA(number of RAXR spectras)
    Note each spectra, there will be an independent set of fitting parameters (a,b,A,P)
RESONANT_EL_LIST(a list of integer number (either 1 or 0))
    Used to specify the domain containing resonant element
    0 means no resonant element on the domain
    1 means considering resonant element on the domain
E0=13000
    Center of Scan energy range for RAXR data
F1F2_FILE="Pb.f1f2"
    Absolute file path for the f1f2 file containing anomalous correction items at each energy
F1F2=None
    Global variable to hold the f1f2 values after loading the f1f2 file
COVALENT_HYDROGEN_RANDOM(bool)
    a switch to not explicitly specify the protonation of surface functional groups
    different protonation scheme (0,1 or 2 protons) will be tried and compared, the one with best bv result will be used
BV_OFFSET_SORBATE(a list of number)
    it is used to define the acceptable range of bond valence sum for sorbates
    [bv_eachbond*N_bonds-offset,bv_eachbond*N_bonds] will be the range
    set a random number for a clean surface (no sorbate), but don't miss that
SEARCH_RANGE_OFFSET(a number)
    used to set the searching range for an atom, which will be used to calculate the bond valence sum of sorbates
    the radius of the searching sphere will be the ideal bond length plus this offset
commands(a list of str to be executed inside sim function)
    eg. ['gp_O1O2_O7O8_D1.setoc(gp_Fe4Fe6_Fe10Fe12_D1.getoc())']
    used to expand the funtionality of grouping or setting something important
USE_COORS(a list of [0,0] or [1,1] with two items for two symmetry sites)
    you may want to add sorbates by specifying the coordinates or having the program calculate the position from the geometry setting you considered
    eg1 USE_COORS=[[0,0]]*len(pickup_index) not use coors for all domains
    eg2 USE_COORS=[[1,1]]*len(pickup_index) use coors for all domains
    eg3 USE_COORS=[[0,0],[1,1],[1,1]] use coors for only domain2 and domain3
COORS(a lib specifying the coordinates for sorbates)
    keys of COORS are the domain index and site index, ignore domain with no sorbates
    len(COORS[(i,j)]['sorbate'])=1 while len(COORS[(i,j)]['oxygen'])>=1, which is the number of distal oxygens
    make sure the setup matches with the pick_up index and the sym_site_index as well as the number of distal oxygens
    if you dont consider oxygen in your model, you still need to specify the coordinates for the oxygen(just one oxygen) to avoid error prompt
O_NUMBER_HL/FL(a list of list of [a,b],where a and b are integer numbers)
    one to one corresponding for the number of distal oxygens, which depend on local structure and binding configuration
    either zero oxygen ligand or enough ligands to complete coordinative shell
O_NUMBER_HL/FL_EXTRA(used to define the distal oxygen number for a surface species binding to the distal oxygen of basal element)
MIRROR(a list of true or false)
    Used to specify the way you add a distal oxygen to a surface complex with monodentate or bidentate binding configuration
    Or in a case of tridentate binding mode with octahedral local structure
SORBATE_NUMBER_HL/FL(a list of list of [a], a can be either 1 or 2 or 0 for clean surface)
    If considering two symmetry sites, then a=2
    If considering one site (distribute the two on two different domains), then a=1
    If considering clean surface, then a=0
SORBATE_NUMBER_HL/FL_EXTRA(used to specify the number of outer-part of ternary complex species)
COUNT_DISTAL_OXYGEN(bool)
    True then consider bond valence also for distal oxygen,otherwise skip the bv contribution from distal oxygen
ADD_DISTAL_LIGAND_WILD(list of bool)
    the distal oxygen could be added by specifying the pars for the spherical coordinate system (r, theta, phi), which is called wild here, or be added
    in a specific geometry setting for a local structure (like tetrahedra)
    you can specify different case for different domains
    and this par is not applicable to outersphere mode, which should be set to None for that domain
DOMAINS_BV(a list of integer numbers)
    Domains being considered for bond valence constrain, counted from 0
BOND_VALENCE_WAIVER(a list of oxygen atom ids [either surface atoms or distals] with domain tag)
    When each two of thoes atoms in the list are being considered for bond valence, the valence effect will be ignored no matter how close they are
    Be careful to select atoms as bond valence waiver
GROUPING_SCHEMES(a list of lists with two items, with each item being the domain index starting from 0)
    Define how you want to group the surface atoms together from two different domains with same termination type
    A function will generate all the associated commands to do the grouping
    [[0,1]] means group surface atoms from the first (0) and second domain(1)
    If you dont want to do any grouping, set this to be []
GROUPING_DEPTH(a list of integers less than 10)
    Define how deep you want to group your atoms. You can define a maximum grouping depth to 10
    You should count the atom layers upward from the 10th atom layer
    [6,10] means you want to group the 5th atom layer to 10th atom layer for domain 1 and group all top ten atom layers together for domain2
    Don't forget that you have a Iron layer which is explicitly included in HL but the occ set to 0 to account for the missing Fe sites
    So you should count that atom layer too when considering the grouping depth
"""
#############################################code update logs############################################################################
"""
##version 1##
consider sorbate (Pb and Sb) of any combination
"""
#########################the order of building up the interfacial structure#############################
##surface atoms-->hydrogen atoms for surface oxygen atoms-->sorbate sets (metal and distal oxygens)-->hydrogen for distal oxygens-->water and the associated hydrogen atoms###
#########################naming rules###################
"""
    ###########atm ids################
    sorbate: Pb1_D1A, Pb2_D2B, HO1_Pb1_D1A,HO2_Pb1_D1A, HO1_Sb2_D1A
    water: Os1_D1A
    surf atms: O1_1_0_D1A (half layer), O1_1_1_D1A(atoms of second slab)
    hydrogen atoms:
        HB1_O1_1_0_D1A, HB2_O1_1_0_D1A(doubly protonated surface oxygen)-->r[phi,theta]_H_1_1, r[phi,theta]_H_1_2 (first number in the tag is the index of surface oxygen, and the second the index of hydrogen atom)
        HB1_HO1_Pb1_D1A, HB2_HO1_Pb1_D1A(doubly protonated distal oxygen)-->r[phi,theta]_H_D_1_1, r[phi,theta]_H_D_1_2 (first number in the tag is the index of distal oxygen, and the second the index of hydrogen atom)
        HB1_Os1_D1A, HB2_Os1_D1A(doubly protonated water oxygen)-->r[phi,theta]_H_W_1_1_1, r[phi,theta]_H_W_1_1_2 (first number in the tag is the index of water set (single or paired), and the second the index of water in each set (at most 2 for water pair), and the last one for index of hydrogen atom)
    ############group names###########
    gp_sorbates_set1_D1
        group first set (what “set1” means, the set index has an increment of 1) of sorbates (including metals and distal oxygens) together without considering symmetry relationship (used to set equal occupancy for metal and its distal oxygens).
    gp_Pb_set1_D1(group two symmetry related Pb atoms together (4 in total if considering those for the symmetry related domains))
        Pb can be replaced with another other element symbol
        set1 means first set consisting of two symmetry related atom within each domain
        you can have multiple sets if you consider multiple sites being occupied simultaneously
        note that the adjacent set indexes are 2 apart, so it goes from set1 to set3 to set5 and so on
    gp_HO1_set1_D1(group two symmetry related distal oxygen atoms together (4 in total if considering those for the symmetry related domains))
        the set index is the same as that described above for the sorbate
        the number after HO specify the distal oxygen, so if there are 3 distal oxygens coordinated with the sorbate, then we use _HO1_, _HO2_ and _HO3_ to distinguish those
    gp_HO_set1_D1
        group all the distal oxygens for the first sorbate (like Pb1, what set1 means here).
        So the set index is different from those defined above (increment of 2) in that it starts from 1 and with an increment of 1.
        Corresponding to gp_sorbates_set1_D1, it is used to set equal u or oc for the distal oxygens associating with the symmetry related sorbate from two different domain (domainA and domainB).
    gp_waters_set1_D1(discrete grouping for each set of water at same layer, group u, oc and dz)
    gp_O1O7_D1(discrete grouping for surface atms, group dx dy in symmetry)
    gp_O1O2_O7O8_D1(sequence grouping for u, oc, dy, dz, or dx in an equal opposite way for O1O2 and O7O8)
    gp_O1O2_O7O8_D1_D2(same as gp_O1O2_O7O8_D1, but group each set of atoms from two different domains, you need to set DOMAIN_GP to have it work)

    some print examples
    #print domain_creator.extract_coor(domain1A,'O1_1_0_D1A')
    #print domain_creator.extract_coor(domain1B,'Pb1_D1B')
    #print_data(N_sorbate=4,N_atm=40,domain=domain1A,z_shift=1,save_file='D://model.xyz')
    #print domain_class_1.cal_bond_valence1(domain1A,'Pb1_D1A',3,False)
"""
###########explanation of some global variables###################
#pars for interfacial waters##
"""
    WATER_NUMBER: must be even number considering 2 atoms each layer
    V_SHIFT: vertical shiftment of water molecules, in unit of angstroms,two items each if consider two water pair
    R=: half distance bw two waters at each layer in unit of angstroms
    ALPHA: alpha angle used to cal the pos of water molecule

    DISCONNECT_BV_CONTRIBUTION=[{('O1_1_0','O1_2_0'):'Pb2'},{}]
    ##if you have two sorbate within the same unit cell, two sorbates will be the coordinative ligands of anchor atoms
    ##but in fact the sorbate cannot occupy the adjacent sites due to steric constrain, so you should delete one ligand in the case of average structure
    ##However, if you consider multiple domains with each domain having only one sorbate, then set the items to be {}
    ##in this case, bv contribution from Pb2 won't be account for both O1 and O2 atom.

    ANCHOR_REFERENCE=[[None],[None]]
    ANCHOR_REFERENCE_OFFSET=[[None],[None]]
    #we use anchor reference to set up a binding configuration in a more intelligible way. We only specify the anchor ref when the two anchor points are
    #not on the same level. The anchor reference will be the center(Fe atom) of the octahedral coordinated by ligands including those two anchors.
    #phi=0 will means that the sorbate is binded in a most feasible way.
    #To ensure the bending on two symmetry site are towards right position, the sorbate attach atom may have reversed order.
    #eg. [O1,O3] correspond to [O4px,O2] rather than [O2,O4px].

    COHERENCE
    #now the coherence looks like [{True:[0,1]},{False:[2,3]}] which means adding up first two domains coherently
    #and last two domains in-coherently. After calculation of structure factor for each item of the list, absolute
    #value of SF will be calculated followed by being summed up
    #so [{True:[0,1]},{True:[2,3]}] is different from [{True:[0,1,2,3]}]

    #IF the covalent_hydrogen_random set to True, then we wont specifically define the number of covalent hydrogen. And it will try [0,1,2] covalent hydrogens
    COVALENT_HYDROGEN_RANDOM=True
    POTENTIAL_COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]

    If covalent_hydrogen_random is set to false then explicitly define the number of covalent hydrogen here

    ADD_DISTAL_LIGAND_WILD=True means adding distal oxygen ligand in a spherical coordinated system with specified r, theta and phi. Otherwise the distal oxygen are added based on the return value of geometry module

    COVALENT_HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0'],['O1_1_0','O1_2_0']]
    COVALENT_HYDROGEN_NUMBER=[[1,1],[1,1]]
    ##means in domain1 both O1 and O2 will accept one covalent hydrogen (bv contribution of 0.8)

    HYDROGEN_ACCEPTOR=[['O1_1_0','O1_2_0','O1_3_0','O1_4_0'],['O1_1_0','O1_2_0']]
    HYDROGEN_NUMBER=[[1,1,1,1],[1,1]]
    ##means in domain1 O1 to O4 will accept one covalent hygrogen (bv contribution of 0.2)
"""

##########################some examples to set up sorbates binding under different configurations#######################
"""
SORBATE=["Pb","Sb"]
eg1.SORBATE_NUMBER=[[1,0],[0,1]]#two domains:one has one Pb sorbate and the other one has one Sb sorbate
eg2.SORBATE_NUMBER=[[0,0],[0,1]]#two domains:first one is clean surface, the other one has one Sb sorbate
eg3.SORBATE_NUMBER=[[1,1],[1,1]]#two domains and each domain has one Pb and one Sb sorbate
eg4.SORBATE_NUMBER=[[2,0],[0,1]]#first domain has two Pb sorbate

#len(O_NUMBER)= # of domains
#len(O_NUMBER[i])= # of sorbates
O_NUMBER=[[[1],[0]],[[0],[0]]]#Pb sorbate has one oxygen ligand in domain1, while no oxygen ligands in domain2
O_NUMBER=[[[1,2],[0]],[[0],[3,5]]]#1st Pb sorbate has one oxygen ligand and 2nd Pb has two oxygen ligands in domain1, while in domain2 1st Sb has 3 oxygen ligands and 2nd has 5 oxygen ligands
SORBATE_LIST=create_sorbate_el_list(SORBATE,SORBATE_NUMBER)
BV_SUM=[[1.33,5],[1.33,5.]]#pseudo bond valence sum for sorbate

#len(SORBATE_ATTACH_ATOM)=# of domains
#len(SORBATE_ATTACH_ATOM[i])=# of sorbates in domaini
SORBATE_ATTACH_ATOM=[[['O1_1_0','O1_2_0']],[['O1_1_0','O1_2_0','O1_3_0']]]
SORBATE_ATTACH_ATOM_OFFSET=[[[None,None]],[[None,None,None]]]
TOP_ANGLE=[[1.38],[1.38]]
PHI=[[0],[0]]
R_S=[[1],[1]]
MIRROR=False
"""
##########################offset symbols######################
"""
None: no translation
'+x':translate along positive x axis by 1 unit
'-x':translate along negative x axis by 1 unit
SAME deal for translation along y axis
"""

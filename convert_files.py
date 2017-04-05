import numpy as np
import os

def convert_best_pars_to_matlab_input_file(file_name=None,domain=None,layered_water=None):
    data=domain._extract_values()
    z=np.array(data[2])*20.1058*np.sin((180-95.787)/180.*np.pi)
    #z=np.array(data[2])*20.1058*np.sin((180-90)/180.*np.pi)
    z_new=list(z-z[0])#first atom is mica surface, now mica surface has a z value of 0
    f=open(file_name,'w')
    f.write('% Param No. Best-Fit     Opt(1,2)             FIT    Initial      Std Dev\n\n')
    index_gaussian_peaks=[np.where(domain.id==each_id)[0][0] for each_id in domain.id if 'Gaussian_' in each_id]
    z_gaussian=[z_new[each_index] for each_index in index_gaussian_peaks]

    #for each_index in index_gaussian_peaks:
    u_gaussian=[data[4][each_index] for each_index in index_gaussian_peaks]
    oc_gaussian=[data[5][each_index]/4.0/2.0 for each_index in index_gaussian_peaks]#occ normalized to the half surface unit cell

    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(1   ,    1.105  ,   0.0010  ,   1.0000  ,   0   ,  1.000000   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(2   ,    1.05  ,   0.0010  ,   0.1  ,   0   ,  1.000000   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(3   ,    10.00  ,   0.0010  ,   5  ,   0   ,  1.000000   ,  0.00))
    #water layered
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(4   ,    layered_water.getFirst_layer_height_w()+1  ,   0.0010  ,   1.0000  ,   0   ,  layered_water.getFirst_layer_height_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(5   ,    layered_water.getU0_w()+1  ,   0.0010  ,   1.0000  ,   0   ,  layered_water.getU0_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(6   ,    layered_water.getD_w()+1  ,   0.0010  ,   1.0000  ,   0   ,  layered_water.getD_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(7   ,    layered_water.getUbar_w()+1  ,   0.0010  ,   1.0000  ,   0   ,  layered_water.getUbar_w()+1   ,  0.00))
    #adsorbed water
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(8   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(9   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(10   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))

    for i in range(3):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(11+i*3   ,    z_gaussian[i]+1  ,   0.0010  ,   0.5  ,   0   ,  z_gaussian[i]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(12+i*3   ,    oc_gaussian[i]+1  ,   0.0010  ,   0.5  ,   0   ,  oc_gaussian[i]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(13+i*3   ,    u_gaussian[i]+1  ,   0.0010  ,   0.5  ,   0   ,  u_gaussian[i]+1   ,  0.00))
    for i in range(26):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(20+i   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
    for i in range(len(z_gaussian)-3):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(46+i*3   ,    z_gaussian[i+3]+1  ,   0.0010  ,   0.5  ,   0   ,  z_gaussian[i+3]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(47+i*3   ,    oc_gaussian[i+3]+1  ,   0.0010  ,   0.5  ,   0   ,  oc_gaussian[i+3]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(48+i*3   ,    u_gaussian[i+3]+1  ,   0.0010  ,   0.5  ,   0   ,  u_gaussian[i+3]+1   ,  0.00))
    f.close()

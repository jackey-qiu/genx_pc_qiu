import numpy as np
from numpy.matlib import repmat
from numpy.linalg import pinv
from matplotlib import pyplot
from scipy import misc
import fnmatch
import os
import matplotlib.patches as patches
import ctr_data

##The background subtraction algoritem is developped by Vincent Mazet with the copyright notice as below###
##The code was originally written by Vincent Mazet based on MATLAB. Canrong Qiu (me) translated the scripts to Python language##
##Correction factors are calculated using TDL modules, developped and maintained by GSECARS 13IDC beamline at APS (Peter Eng and Joanne Stubbs are responsible persons)##

'''
Copyright (c) 2012, Vincent Mazet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

"""
# BACKCOR   Background estimation by minimizing a non-quadratic cost function.
#
#   [EST,COEFS,IT] = BACKCOR(N,Y,ord_cusER,THRESHOLD,FUNCTION) computes and estimation EST
#   of the background (aka. baseline) in a spectroscopic signal Y with wavelength N.
#   The background is estimated by a polynomial with ord_cuser ord_cusER using a cost-function
#   FUNCTION with parameter THRESHOLD. FUNCTION can have the four following values:
#       'sh'  - symmetric Huber function :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                  { 2*THRESHOLD*abs(x)-THRESHOLD^2  otherwise.
#       'ah'  - asymmetric Huber function :  f(x) = { x^2  if x < THRESHOLD,
#                                                   { 2*THRESHOLD*x-THRESHOLD^2  otherwise.
#       'stq' - symmetric truncated quadratic :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                       { THRESHOLD^2  otherwise.
#       'atq' - asymmetric truncated quadratic :  f(x) = { x^2  if x < THRESHOLD,
#                                                        { THRESHOLD^2  otherwise.
#   COEFS returns the ord_cusER+1 vector of the estimated polynomial coefficients.
#   IT returns the number of iterations.
#
#   [EST,COEFS,IT] = BACKCOR(N,Y) does the same, but run a graphical user interface
#   to help setting ord_cusER, THRESHOLD and FCT.
#
# For more informations, see:
# - V. Mazet, C. Carteret, D. Brie, J. Idier, B. Humbert. Chemom. Intell. Lab. Syst. 76 (2), 2005.
# - V. Mazet, D. Brie, J. Idier. Proceedings of EUSIPCO, pp. 305-308, 2004.
# - V. Mazet. PhD Thesis, University Henri Poincare Nancy 1, 2005.
#
# 22-June-2004, Revised 19-June-2006, Revised 30-April-2010,
# Revised 12-November-2012 (thanks E.H.M. Ferreira!)
# Comments and questions to: vincent.mazet@unistra.fr.

# Check arguments
if nargin < 2, error('backcor:NotEnoughInputArguments','Not enough input arguments'); end;
if nargin < 5, [z,a,it,ord_cus,s,fct] = backcorgui(n,y); return; end; # delete this line if you do not need GUI
if ~isequal(fct,'sh') && ~isequal(fct,'ah') && ~isequal(fct,'stq') && ~isequal(fct,'atq'),
    error('backcor:UnknownFunction','Unknown function.');
end;
"""

#global variables
PLOT_LIVE=True
BRAGG_PEAKS=range(0,19)#L values of Bragg peaks
BRAGG_PEAK_CUTOFF=0.04#excluding range on L close to a Bragg peak (L_Bragg+-this value will be excluded for plotting)
###########integration setup here##############
INTEG_PARS={}
INTEG_PARS['cutoff_scale']=0.001
INTEG_PARS['use_scale']=False#Set this to False always
INTEG_PARS['center_pix']=[53,153]#Center pixel index (know Python is column basis, so you need to swab the order of what you see at pixe image)
INTEG_PARS['r_width']=20#integration window in row direction (total row length is twice that value)
INTEG_PARS['c_width']=50#integration window in column direction (total column length is twice that value)
INTEG_PARS['integration_direction']='y'#integration direction (x-->row direction, y-->column direction), you should use 'y' for horizontal mode (Bragg peak move left to right), and 'x' for vertical mode (Bragg peak move up and down)
INTEG_PARS['ord_cus_s']=[1,2,4,6] #A list of integration power to be tested for finding the best background subtraction. Flat if the value is 0. More wavy higher value
INTEG_PARS['ss']=[0.01,0.05,0.1]#a list of thereshold factors used in cost function (0: all signals are through, means no noise background;1:means all backround, no peak signal. You should choose a value between 0 and 1)
INTEG_PARS['fct']='ah'#Type of cost function ('sh','ah','stq' or 'atq')
################################################
#############spec file info here################
GENERAL_LABELS={'H':'H','K':'K','L':'L','E':'Energy'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
CORRECTION_LABELS={'time':'Seconds','norm':'io','transmision':'transm'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
ANGLE_LABELS={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
ANGLE_LABELS_ESCAN={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
#G label positions (n_azt: azimuthal reference vector positions @3rd to 6th numbers counting from left to right at G0 line)
#so are the other symbols: cell (lattice cell info), or0 (first orientation matrix), or1 (second orientation matrix), lambda (x ray wavelength)
G_LABELS={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}
IMG_EXTENTION='.tif'#image extention (.tif or .tiff)
CORR_PARAMS={'scale':1000000,'geom':'psic','beam_slits':{'horz':0.06,'vert': 1},'det_slits':None,'sample':{'dia':10,'polygon':[],'angles':[]}}#slits are in mm
###############################################################################################################################################################
class data_integration:
    def __init__(self,spec_path='M:\\fwog\\members\\qiu05\\1704_APS_13IDC\\mica',spec_name='s5_100mM_NH4Cl_Zr_1.spec',scan_number=[11],\
                corr_params=CORR_PARAMS,
                integ_pars=INTEG_PARS,\
                general_labels=GENERAL_LABELS,\
                correction_labels=CORRECTION_LABELS,\
                angle_labels=ANGLE_LABELS,\
                angle_labels_escan=ANGLE_LABELS_ESCAN,\
                G_labels=G_LABELS,\
                img_extention=IMG_EXTENTION):

        self.spec_path=spec_path
        self.spec_name=spec_name
        self.scan_number=scan_number
        self.data_info={}
        self.corr_params=corr_params
        self.integ_pars=integ_pars
        self.general_labels=general_labels
        self.correction_labels=correction_labels
        self.angle_labels=angle_labels
        self.angle_labels_escan=angle_labels_escan
        self.G_labels=G_labels
        self.img_extention=img_extention
        self.combine_spec_image_info()
        self.batch_image_integration()

    def set_spec_info(self,path=None,name=None,scan_number=None):
        if path!=None:
            self.spec_path=path
        if name!=None:
            self.spec_name=name
        if scan_number!=None:
            self.scan_number=scan_number
        if [path,name,scan_number]!=[None,None,None]:
            self.combine_spec_image_info()
            self.batch_image_integration()
        return None

    def set_corr_params(self,corr_params):
        self.corr_params=corr_params
        self.combine_spec_image_info()
        self.batch_image_integration()
        return None

    def set_integ_pars(self,integ_pars={'ord_cus':4,'s':0.1,'fct':'sh'}):
        self.integ_pars=integ_pars
        self.combine_spec_image_info()
        self.batch_image_integration()
        return None

    #engine function to subtraction background
    def backcor(self,n,y,ord_cus,s,fct):
        # Rescaling
        N = len(n)
        index = np.argsort(n)
        n=np.array([n[i] for i in index])
        y=np.array([y[i] for i in index])
        maxy = max(y)
        dely = (maxy-min(y))/2.
        n = 2. * (n-n[N-1]) / float(n[N-1]-n[0]) + 1.
        n=n[:,np.newaxis]
        y = (y-maxy)/dely + 1

        # Vandermonde matrix
        p = np.array(range(ord_cus+1))[np.newaxis,:]
        T = repmat(n,1,ord_cus+1) ** repmat(p,N,1)
        Tinv = pinv(np.transpose(T).dot(T)).dot(np.transpose(T))

        # Initialisation (least-squares estimation)
        a = Tinv.dot(y)
        z = T.dot(a)

        # Other variables
        alpha = 0.99 * 1/2     # Scale parameter alpha
        it = 0                 # Iteration number
        zp = np.ones((N,1))         # Previous estimation

        # LEGEND
        while np.sum((z-zp)**2)/np.sum(zp**2) > 1e-10:

            it = it + 1        # Iteration number
            zp = z             # Previous estimation
            res = y - z        # Residual

            # Estimate d
            if fct=='sh':
                d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
            elif fct=='ah':
                d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
            elif fct=='stq':
                d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
            elif fct=='atq':
                d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
            else:
                pass

            # Estimate z
            a = Tinv.dot(y+d)   # Polynomial coefficients a
            z = T.dot(a)            # Polynomial

        z=np.array([(z[list(index).index(i)]-1)*dely+maxy for i in range(len(index))])

        return z,a,it,ord_cus,s,fct

    def _get_col_from_file(self,lines,start_row,end_row,col,type=float):
        numbers=[]
        for i in range(start_row,end_row):
            numbers.append(type(lines[i].rstrip().rsplit()[col]))
        return numbers

    #extract info from spec file
    def sort_spec_file(self,spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,17,19],\
                    general_labels={'H':'H','K':'K','L':'L','E':'Energy'},correction_labels={'time':'Seconds','norm':'io','transmision':'transm'},\
                    angle_labels={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'},\
                    angle_labels_escan={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'},\
                    G_labels={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}):
        matches = []
        data_info,col_label={},{}
        data_info['scan_type']=[]
        data_info['scan_number']=scan_number
        data_info['row_number_range']=[]
        data_info['spec_path']=os.path.join(spec_path,spec_name)

        for key in general_labels.keys():
            data_info[key]=[]

        for key in correction_labels.keys():
            data_info[key]=[]

        for key in angle_labels.keys():
            data_info[key]=[]

        for key in G_labels.keys():
            data_info[key]=[]

        f_spec=open(os.path.join(spec_path,spec_name))
        spec_lines=f_spec.readlines()
        scan_rows=[]
        data_rows=[]
        G0_rows=[]
        G1_rows=[]
        G3_rows=[]
        G4_rows=[]
        for i in range(len(spec_lines)):
            if spec_lines[i].startswith("#S"):
                scan_rows.append([i,int(spec_lines[i].rsplit()[1])])
            elif spec_lines[i].startswith("#L"):
                data_rows.append(i+1)
            elif spec_lines[i].startswith("#G0"):
                G0_rows.append(i)
            elif spec_lines[i].startswith("#G1"):
                G1_rows.append(i)
            elif spec_lines[i].startswith("#G3"):
                G3_rows.append(i)
            elif spec_lines[i].startswith("#G4"):
                G4_rows.append(i)

        if scan_number==[]:
            for i in range(len(scan_rows)):
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                j=0
                while not spec_lines[data_start+j].startswith("#"):
                    j+=1
                row_number_range=[data_start,data_start+j]
                data_info['scan_type'].append(scan_type_temp)
                data_info['scan_number'].append(scan_number_temp)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])

                for key in correction_labels.keys():
                    data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))

                for key in angle_labels.keys():
                    if scan_type_temp=='rodscan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                    if scan_type_temp=='Escan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

                for key in G_labels.keys():
                    G_type=G_labels[key][0]
                    inxes=G_labels[key][1]
                    #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                    ff=lambda items,inxes:[float(items[i]) for i in indxes]
                    if G_type=='G0':
                        data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G1':
                        data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G3':
                        data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G4':
                        data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                if scan_type_temp in col_label.keys():
                    pass
                else:
                    col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()[1:]
        else:
            for ii in range(len(scan_number)):
                _scan=scan_number[ii]
                i=np.where(np.array(scan_rows)[:,1]==_scan)[0][0]
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                j=0
                while not spec_lines[data_start+j].startswith("#"):
                    j+=1
                row_number_range=[data_start,data_start+j]
                data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(scan)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])

                for key in correction_labels.keys():
                    data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))

                for key in angle_labels.keys():
                    if scan_type_temp=='rodscan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                    if scan_type_temp=='Escan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

                for key in G_labels.keys():
                    G_type=G_labels[key][0]
                    inxes=G_labels[key][1]
                    #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                    ff=lambda items,inxes:[float(items[i]) for i in inxes]
                    if G_type=='G0':
                        data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G1':
                        data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G3':
                        data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G4':
                        data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                #data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(_scan)
                #data_info['row_number_range'].append(row_number_range)
                if scan_type_temp in col_label.keys():
                    pass
                else:
                    col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()
            data_info['col_label']=col_label
            #print data_info['scan_number']
            f_spec.close()
        return data_info

    #build images path based on scan number and info from spec file
    def match_images(self,data_info,img_extention='.tiff'):
        data_info=data_info
        spec_name=os.path.basename(os.path.normpath(data_info['spec_path'])).replace(".spec","")
        image_head=os.path.join(os.path.dirname(data_info['spec_path']),"images")
        image_head=os.path.join(image_head,spec_name)
        data_info["images_path"]=[]
        def _number_to_string(place=4,number=1):
            i=0
            #print place-i
            if number==0:
                return '0'*place
            else:
                while int(number/(10**(place-i)))==0:
                    i+=1
                return '0'*(i-1)+str(number)

        for i in range(len(data_info["scan_number"])):
            scan_temp=data_info["scan_number"][i]
            scan_number_str='S'+_number_to_string(3,scan_temp)
            range_data_temp=data_info["row_number_range"][i]
            temp_img_container=[]
            for j in range(range_data_temp[1]-range_data_temp[0]):
                img_number=_number_to_string(5,j)+img_extention
                temp_img_container.append(os.path.join(os.path.join(image_head,scan_number_str),"_".join([spec_name,scan_number_str,img_number])))
            data_info["images_path"].append(temp_img_container)

        return data_info

    def combine_spec_image_info(self):
        data_info=self.sort_spec_file(spec_path=self.spec_path,spec_name=self.spec_name,scan_number=self.scan_number,general_labels=self.general_labels,angle_labels=self.angle_labels,angle_labels_escan=self.angle_labels_escan,G_labels=self.G_labels)
        data_info=self.match_images(data_info,self.img_extention)
        self.data_info=data_info
        return None

    #If you want to twick through data points (If necessary run this after finishing one run of auto integration)
    def integrate_images_twick_mode(self,scan_number=[]):
        for scan in scan_number:
            scan_index=self.data_info['scan_number'].index(scan)
            scan_images=self.data_info['images_path'][scan_index]
            scan_check=raw_input('Doing scan'+str(scan)+'now! Continue? y or n ')
            if scan_check=='y':
                for image in scan_images:
                    center_pix=INTEG_PARS['center_pix']
                    r_width=INTEG_PARS['r_width']
                    c_width=INTEG_PARS['c_width']
                    image_index=scan_images.index(image)
                    image_check=raw_input('Doing image '+str(image_index+1)+'now! Continue? y or n ')
                    if image_check=='y':
                        k,kk,kkk=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                        I,Ibrg,Ier=0,0,0
                        while 1:
                            input_items=raw_input('Move window? using w(UP), s(DOWN), a(LEFT), d(RIGHT)\nChange integration width? using ci(column increasing), cd (column decreasing), \nri(row increasing),rd(row decrasing)\n')
                            if input_items.startswith('w'):
                                value=int(input_items.rsplit('w')[-1])
                                center_pix[0]=center_pix[0]-value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('s'):
                                value=int(input_items.rsplit('s')[-1])
                                center_pix[0]=center_pix[0]+value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('a'):
                                value=int(input_items.rsplit('a')[-1])
                                center_pix[1]=center_pix[1]-value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('d'):
                                value=int(input_items.rsplit('d')[-1])
                                center_pix[1]=center_pix[1]+value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('ci'):
                                value=int(input_items.rsplit('i')[-1])
                                c_width=c_width+value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('cd'):
                                value=int(input_items.rsplit('d')[-1])
                                c_width=c_width-value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('ri'):
                                value=int(input_items.rsplit('i')[-1])
                                r_width=r_width+value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items.startswith('rd'):
                                value=int(input_items.rsplit('d')[-1])
                                r_width=r_width-value
                                pyplot.close()
                                I,Ibrg,Ier=self.integrate_one_image_for_tick(image,center_pix,r_width,c_width,PLOT_LIVE)
                            elif input_items=='q':
                                break
                        input_outside=raw_input('Keep the current value? y or n ')
                        if input_outside=='y' and [I,Ibrg,Ier]!=[0,0,0]:
                            self.data_info['I'][scan_index][image_index]=I
                            self.data_info['Ierr'][scan_index][image_index]=Ier
                            self.data_info['Ibgr'][scan_index][image_index]=Ibrg
                            scan_dict=self._formate_scan_from_data_info(self.data_info,scan,image_index,I,Ier,Ibrg)
                            result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
                            self.data_info['F'][scan_index][image_index]=result_dict['F']
                            self.data_info['Ferr'][scan_index][image_index]=result_dict['Ferr']
                        else:
                            pass
                    else:
                        pass
            else:
                pass
        return None

    def integrate_one_image(self,img_path="S3_Zr_100mM_KCl_3_S136_0000.tiff",plot_live=PLOT_LIVE):
        cutoff_scale=INTEG_PARS['cutoff_scale']
        use_scale=INTEG_PARS['use_scale']
        center_pix=INTEG_PARS['center_pix']
        r_width=INTEG_PARS['r_width']
        c_width=INTEG_PARS['c_width']
        integration_direction=INTEG_PARS['integration_direction']
        ord_cus_s=INTEG_PARS['ord_cus_s']
        ss=INTEG_PARS['ss']
        fct=INTEG_PARS['fct']
        img=misc.imread(img_path)
        #center_pix= list(np.where(img==np.max(img[center_pix[0]-20:center_pix[0]+20,center_pix[1]-20:center_pix[1]+20])))
        if use_scale:
            if cutoff_scale<1:
                cutoff=np.max(img)*cutoff_scale
            else:
                cutoff=cutoff_scale
            index_cutoff=np.argwhere(img>=cutoff)
        else:
            index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        x_span,y_span=x_max-x_min,y_max-y_min

        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        index=None
        peak_width=10
        if INTEG_PARS['integration_direction']=='y':
            peak_width==INTEG_PARS['c_width']/5
        elif INTEG_PARS['integration_direction']=='x':
            peak_width==INTEG_PARS['r_width']/5
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[rt:-1]-z[rt:-1]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            return sum_temp/(len(y)-peak_width*2)*len(y)

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = self.backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[index]-z[index]))
                Ibgr_container.append(abs(np.sum(z[index])))
                FOM_container.append(_cal_FOM(y,z,peak_width))
                Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                z_container.append(z)
        index_best=FOM_container.index(min(FOM_container))
        index = np.argsort(n)
        if plot_live:
            z=z_container[index_best]
            fig,ax=pyplot.subplots()
            ax.imshow(img)
            rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            pyplot.figure()
            pyplot.plot(n[index],y[index],color='blue',label="data")
            pyplot.plot(n[index],z[index],color="red",label="background")
            pyplot.plot(n[index],y[index]-z[index],color="m",label="data-background")
            pyplot.plot(n[index],[0]*len(index),color='black')
            pyplot.legend()
            print "When s=",ss[int(index_best/len(ord_cus_s))],'pow=',ord_cus_s[int(index_best%len(ord_cus_s))],"integration sum is ",np.sum(y[index]-z[index]), " counts!"
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I_container[index_best],FOM_container[index_best],Ierr_container[index_best]

    def integrate_one_image_for_tick(self,img_path,center_pix,r_width,c_width,plot_live=PLOT_LIVE):
        cutoff_scale=INTEG_PARS['cutoff_scale']
        use_scale=INTEG_PARS['use_scale']
        #center_pix=INTEG_PARS['center_pix']
        #r_width=INTEG_PARS['r_width']
        #c_width=INTEG_PARS['c_width']
        integration_direction=INTEG_PARS['integration_direction']
        ord_cus_s=INTEG_PARS['ord_cus_s']
        ss=INTEG_PARS['ss']
        fct=INTEG_PARS['fct']
        img=misc.imread(img_path)
        #center_pix= list(np.where(img==np.max(img[center_pix[0]-20:center_pix[0]+20,center_pix[1]-20:center_pix[1]+20])))
        if use_scale:
            if cutoff_scale<1:
                cutoff=np.max(img)*cutoff_scale
            else:
                cutoff=cutoff_scale
            index_cutoff=np.argwhere(img>=cutoff)
        else:
            index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        x_span,y_span=x_max-x_min,y_max-y_min

        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        index=None
        peak_width=10
        if INTEG_PARS['integration_direction']=='y':
            peak_width==INTEG_PARS['c_width']/5
        elif INTEG_PARS['integration_direction']=='x':
            peak_width==INTEG_PARS['r_width']/5
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[rt:-1]-z[rt:-1]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            return sum_temp/(len(y)-peak_width*2)*len(y)

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = self.backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[index]-z[index]))
                Ibgr_container.append(abs(np.sum(z[index])))
                FOM_container.append(_cal_FOM(y,z,peak_width))
                Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                z_container.append(z)
        index_best=FOM_container.index(min(FOM_container))
        index = np.argsort(n)
        if plot_live:
            z=z_container[index_best]
            fig,ax=pyplot.subplots(2)
            ax[0].imshow(img)
            rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
            ax[0].add_patch(rect)

            ax[1].plot(n[index],y[index],color='blue',label="data")
            ax[1].plot(n[index],z[index],color="red",label="background")
            ax[1].plot(n[index],y[index]-z[index],color="m",label="data-background")
            ax[1].plot(n[index],[0]*len(index),color='black')
            pyplot.legend()
            print "When s=",ss[int(index_best/len(ord_cus_s))],'pow=',ord_cus_s[int(index_best%len(ord_cus_s))],"integration sum is ",np.sum(y[index]-z[index]), " counts!"
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I_container[index_best],FOM_container[index_best],Ierr_container[index_best]

    def batch_image_integration(self):
        data_info=self.data_info
        scan_number=data_info['scan_number']
        scan_type=data_info['scan_type']
        images_path=data_info['images_path']
        data_info['I']=[]
        data_info['Ierr']=[]
        data_info['Ibgr']=[]
        data_info['F']=[]
        data_info['Ferr']=[]
        data_info['ctot']=[]
        data_info['alpha']=[]
        data_info['beta']=[]

        for i in range(len(scan_number)):
            images_temp=images_path[i]
            I_temp,I_bgr_temp,I_err_temp,F_temp,Ferr_temp,ctot_temp,alpha_temp,beta_temp=[],[],[],[],[],[],[],[]
            for image in images_temp:
                print 'processing scan',str(scan_number[i]),'image',images_temp.index(image)
                I,I_bgr,I_err=self.integrate_one_image(image,plot_live=False)
                I_temp.append(I)
                I_bgr_temp.append(I_bgr)
                I_err_temp.append(I_err)
                scan_dict=self._formate_scan_from_data_info(data_info,scan_number[i],images_temp.index(image),I,I_err,I_bgr)
                #calculate the correction factor
                result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
                F_temp.append(result_dict['F'])
                Ferr_temp.append(result_dict['Ferr'])
                ctot_temp.append(result_dict['ctot'])
                alpha_temp.append(result_dict['alpha'])
                beta_temp.append(result_dict['beta'])
            data_info['I'].append(I_temp)
            data_info['Ierr'].append(I_err_temp)
            data_info['Ibgr'].append(I_bgr_temp)
            data_info['F'].append(F_temp)
            data_info['Ferr'].append(Ferr_temp)
            data_info['ctot'].append(ctot_temp)
            data_info['alpha'].append(alpha_temp)
            data_info['beta'].append(F_temp)
        self.data_info=data_info

    def _formate_scan_from_data_info(self,data_info,scan_number,image_number,I,Ierr,Ibgr):
        scan_index=data_info['scan_number'].index(scan_number)
        image_index=image_number
        or0_list=data_info['or0'][scan_index]
        or1_list=data_info['or1'][scan_index]
        or0_lib={'h':or0_list[0:3]}
        or0_lib['delta'],or0_lib['eta'],or0_lib['chi'],or0_lib['phi'],or0_lib['nu'],or0_lib['mu'],or0_lib['lam']=or0_list[3:10]
        or1_lib={'h':or1_list[0:3]}
        or1_lib['delta'],or1_lib['eta'],or1_lib['chi'],or1_lib['phi'],or1_lib['nu'],or1_lib['mu'],or1_lib['lam']=or1_list[3:10]

        psicG=(data_info['cell'][scan_index],or0_lib,or1_lib,data_info['n_azt'][scan_index])
        scan_dict = {'I':[I],
                     'norm':[data_info['norm'][scan_index][image_index]],
                     'Ierr':[Ierr],
                     'Ibgr':[Ibgr],
                     'dims':(1,0),
                     'transmision':[data_info['transmision'][scan_index][image_index]],
                     'phi':[data_info['phi'][scan_index][image_index]],
                     'chi':[data_info['chi'][scan_index][image_index]],
                     'eta':[data_info['eta'][scan_index][image_index]],
                     'mu':[data_info['mu'][scan_index][image_index]],
                     'nu':[data_info['nu'][scan_index][image_index]],
                     'del':[data_info['del'][scan_index][image_index]],
                     'G':psicG}
        return scan_dict

    #remove spikes for plotting and saving results purpose
    def remove_spikes(self,L,col_data,bragg_peaks=BRAGG_PEAKS,offset=BRAGG_PEAK_CUTOFF):
        cutoff_ranges=[]
        L_new=[]
        col_data_new=[]
        for peak in bragg_peaks:
            cutoff_ranges.append([peak-offset,peak+offset])
        for i in range(len(L)):
            l=L[i]
            sensor=False
            for cutoff in cutoff_ranges:
                if l>cutoff[0] and l<cutoff[1]:
                    sensor=True
                    break
                else:pass
            if not sensor:
                L_new.append(l)
                col_data_new.append(col_data[i])
            else:pass
        return L_new,col_data_new

    #you can plot results for several scans
    def plot_results(self,scan_number=None):
        data=self.data_info
        if scan_number!=None:
            scan_number=scan_number
        else:
            scan_number=data['scan_number']
        for scan in scan_number:
            scan_index=data['scan_number'].index(scan)
            scan_type=data['scan_type'][scan_index]
            if scan_type=='rodscan':
                x,y,yer=data['L'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
                x_,y_=self.remove_spikes(x,y)
                x_,yer_=self.remove_spikes(x,yer)
                x,y,yer=x_,y_,yer_
                title='CTR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+'L)'
                fig=pyplot.figure(figsize=(10,5))
                ax=fig.add_subplot(1,1,1)
                ax.set_yscale('log')
                ax.scatter(x,y,marker='s',s=5)
                ax.errorbar(x,y,yerr=yer,fmt=None)
                ax.plot(x,y,linestyle='-',lw=1.5)
                pyplot.xlim(xmin=-0.3,xmax=max(x)+0.3)
                pyplot.title(title)
            elif scan_type=='Escan':
                x,y,yer=data['E'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
                title='RAXR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+str(data['L'][scan_index][0])+')'
                fig=pyplot.figure(figsize=(8,4))
                ax=fig.add_subplot(1,1,1)
                #ax.set_yscale('log')
                ax.scatter(x,y,marker='s',s=7)
                ax.errorbar(x,y,yerr=yer,fmt=None)
                ax.plot(x,y,linestyle='-',lw=1.5)
                pyplot.title(title)
            else:
                pass
        return None
    def save_data(self,file_path='.',file_name='data',formate={'rodscan':['H','K','L',0,'F','Ferr'],'Escan':['E','H','K',0,'L','F','Ferr',2,2]}):
        #to be finished
        return None

#Call this when you have no idea where is the center pixe
def show_pixe_image(data_info,scan_number):
    scan_index=data_info['scan_number'].index(scan_number)
    images=data_info['images_path'][scan_index][0:min([10,len(data_info['images_path'][scan_index])])]
    for image in images:
        img=misc.imread(image)
        fig,ax=pyplot.subplots()
        ax.imshow(img)
    return None

if __name__=="__main__":
    spec_path='M:\\fwog\\members\\qiu05\\1704_APS_13IDC\\mica'
    spec_name='s5_100mM_NH4Cl_Zr_1.spec'
    scan_number=[11]
    dataset=data_integration.data_integration(spec_path=spec_path,spec_name=spec_name,scan_number=scan_number)
    dataset.plot_results()
    data_info=dataset.data_info
    dataset.integrate_one_image(img_path=data_info['images_path'][0][0],plot_live=PLOT_LIVE)
    dataset.integrate_images_twick_mode(scan_number=[11])

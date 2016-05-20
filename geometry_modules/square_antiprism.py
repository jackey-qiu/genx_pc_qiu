import numpy as np
from numpy.linalg import inv
import os

#see detail comments in hexahedra_4

x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

class monomer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=1):
        self.r=r
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.origin=origin
        self.domain_tag=domain_tag
        self.offset=index_offset*1
        self.build()
    
    def build(self,**arg):
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        i=1
        for each_theta in [theta]:
            for each_phi in np.arange(0,np.pi*2,phi*2):
                self.coordinative_members[self.coor_el+str(i)+'_'+self.center_el+str(1+self.offset)+self.domain_tag]=np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.origin
                i+=1
        for each_theta in [np.pi-theta]:
            for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                self.coordinative_members[self.coor_el+str(i)+'_'+self.center_el+str(1+self.offset)+self.domain_tag]=np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.origin
                i+=1
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=self.origin
        return True
        
    def rotate_translate(self,translate_mag=np.array([0.,0.,0.]),rot_axis=np.array([0,0,1]),rot_point=np.array([0,0,0]),rot_angle=0):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=_rotate(self.center_point[self.center_el+str(1+self.offset)+self.domain_tag],rot_axis,rot_point,rot_angle)+translate_mag
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=_rotate(self.coordinative_members[key],rot_axis,rot_point,rot_angle)+translate_mag
        return True
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('9\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[self.center_el][0],self.center_point[self.center_el][1],self.center_point[self.center_el][2])
        f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
class tetramer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0):
        self.r=r
        self.offset=index_offset*4
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        center_1,center_2,center_3,center_4=np.array([r,0,0]),np.array([0,r,0]),np.array([-r,0,0]),np.array([0,-r,0])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=center_1
        self.center_point[self.center_el+str(2+self.offset)+self.domain_tag]=center_2
        self.center_point[self.center_el+str(3+self.offset)+self.domain_tag]=center_3
        self.center_point[self.center_el+str(4+self.offset)+self.domain_tag]=center_4
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            if key==self.center_el+str(1+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key==self.center_el+str(2+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key==self.center_el+str(4+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            i=1
            for each_theta in [theta]:
                for each_phi in np.arange(0,np.pi*2,phi*2):
                    self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                    #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                    i+=1
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(3+self.offset)+self.domain_tag:
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('28\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
class hexamer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0):
        self.r=r
        self.offset=index_offset*6
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        center_1,center_2,center_3,center_4,center_5,center_6=np.array([r,0,0]),np.array([0,r,0]),np.array([-r,0,0]),np.array([0,-r,0]),np.array([0,0,-r]),np.array([0,0,r])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=center_1
        self.center_point[self.center_el+str(2+self.offset)+self.domain_tag]=center_2
        self.center_point[self.center_el+str(3+self.offset)+self.domain_tag]=center_3
        self.center_point[self.center_el+str(4+self.offset)+self.domain_tag]=center_4
        self.center_point[self.center_el+str(5+self.offset)+self.domain_tag]=center_5
        self.center_point[self.center_el+str(6+self.offset)+self.domain_tag]=center_6
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            if key==self.center_el+str(1+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key==self.center_el+str(2+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key==self.center_el+str(4+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            else:
                rot_angle=0
                rot_axis=np.array([0,0,1])
            i=1
            if key!=self.center_el+str(5+self.offset)+self.domain_tag:
                for each_theta in [theta]:
                    for each_phi in np.arange(0,np.pi*2,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(3+self.offset)+self.domain_tag or key==self.center_el+str(5+self.offset)+self.domain_tag:
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('38\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
        
class heptamer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0):
        self.r=r
        self.offset=index_offset*7
        #self.offset=index_offset
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        center_1,center_2,center_3,center_4,center_5,center_6,center_7=np.array([r,0,0]),np.array([0,r,0]),np.array([-r,0,0]),np.array([0,-r,0]),np.array([3*r,0,0]),np.array([2*r,r,0]),np.array([2*r,-r,0])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=center_1
        self.center_point[self.center_el+str(2+self.offset)+self.domain_tag]=center_2
        self.center_point[self.center_el+str(3+self.offset)+self.domain_tag]=center_3
        self.center_point[self.center_el+str(4+self.offset)+self.domain_tag]=center_4
        self.center_point[self.center_el+str(5+self.offset)+self.domain_tag]=center_5
        self.center_point[self.center_el+str(6+self.offset)+self.domain_tag]=center_6
        self.center_point[self.center_el+str(7+self.offset)+self.domain_tag]=center_7
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(5+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key==self.center_el+str(2+self.offset)+self.domain_tag or key==self.center_el+str(6+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key==self.center_el+str(4+self.offset)+self.domain_tag or key==self.center_el+str(7+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            i=1
            for each_theta in [theta]:
                for each_phi in np.arange(0,np.pi*2,phi*2):
                    self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                    #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                    i+=1
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(3+self.offset)+self.domain_tag or key==self.center_el+str(5+self.offset)+self.domain_tag:
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
        #extra rotation for center5 to center7
        for key in self.center_point.keys():
            if self.center_el+str(5+self.offset) in key or self.center_el+str(6+self.offset) in key or self.center_el+str(7+self.offset) in key:
                self.center_point[key]=_rotate(self.center_point[key],np.array([1,0,0]),np.array([1,0,0]),45)
                
        for key in self.coordinative_members.keys():
            if self.center_el+str(5+self.offset) in key or self.center_el+str(6+self.offset) in key or self.center_el+str(7+self.offset) in key:
                self.coordinative_members[key]=_rotate(self.coordinative_members[key],np.array([1,0,0]),np.array([1,0,0]),45)
        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('47\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
class decamer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0):
        self.r=r
        self.offset=index_offset*10
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        center_1,center_2,center_3,center_4,center_5,center_6,center_7,center_8,center_9,center_10=np.array([r,0,0]),np.array([0,r,0]),np.array([-r,0,0]),np.array([0,-r,0]),np.array([3*r,0,0]),np.array([2*r,r,0]),np.array([2*r,-r,0]),np.array([5*r,0,0]),np.array([4*r,r,0]),np.array([4*r,-r,0])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=center_1
        self.center_point[self.center_el+str(2+self.offset)+self.domain_tag]=center_2
        self.center_point[self.center_el+str(3+self.offset)+self.domain_tag]=center_3
        self.center_point[self.center_el+str(4+self.offset)+self.domain_tag]=center_4
        self.center_point[self.center_el+str(5+self.offset)+self.domain_tag]=center_5
        self.center_point[self.center_el+str(6+self.offset)+self.domain_tag]=center_6
        self.center_point[self.center_el+str(7+self.offset)+self.domain_tag]=center_7
        self.center_point[self.center_el+str(8+self.offset)+self.domain_tag]=center_8
        self.center_point[self.center_el+str(9+self.offset)+self.domain_tag]=center_9
        self.center_point[self.center_el+str(10+self.offset)+self.domain_tag]=center_10
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(5+self.offset)+self.domain_tag or key==self.center_el+str(8+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key==self.center_el+str(2+self.offset)+self.domain_tag or key==self.center_el+str(6+self.offset)+self.domain_tag or key==self.center_el+str(9+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key==self.center_el+str(4+self.offset)+self.domain_tag or key==self.center_el+str(7+self.offset)+self.domain_tag or key==self.center_el+str(10+self.offset)+self.domain_tag:
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            i=1
            for each_theta in [theta]:
                for each_phi in np.arange(0,np.pi*2,phi*2):
                    self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                    #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                    i+=1
            if key==self.center_el+str(1+self.offset)+self.domain_tag or key==self.center_el+str(3+self.offset)+self.domain_tag or key==self.center_el+str(5+self.offset)+self.domain_tag or key==self.center_el+str(8+self.offset)+self.domain_tag:
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
        #extra rotation for center5 to center7
        for key in self.center_point.keys():
            if self.center_el+str(5+self.offset) in key or self.center_el+str(6+self.offset) in key or self.center_el+str(7+self.offset) in key:
                self.center_point[key]=_rotate(self.center_point[key],np.array([1,0,0]),np.array([1,0,0]),45)
                
        for key in self.coordinative_members.keys():
            if self.center_el+str(5+self.offset) in key or self.center_el+str(6+self.offset) in key or self.center_el+str(7+self.offset) in key:
                self.coordinative_members[key]=_rotate(self.coordinative_members[key],np.array([1,0,0]),np.array([1,0,0]),45)
        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('66\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
class polymer_old_version():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0,level=10):
        self.r=r
        self.offset=index_offset*level
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.level=level
        self.level_top=[2]+range(6,level,3)
        self.level_bottom=range(4,level,3)+[level]
        self.level_middle=[3,1]+range(5,level-1,3)
        self.level_extra_rotation=range(5,self.level,6)+range(6,self.level,6)+range(7,self.level+1,6)
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        level_top_values=[np.array([r*2*i,r,0]) for i in range(len(self.level_top))]
        level_bottom_values=[np.array([r*2*i,-r,0]) for i in range(len(self.level_bottom))]
        level_middle_values=[np.array([-r+r*2*i,0,0]) for i in range(len(self.level_middle))]
        for i in range(len(self.level_top)):
            self.center_point[self.center_el+str(self.level_top[i]+self.offset)+self.domain_tag]=level_top_values[i]            
        for i in range(len(self.level_middle)):
            self.center_point[self.center_el+str(self.level_middle[i]+self.offset)+self.domain_tag]=level_middle_values[i]            
        for i in range(len(self.level_bottom)):
            self.center_point[self.center_el+str(self.level_bottom[i]+self.offset)+self.domain_tag]=level_bottom_values[i]
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            if key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_middle):
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_top):
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_bottom):
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            i=1
            for each_theta in [theta]:
                for each_phi in np.arange(0,np.pi*2,phi*2):
                    self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                    #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                    i+=1
            if key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_middle):
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
        #extra rotation for center5 to center7
        for key in self.center_point.keys():
            if key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_extra_rotation):
                self.center_point[key]=_rotate(self.center_point[key],np.array([1,0,0]),np.array([1,0,0]),45)
                
        for key in self.coordinative_members.keys():
            keys_center_point_extra_rotation=map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_extra_rotation)
            for each_key in keys_center_point_extra_rotation:
                if each_key in key:
                    self.coordinative_members[key]=_rotate(self.coordinative_members[key],np.array([1,0,0]),np.array([1,0,0]),45)
        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write(str(len(self.center_point.keys())+len(self.coordinative_members.keys()))+'\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
#level is a number in the list of [7,10,13,...,n-3,n], cap is a list of [0,2,4,...,m-2,m], and m must <= (n-1)/3.
#level is used to specify how long is the polymer, m is used to specify the occupied sites surrounding the central axis of the polymer
#the total number of center element is n+len(cap) *2
class polymer():
    def __init__(self,origin=np.array([0.,0.,0.]),r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=0,level=10,cap=[]):
        self.r=r
        self.offset=index_offset*level
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.level=level
        self.cap_level=cap
        self.level_top=[2]+range(6,level,3)
        self.level_bottom=range(4,level,3)+[level]
        self.level_middle=[3,1]+range(5,level-1,3)
        self.level_extra_rotation=range(5,self.level,6)+range(6,self.level,6)+range(7,self.level+1,6)
        self.center_el=center_el
        self.coor_el=coor_el
        self.domain_tag=domain_tag
        self.origin=origin
        self.find_centers()
        self.build()
        
        
    def find_centers(self):
        h=self.r*np.cos(self.theta)
        square_edge_len=2*self.r*np.sin(self.theta)*np.sin(np.deg2rad(45))
        r=h+0.5*square_edge_len
        level_top_values=[np.array([r*2*i,r,0]) for i in range(len(self.level_top))]
        level_bottom_values=[np.array([r*2*i,-r,0]) for i in range(len(self.level_bottom))]
        level_middle_values=[np.array([-r+r*2*i,0,0]) for i in range(len(self.level_middle))]
        for i in range(len(self.level_top)):
            self.center_point[self.center_el+str(self.level_top[i]+self.offset)+self.domain_tag]=level_top_values[i]            
        for i in range(len(self.level_middle)):
            self.center_point[self.center_el+str(self.level_middle[i]+self.offset)+self.domain_tag]=level_middle_values[i]            
        for i in range(len(self.level_bottom)):
            self.center_point[self.center_el+str(self.level_bottom[i]+self.offset)+self.domain_tag]=level_bottom_values[i]
        for i in self.cap_level:
            if i%4==0:
                self.center_point[self.center_el+str(i+self.offset)+'rA'+self.domain_tag]=np.array([r*i,0,r])
                self.center_point[self.center_el+str(i+self.offset)+'rB'+self.domain_tag]=np.array([r*i,0,-r])
            else:
                self.center_point[self.center_el+str(i+self.offset)+'rAR'+self.domain_tag]=np.array([r*i,0,r])
                self.center_point[self.center_el+str(i+self.offset)+'rBR'+self.domain_tag]=np.array([r*i,0,-r])
        
    def build(self,**arg):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+w**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        if arg.keys()!=[]:
            r,theta,phi=arg['r'],np.deg2rad(arg['theta']),np.deg2rad(arg['phi'])
        else:
            r,theta,phi=self.r,self.theta,self.phi
        for key in np.sort(self.center_point.keys()):
            rot_point=self.center_point[key]
            rot_angle=0
            rot_axis=np.array([1,0,0])
            if key==self.center_el+str(3+self.offset)+self.domain_tag:
                rot_angle=90
                rot_axis=np.array([0,-1,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_middle):
                rot_angle=90
                rot_axis=np.array([0,1,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_top):
                rot_angle=-90
                rot_axis=np.array([1,0,0])
            elif key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_bottom):
                rot_angle=-90
                rot_axis=np.array([-1,0,0])
            
            if ('rA' not in key) and ('rB' not in key):
                i=1
                for each_theta in [theta]:
                    for each_phi in np.arange(0,np.pi*2,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                        i+=1
            
                if key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_middle):
                    for each_theta in [np.pi-theta]:
                        for each_phi in np.arange(phi,np.pi*2+phi,phi*2):
                            self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                            #self.coordinative_members[key+'_'+self.coor_el+str(i)]=_rotate(self.coordinative_members[key+'_'+self.coor_el+str(i)],rot_axis,rot_point,rot_angle)
                            i+=1
            elif 'rA' in key:
                i=1
                for each_theta in [theta]:
                    for each_phi in np.arange(0,np.pi*2,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        i+=1
            elif 'rB' in key:
                i=1
                for each_theta in [np.pi-theta]:
                    for each_phi in np.arange(0,np.pi*2,phi*2):
                        self.coordinative_members[self.coor_el+str(i)+'_'+key]=_rotate(np.array([r*np.sin(each_theta)*np.cos(each_phi),r*np.sin(each_theta)*np.sin(each_phi),r*np.cos(each_theta)])+self.center_point[key],rot_axis,rot_point,rot_angle)
                        i+=1
        #extra rotation for center5 to center7
        for key in self.center_point.keys():
            if key in map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_extra_rotation) or 'rAR' in key or 'rBR' in key:
                self.center_point[key]=_rotate(self.center_point[key],np.array([1,0,0]),np.array([1,0,0]),45)
                
        for key in self.coordinative_members.keys():
            keys_center_point_extra_rotation=map(lambda x:self.center_el+str(x+self.offset)+self.domain_tag,self.level_extra_rotation)
            for each_key in keys_center_point_extra_rotation:
                if each_key in key:
                    self.coordinative_members[key]=_rotate(self.coordinative_members[key],np.array([1,0,0]),np.array([1,0,0]),45)
            if 'rAR' in key or 'rBR' in key:
                self.coordinative_members[key]=_rotate(self.coordinative_members[key],np.array([1,0,0]),np.array([1,0,0]),45)

        #translation operation
        for key in self.center_point.keys():
            self.center_point[key]=self.center_point[key]+self.origin
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=self.coordinative_members[key]+self.origin
            
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write(str(len(self.center_point.keys())+len(self.coordinative_members.keys()))+'\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
        
class oligomer(monomer,tetramer):
    def __init__(self,grid_matrix=[1,1,1],seperation=3,r=2.2,theta=59.2641329,center_el='Zr',coor_el='O',building_block='Monermer'):
        self.grid_matrix=grid_matrix
        self.seperation=seperation
        self.r=r
        self.theta=np.deg2rad(theta)
        self.phi=np.deg2rad(45)
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.building_block=building_block
        if self.building_block=='Monermer':
            self.motif=monomer(r=r,theta=theta,center_el=center_el,coor_el=coor_el)
        elif self.building_block=='Tetramer':
            self.motif=tetramer(r=r,theta=theta,center_el=center_el,coor_el=coor_el)
        self.grid=[]
        self.grid_pattern()
        
    def grid_pattern(self):
        for i in range(self.grid_matrix[0]):
            for j in range(self.grid_matrix[1]):
                for k in range(self.grid_matrix[2]):
                    self.grid.append([i*self.seperation,j*self.seperation,k*self.seperation])
    
    def build(self):
        for each_key in self.motif.center_point.keys():
            for each_grid in self.grid:
                i,j,k=np.array(each_grid)/self.seperation
                i,j,k=int(i),int(j),int(k)
                tag='_'+str(i)+'_'+str(j)+'_'+str(k)
                self.center_point[each_key+tag]=each_grid+self.motif.center_point[each_key]
        for each_key in self.motif.coordinative_members.keys():
            for each_grid in self.grid:
                i,j,k=np.array(each_grid)/self.seperation
                i,j,k=int(i),int(j),int(k)
                tag='_'+str(i)+'_'+str(j)+'_'+str(k)
                self.coordinative_members[each_key+tag]=each_grid+self.motif.coordinative_members[each_key]
                
    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        f.write('28\n#\n')
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
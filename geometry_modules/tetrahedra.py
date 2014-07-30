import numpy as np
from numpy.linalg import inv

#see detail comment in hexahedra_4
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

class share_face():
    def __init__(self,face=np.array([[0.,0.,0.],[0.5,0.5,0.5],[1.0,1.0,1.0]])):
        self.face=face
        
    def share_face_init(self,**args):
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list)) 
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        z_v=f3(np.zeros(3),np.cross(p1-center_point,p0-center_point))
        x_v=f3(np.zeros(3),p1-center_point)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        self.T=T
        r=f2(p0,center_point)
        r_bc=r*(np.sqrt(2.)/4.)
        r_ed=r*(np.sqrt(2.))
        body_center_new=np.array([0.,0.,r_bc*np.cos(0.)])
        body_center_old=np.dot(inv(T),body_center_new)+center_point
        p3_new=np.array([0.,0.,r_ed*np.cos(0.)])
        p3_old=np.dot(inv(T),p3_new)+center_point
        self.p3,self.center_point,self.r=p3_old,body_center_old,f2(body_center_old,p0)
        
    def cal_point_in_fit(self,r,theta,phi):
        #during fitting,use the same coordinate system, but a different origin
        #note the origin_coor is the new position for the sorbate0, ie new center point
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        point_in_original_coor=np.dot(inv(self.T),np.array([x,y,z]))+self.center_point
        return point_in_original_coor
        
    def print_xyz(self,file="D:\\test.xyz"):
        f=open(file,"w")
        f.write('5\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Sb', self.center_point[0],self.center_point[1],self.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[0,:][0],self.face[0,:][1],self.face[0,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[1,:][0],self.face[1,:][1],self.face[1,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[2,:][0],self.face[2,:][1],self.face[2,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p3[0],self.p3[1],self.p3[2])
        f.write(s)

        f.close()  
class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[2.5,2.5,2.5]])):
        self.edge=edge
        self.flag=None
        self.p0,self.p1=edge[0],edge[1]
    def cal_p2(self,ref_p=None,phi=0,**args):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
        ref_point=0
        if diff[2]==0:
            ref_point=origin+[0,0,1]
        elif ref_p!=None:
            ref_point=np.cross(p0-p1,np.cross(p0-p1,ref_p-p1))+origin
            #ref_point=ref_p
        else:
            x,y,z=0.,0.,0.
            #set the reference point as simply as possible,using the same distance assumption, we end up with a plane equation
            #then we try to find one cross point between one of the three basis and the plane we just got
            #here combine two line equations (ref-->p0,and ref-->p1,the distance should be the same)
            if diff[0]!=0:
                x=c/(2*diff[0])
            elif diff[1]!=0.:
                y=c/(2*diff[1])
            elif diff[2]!=0.:
                z=c/(2*diff[2])
            ref_point=np.array([x,y,z])
            if sum(ref_point)==0:
                #if the vector (p0-->p1) pass through origin [0,0,0],we need to specify another point satisfying the same-distance condition
                #here, we a known point (x0,y0,z0)([0,0,0] in this case) and the normal vector to calculate the plane equation, 
                #which is a(x-x0)+b(y-y0)+c(z-z0)=0, we specify x y to 1 and 0, calculate z value.
                #a b c coresponds to vector origin-->p0
                ref_point=np.array([1.,0.,-p0[0]/p0[2]])
        x1_v=f3(np.zeros(3),ref_point-origin)
        z1_v=f3(np.zeros(3),p1-origin)
        y1_v=np.cross(z1_v,x1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        #note the r is different from that in the case above
        #note in this case, phi can be in the range of [0,2pi], theta is pi/2
        r=dist/2*np.sqrt(3.)
        #ang_offset is used to ensure the ref point, body center and the two anchors are on the same plane when phi is set to 0
        ang1=109.5/180*np.pi/2
        ang2=np.pi/3
        ang_offset=np.arccos((np.cos(ang1)**2+(2*np.sin(ang1)*np.sin(ang2))**2-1)/(2*np.cos(ang1)*2*np.sin(ang1)*np.sin(ang2)))
        
        theta=np.pi/2
        x_p2=r*np.cos(phi+ang_offset)*np.sin(theta)
        y_p2=r*np.sin(phi+ang_offset)*np.sin(theta)
        z_p2=r*np.cos(theta)
        p2_new=np.array([x_p2,y_p2,z_p2])
        p2_old=np.dot(inv(T),p2_new)+origin
        self.p2=p2_old
        self.face=np.append(self.edge,[p2_old],axis=0)
        
    def apply_angle_offset_BD(self,distal_angle_offset=[0,0],distal_length_offset=[0,0]):
    
        p2,p3,ct,r=self.p2,self.p3,self.center_point,self.r
        r1,r2=r+distal_length_offset[0],r+distal_length_offset[1]
        ang1,ang2=distal_angle_offset[0]/180*np.pi,(distal_angle_offset[1]+109.5)/180*np.pi
        z1_v=f3(np.zeros(3),p2-ct)
        y1_v=f3(np.zeros(3),np.cross(p3-ct,p2-ct))
        x1_v=np.cross(z1_v,y1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        p2_new=np.dot(inv(T),np.array([r1*np.cos(0)*np.sin(ang1),r1*np.sin(0)*np.sin(ang1),r1*np.cos(ang1)]))+ct
        p3_new=np.dot(inv(T),np.array([r2*np.cos(0)*np.sin(ang2),r2*np.sin(0)*np.sin(ang2),r2*np.cos(ang2)]))+ct
        self.p2,self.p3=p2_new,p3_new
        
    def apply_top_angle_offset_BD(self,top_angle_offset=0):
        #the top angle by default is 109.5 dg, by using this function, you can customize the top_angle by setting the angle offset
        p0,p1,p2,p3,ct,r=self.p0,self.p1,self.p2,self.p3,self.center_point,self.r
        origin=(p0+p1)/2
        base=f2(p0,p1)
        original_top_angle=109.47/180*np.pi
        new_top_angle=(109.47+top_angle_offset)/180*np.pi
        height_tri_old=base/2/np.tan(original_top_angle/2)
        height_tri_new=base/2/np.tan(new_top_angle/2)
        length_diff=height_tri_new-height_tri_old
        transfer_vector=(f3(origin,ct)-origin)*length_diff
        self.center_point,self.p2,self.p3=self.center_point+transfer_vector,self.p2+transfer_vector,self.p3+transfer_vector
        
        
class share_corner(share_edge):
#if want to share none, then just set the corner coordinate to the first point arbitratly.
    def __init__(self,corner=np.array([0.,0.,0.])):
        self.corner=corner
        self.flag=None
    def cal_p1(self,r,theta,phi):
    #here we simply use the original coordinate system converted to spherical coordinate system, but at different origin
        x_p1=r*np.cos(phi)*np.sin(theta)+self.corner[0]
        y_p1=r*np.sin(phi)*np.sin(theta)+self.corner[1]
        z_p1=r*np.cos(theta)+self.corner[2]
        p1=np.array([x_p1,y_p1,z_p1])
        self.p1=p1
        self.edge=np.append(self.corner[np.newaxis,:],p1[np.newaxis,:],axis=0)
        
if __name__=='__main__':
    test1=tetrahedra_3.share_edge(edge=np.array([[0.,0.,0.],[5.,5.,5.]]))
    test1.cal_p2(theta=0,phi=np.pi/2)
    test1.share_face_init()
    print test1.face,test1.p3,test1.center_point
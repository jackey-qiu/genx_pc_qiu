import numpy as np
#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

O1=np.array([ 6.81396583,  2.53559692,  2.53559692]  )
O2=np.array([ 4.92299432,  3.87271569,  1.19847814] )
O4=np.array([ 7.13840525,  4.62794134,  4.16911471] )
O3=np.array([ 5.24743374,  5.96506011,  2.83199593] )
vector=np.array([ 1.56653209,  0.83488552, -1.38052541])
Zr=[ 6.81396583,  4.66777127,  1.99353372]
vector2=np.cross(O1-O2,O3-O2)


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
    
O1_opposit =_rotate(O1+vector,vector2,Zr,45)
O2_opposit =_rotate(O2+vector,vector2,Zr,45)
O3_opposit =_rotate(O3+vector,vector2,Zr,45)
O4_opposit =_rotate(O4+vector,vector2,Zr,45)

f=open('D://debug_square_antiprism.xyz',"w")
f.write('9\n#\n')
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O1[0],O1[1],O1[2])
f.write(s)
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O2[0],O2[1],O2[2])
f.write(s)
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O3[0],O3[1],O3[2])
f.write(s)
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O4[0],O4[1],O4[2])
f.write(s)
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O1_opposit[0],O1_opposit[1],O1_opposit[2])
f.write(s)                                   
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O2_opposit[0],O2_opposit[1],O2_opposit[2])
f.write(s)                                   
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O3_opposit[0],O3_opposit[1],O3_opposit[2])
f.write(s)                                   
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', O4_opposit[0],O4_opposit[1],O4_opposit[2])
f.write(s)
s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Zr', Zr[0],Zr[1],Zr[2])
f.write(s)
f.close()
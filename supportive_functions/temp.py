def test(a=1,b=2,**c):
    test2(a,b,**c)
    return None
    
def test2(a,b,**c):
    print a
    print b
    print c
    return None
    
if __name__=="__main__":
    test(1,2,d=6,e=8)
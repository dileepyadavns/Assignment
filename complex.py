#Complex operations using Python 

#importing Libraries
import re
import cmath
import math

#declared varaibles
x=2
y=4

z=complex(x,y) 

print(z)

cmath.phase(z)

print(z.real)

print(z.imag)

po=cmath.polar(z)

r,ph=po

print(r)

rect1=cmath.rect(r,ph)

print(rect1)

exp1=cmath.exp(z)

print(exp1)

logFun=cmath.log(z,2)

print(logFun)

logFun2=cmath.log10(z)

print(logFun2)

ex2=cmath.sqrt(z)

print(ex2)

x=1.0
y=1.0
a=math.inf
b=math.nan

p=complex(x,y)

print(p)

w = complex(x,a)

v = complex(x,b)

if cmath.isfinite(p):
       print ("Complex number is finite")
else : print ("Complex number is infinite")

if cmath.isinf(w):
       print ("Complex number is infinite")
else : print ("Complex number is finite")

if cmath.isnan(v):
       print ("Complex number is NaN")
else : print ("Complex number is not NaN")

print(cmath.pi)

print(cmath.e)

print(cmath.sin(z))

print(cmath.cos(z))

print(cmath.tan(z))

print(cmath.asin(z))

print(cmath.acos(z))

print(cmath.atan(z))

print(cmath.asinh(z))

print(cmath.acosh(z))

print(cmath.atanh(z))

print(cmath.sinh(z))

print(cmath.cosh(z))

print(cmath.tanh(z))


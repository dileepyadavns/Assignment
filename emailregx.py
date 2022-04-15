
import re
import cmath
import math

pat= '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$' # pattern which we want to check for 


#function for validating pattern in email
def validate(email):   
  
    if(re.search(pat,email)):   
        print("Valid Email")   
    else:   
        print("Invalid Email")   

   
email = input() #for taking input
validate(email) #calling function


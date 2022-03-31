#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openpyxl


# In[17]:


path="multi_threading_activity.xlsx"


# In[18]:


obj = openpyxl.load_workbook(path)


# In[20]:


sheet1 = obj.active


# In[21]:


mat=[]


# In[25]:


for i in range(1, sheet1.max_row+1):
  row=[]
  for j in range(1, sheet1.max_column+1):
    cell_obj = sheet1.cell(row=i, column=j)
    row.append(cell_obj.value)
  mat.append(row)
print(mat)  


# In[102]:


mat.pop()


# In[103]:


mat.pop()


# In[105]:


new=mat[1:]


# In[106]:


for i in range(len(new)):
  for j in range(2):
    if  type(new[i][j])==str:
      print(new[i][j])
      new[i][j]=0
    if  type(new[i][j])==None:
      print(new[i][j])
      new[i][j]=0


print(new)      


# In[107]:


print(len(new))


# In[195]:


import threading


# In[209]:


result=[]
def sum_of_two_rows(new):
  sumMat=[]
  for i in range(len(new)):
    sumMat.append(new[i][0]+new[i][1])
  result.append(sumMat) 
  print(sumMat)


# In[210]:


def diff_of_tow_rows(new):
  diffMat=[]
  for i in range(len(new)):
    diffMat.append(new[i][0]-new[i][1])
  result.append(diffMat)
  print(diffMat)


# In[211]:


t1=threading.Thread(target=sum_of_two_rows,args=(new,))


# In[212]:


t2=threading.Thread(target=diff_of_tow_rows,args=(new,))


# In[213]:


t1.start()


# In[214]:


t2.start()


# In[215]:


a=t1.join()


# In[216]:


b=t2.join()


# In[217]:


print(result)


# In[220]:


col1=result[0]


# In[223]:


col1


# In[222]:


col2=result[1]


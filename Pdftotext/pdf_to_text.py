#PDF to Text Convertion

import PyPDF2 #imported library

pdfFileObj = open('sample.pdf', 'rb') #given a pdf here as argument

print(pdfFileObj)

pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 

n=pdfReader.numPages #number of pages in pdf

print(n)

textfile= open('textfile.txt','w') #created a new text file to write contents from pdf after reading

#used for loop to iterrate over all the pages of pdf
for i in range(n):
  
  content=pdfReader.getPage(i).extractText() #read content from each page
  print(content)
  textfile.write(content)
  textfile.write('\n') #written into text file

file=open('textfile.txt','r') # to open text file

data = file.read() # to read text file

#function to search whether particular word in text file or not
def seaching_Word(sw): 
  if sw in data:
    occurrences = data.count(sw)
    return occurrences
  else:
    return "Not found matched word"

sw=input("Enter the word to search")
print(seaching_Word(sw))

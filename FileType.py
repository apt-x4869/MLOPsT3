import re
import os
import sys 
file = open(sys.argv[1]).readlines()
for line in file :
    line = line.replace('\n','')
    a = re.split(r'[.\s]\s*', line)
    DL = [ 'keras', 'tensorflow']
    if any(item in DL for item in a):
        print("Found to be a Deep Learning Project")
        exit()

for line in file :
    line = line.replace('\n','')
    a = re.split(r'[.\s]\s*', line)
    ML = ['sklearn']
    if any(item in ML for item in a):
        print("Found to be a Machine Learning Project")
        exit()
print("No Match Found")
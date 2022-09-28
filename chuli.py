import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, required = True)
args = parser.parse_args()

with open(args.f,'r') as f:
    #reader = csv.DictReader(f)
    reader = csv.reader(f)
    #head=reader.fieldnames
    logs = [row for row in reader]
    head = range(len(logs[0]))
log={}
mean={}
std={}
num = 2
for h in head:
    log[h]=np.array([row[h] for row in logs],dtype=np.float)
    mean[h] = [log[h][i::num].mean() for i in range(num)]
    std[h] = [log[h][i::num].std(ddof=1) for i in range(num)]

with open('result.csv','w') as f:
    w = csv.writer(f)
    w.writerow(head)
    for i in range(num):
        row1=[]
        row2=[]
        for h in head:
            row1.append(mean[h][i])
            row2.append(std[h][i])
        w.writerow(row1)
        w.writerow(row2)


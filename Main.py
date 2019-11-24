import collections
from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree
import pandas as pd

class Solution:
    def __init__(self):
        self.HashTable=defaultdict(lambda:defaultdict(set))
        self.BTreeTbale=defaultdict(lambda:OOBTree)
        self.map_col=None
        self.tables={}

    def inputfromfile(self,filename):
        raw,currow=[],0
        f=open(filename,"r")
        for strs in f.readlines():
            strs=strs.strip('\n')
            data=strs.split('|')
            raw.append([currow]+data)
            currow+=1
        self.map_col={name:i+1 for i,name in enumerate(raw[0][1:])}
        return raw

    def Hash(self,table,column):
        curHash=defaultdict(set)
        col=self.map_col[column]
        for data in table[1:]:
            curHash[data[col]].add(data[0])
        self.HashTable[column]=curHash

    def BTree(self,table,column):
        curBTree=OOBTree()
        col=self.map_col[column]
        for data in table[1:]:
            d=data[col]
            if d in curBTree.keys():
                curBTree[d].add(data[0])
            else:
                curBTree[d]={data[0]}
        self.BTreeTbale[column]=curBTree

    def parse(self,query):
        parsed_q=[]
        parsed_q.append((query[0],query[2],query[1]))
        for i in range(3,len(query)):
            if query[i] in ('and','or'):
                parsed_q.append((query[i+1],query[i+3],query[i+2]))
        return parsed_q

    def filter_and(self,table,parsed_lm):
        equal_rows=[]
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    equal_rows.append(self.HashTable[k][v])
                elif k in self.BTreeTbale:
                    equal_rows.append(self.BTreeTbale[k][v])
                else:
                    table=[row for row in table if row[self.map_col[k]]==v]
            elif symbol=='<':
                table=[row for row in table if row[self.map_col[k]]<v]
            elif symbol=='>':
                table=[row for row in table if row[self.map_col[k]]>v]
            elif symbol=='!=':
                table=[row for row in table if row[self.map_col[k]]!=v]
            elif symbol=='≥':
                table=[row for row in table if row[self.map_col[k]]>=v]
            elif symbol=='≤':
                table=[row for row in table if row[self.map_col[k]]<=v]
        if equal_rows:
            target_rows=set.intersection(*equal_rows)
            return [r for r in table if r[0] in target_rows]
        else:
            return table

    def filter_or(self,table,parsed_lm):
        rows=set()
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    rows|=self.HashTable[k][v]
                elif k in self.BTreeTbale:
                    rows|=self.BTreeTbale[k][v]
                else:
                    rows|=set([row[0] for row in table if row[self.map_col[k]]==v])
            elif symbol=='<':
                rows|=set([row[0] for row in table if row[self.map_col[k]]<v])
            elif symbol=='>':
                rows|=set([row[0] for row in table if row[self.map_col[k]]>v])
            elif symbol=='!=':
                rows|=set([row[0] for row in table if row[self.map_col[k]]!=v])
            elif symbol=='≥':
                rows|=set([row[0] for row in table if row[self.map_col[k]]>=v])
            elif symbol=='≤':
                rows|=set([row[0] for row in table if row[self.map_col[k]]<=v])
        return [r for r in table if r[0] in rows]


    def select(self,table,query):
        if 'and' in query:
            parsed_lm=self.parse(query)
            return self.filter_and(table,parsed_lm)
        elif 'or' in query:
            parsed_lm=self.parse(query)
            return self.filter_or(table,parsed_lm)
        else:
            parsed_lm=self.parse(query)
            for k,v,symbol in parsed_lm:
                if symbol=='=':
                    if k in self.HashTable:
                        target_rows=self.HashTable[k][v]
                        return [row for row in table if row[0] in target_rows]
                    elif k in self.BTreeTbale:
                        target_rows=self.BTreeTbale[k][v]
                        return [row for row in table if row[0] in target_rows]
                    else:
                        return [row for row in table if row[self.map_col[k]]==v]
                elif symbol == '<':
                    return [row for row in table if row[self.map_col[k]]<v]
                elif symbol == '>':
                    return [row for row in table if row[self.map_col[k]]>v]
                elif symbol == '!=':
                    return [row for row in table if row[self.map_col[k]]!=v]
                elif symbol == '≥':
                    return [row for row in table if row[self.map_col[k]]>=v]
                elif symbol == '≤':
                    return [row for row in table if row[self.map_col[k]]<=v]

    def project(self,table,query):
        pass

    def avg(self,table,query):
        pass

    def sumgroup(self,table,query):
        pass

    def avggroup(self,table,query):
        pass

    def join(self,table1,table2,query):
        pass

    def sort(self,table,query):
        pass

    def movavg(self,table,query):
        pass

    def movsum(self,table,query):
        pass

    def concat(self,table1,table2):
        pass

    def outputfile(self,table,filename):
        pass

    def ReadFromInput(self,testfile):
        f=open(testfile,"r")
        for strs in f.readlines():
            strs=strs.strip('\n')
            paras=re.split("[:=(), ]+",strs)
            returnTable=paras[0]
            func=paras[1]
            if func=='inputfromfile':
                self.tables[returnTable]=self.inputfromfile(paras[2]+'.txt')
            elif func=='select':
                self.tables[returnTable]=self.select(self.tables[paras[2]],paras[3:])
            elif func=='project':
                self.tables[returnTable]=self.project(self.tables[paras[2]],paras[3:])
            elif func=='avg':
                self.tables[returnTable]=self.avg(self.tables[paras[2]],paras[3:])
            elif func=='sumgroup':
                self.tables[returnTable]=self.sumgroup(self.tables[paras[2]],paras[3:])
            elif func=='avggroup':
                self.tables[returnTable]=self.avggroup(self.tables[paras[2]], paras[3:])
            elif func=='join':
                self.tables[returnTable]=self.join(self.tables[paras[2]],self.tables[paras[3]],paras[4:])
            elif func=='sort':
                self.tables[returnTable]=self.sort(self.tables[paras[2]], paras[3:])
            elif func=='movavg':
                self.tables[returnTable]=self.movavg(self.tables[paras[2]], paras[3:])
            elif func=='movsum':
                self.tables[returnTable]=self.movsum(self.tables[paras[2]], paras[3:])
            elif func=='concat':
                self.tables[returnTable]=self.concat(self.tables[paras[2]], self.tables[paras[3]])
            elif paras[0]=='outputfile':
                self.outputfile(self.tables[paras[1]],paras[2])


a=Solution()
raw=a.inputfromfile('sales1.txt')
test1_data=a.select(raw,"(   time > 50  ) and ( qty<30)")
test1=pd.DataFrame(data=test1_data)
test1.to_csv('C:/Users/asus/Desktop/test1.csv',index=False)

# test2_data=a.select(raw,"(   time > 80  ) or  (qty < 10)")
# test2=pd.DataFrame(data=test2_data)
# test2.to_csv('C:/Users/asus/Desktop/test2.csv',index=False)
#
# test3_data=a.select(raw,"(   time > 80  )")
# test3=pd.DataFrame(data=test3_data)
# test3.to_csv('C:/Users/asus/Desktop/test3.csv',index=False)

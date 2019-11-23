import collections
from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree
# import pandas as pd

class Solution:
    def __init__(self):
        self.HashTable=defaultdict(lambda:defaultdict(set))
        self.BTreeTbale=defaultdict(lambda:OOBTree)

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

    def parse(self,limits):
        parsed_lm=[]
        for limit in limits:
            limit=limit.strip(')( ')
            k,v=re.split("[^0-9a-zA-Z]+",limit)
            symbol=limit[len(k):len(limit)-len(v)]
            symbol=symbol.strip()
            parsed_lm.append((k,v,symbol))
        return parsed_lm

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
            limits=re.split('and',query)
            parsed_lm=self.parse(limits)
            return self.filter_and(table,parsed_lm)
        elif 'or' in query:
            limits=re.split('or',query)
            parsed_lm=self.parse(limits)
            return self.filter_or(table,parsed_lm)
        else:
            parsed_lm=self.parse([query])
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


a=Solution()
raw=a.inputfromfile('sales1.txt')
# test1_data=a.select(raw,"(   time > 50  ) and ( qty<30)")
# test1=pd.DataFrame(data=test1_data)
# test1.to_csv('C:/Users/asus/Desktop/test1.csv',index=False)
#
# test2_data=a.select(raw,"(   time > 80  ) or  (qty < 10)")
# test2=pd.DataFrame(data=test2_data)
# test2.to_csv('C:/Users/asus/Desktop/test2.csv',index=False)
#
# test3_data=a.select(raw,"(   time > 80  )")
# test3=pd.DataFrame(data=test3_data)
# test3.to_csv('C:/Users/asus/Desktop/test3.csv',index=False)

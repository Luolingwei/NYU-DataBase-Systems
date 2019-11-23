import collections
from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree

class Solution:
    def __int__(self):
        self.HashTable=defaultdict(lambda:defaultdict(list))
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
        curHash=defaultdict(list)
        col=self.map_col[column]
        for data in table[1:]:
            curHash[data[col]].append(data[0])
        self.HashTable[column]=curHash

    def BTree(self,table,column):
        curBTree=OOBTree()
        col=self.map_col[column]
        for data in table[1:]:
            d=data[col]
            if d in curBTree.keys():
                curBTree[d]+=[data[0]]
            else:
                curBTree[d]=[data[0]]
        self.BTreeTbale[column]=curBTree

    def parse(self,limits):
        parsed_lm=[]
        for limit in limits:
            limit=limit.strip(')( ')
            k,v=re.split("[^0-9a-zA-Z]+",limit)
            symbol=limit[len(k):len(limit)-len(v)]
            symbol=symbol.strip()
            parsed_lm.append((k,int(v),symbol))
        return parsed_lm

    def filter_and(self,table,parsed_lm):
        equal_cols=[]
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    equal_cols.append(self.HashTable[k][v])
                elif k in self.BTreeTbale:
                    equal_cols.append(self.BTreeTbale[k][v])
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
        target_rows=set(equal_cols[0]).intersection(equal_cols[1:])
        table=[row for row in table if row[0] in target_rows]
        return table

    def filter_or(self,table,parsed_lm):
        pass



    def select(self,table,query):
        if 'and' in query:
            limits=re.split('and',query)
            parsed_lm=self.parse(limits)
            return self.filter_and(table,parsed_lm)
        elif 'or' in query:
            limits=re.split('or',query)
            parsed_lm=self.parse(limits)
            return self.filter_or(table,parsed_lm)
        elif '=' in query:
            pass


a=Solution()
a.inputfromfile('sales1.txt')
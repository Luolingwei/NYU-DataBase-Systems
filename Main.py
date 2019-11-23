import collections
from collections import defaultdict
import pandas as pd
import re
from BTrees.OOBTree import OOBTree

class Solution:
    def __int__(self):
        self.HashTable=defaultdict(lambda:defaultdict(list))
        self.BTreeTbale=defaultdict(lambda:OOBTree)

    def inputfromfile(self,filename):
        raw=pd.read_csv(filename,sep='|')
        return raw

    def Hash(self,table,column):
        col=list(table.loc[:,column])
        curHash=defaultdict(list)
        for i,c in enumerate(col):
            curHash[c].append(i)
        self.HashTable[column]=curHash

    def BTree(self,table,column):
        col=list(table.loc[:,column])
        curBTree=OOBTree()
        for i,c in enumerate(col):
            if c in curBTree.keys():
                curBTree[c]+=[i]
            else:
                curBTree[c]=[i]
        self.BTreeTbale[column]=curBTree

    def parse(self,limits):
        parsed_lm=[]
        for limit in limits:
            k,v=re.split("[^0-9a-zA-Z]+",limit)
            symbol=limit[len(k):len(limit)-len(v)]
            parsed_lm.append((k,int(v),symbol))
        return parsed_lm

    def filter_and(self,table,parsed_lm):
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    postions=self.HashTable[k][v]
                    table=table.loc[postions,:]
                elif k in self.BTreeTbale:
                    postions=self.BTreeTbale[k][v]
                    table=table.ioc[postions, :]
                else:
                    table=table[table[k]==v]
            elif symbol=='<':
                table=table[table[k]<v]
            elif symbol=='>':
                table=table[table[k]>v]
            elif symbol=='!=':
                table=table[table[k]!=v]
            elif symbol=='≥':
                table=table[table[k]>=v]
            elif symbol=='≤':
                table=table[table[k]<=v]
        return table


    def select(self,table,query):
        if 'and' in query:
            limits=re.split('and',query)
            parsed_lm=self.parse(limits)
            return self.filter_and(table,parsed_lm)
        elif 'or' in query:
            strs=re.split('and',query)
            for limit in strs:
                pass
        elif '=' in query:
            pass


a=Solution()
a.inputfromfile('sales1.txt')
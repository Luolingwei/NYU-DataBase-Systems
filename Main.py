import collections
from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree
import pandas as pd

class Solution:
    def __init__(self):
        self.HashTable=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
        self.BTreeTbale=defaultdict(lambda:defaultdict(lambda:OOBTree()))
        self.map_func={'=':lambda x,y:x==y,'<':lambda x,y:x<y,'>':lambda x,y:x>y,'!=':lambda x,y:x!=y,'≥':lambda x,y:x>=y,'≤':lambda x,y:x<=y}
        self.map_col=defaultdict(dict)
        self.tables={}

    def inputfromfile(self,tablename,filename):
        raw,currow=[],0
        f=open(filename,"r")
        for strs in f.readlines():
            strs=strs.strip('\n')
            data=strs.split('|')
            raw.append([currow]+data)
            currow+=1
        self.map_col[tablename]={name:i+1 for i,name in enumerate(raw[0][1:])}
        return raw

    def Hash(self,tablename,table,column):
        curHash=defaultdict(set)
        col=self.map_col[tablename][column]
        for data in table[1:]:
            curHash[data[col]].add(data[0])
        self.HashTable[tablename][column]=curHash

    def BTree(self,tablename,table,column):
        curBTree=OOBTree()
        col=self.map_col[tablename][column]
        for data in table[1:]:
            d=data[col]
            if d in curBTree.keys():
                curBTree[d].add(data[0])
            else:
                curBTree[d]={data[0]}
        self.BTreeTbale[tablename][column]=curBTree

    def parse(self,query):
        parsed_q=[]
        parsed_q.append((query[0],query[2],query[1]))
        for i in range(3,len(query)):
            if query[i] in ('and','or'):
                parsed_q.append((query[i+1],query[i+3],query[i+2]))
        return parsed_q

    def filter_and(self,tablename,table,parsed_lm):
        equal_rows=[]
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    equal_rows.append(self.HashTable[k][v])
                elif k in self.BTreeTbale:
                    equal_rows.append(self.BTreeTbale[k][v])
                else:
                    table=[row for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)]
            else:
                table=[row for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)]
        if equal_rows:
            target_rows=set.intersection(*equal_rows)
            return [r for r in table if r[0] in target_rows]
        else:
            return table

    def filter_or(self,tablename,table,parsed_lm):
        rows=set()
        for k,v,symbol in parsed_lm:
            if symbol=='=':
                if k in self.HashTable:
                    rows|=self.HashTable[k][v]
                elif k in self.BTreeTbale:
                    rows|=self.BTreeTbale[k][v]
                else:
                    rows|=set([row[0] for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)])
            else:
                rows|=set([row[0] for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)])
        return [r for r in table if r[0] in rows]


    def select(self,tablename,table,query):
        if 'and' in query:
            parsed_lm=self.parse(query)
            return self.filter_and(tablename,table,parsed_lm)
        elif 'or' in query:
            parsed_lm=self.parse(query)
            return self.filter_or(tablename,table,parsed_lm)
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
                        return [row for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)]
                else:
                    return [row for row in table if self.map_func[symbol](row[self.map_col[tablename][k]],v)]


    def project(self,table,query):
        pass

    def avg(self,table,query):
        pass

    def sumgroup(self,table,query):
        pass

    def avggroup(self,table,query):
        pass

    def join(self,tablename1,table1,tablename2,table2,query):
        attr1,attr2,symbol=query[0].split('.')[1],query[2].split('.')[1],query[1]
        joined,row_idx=[],1
        joined+=[table1[0]+table2[0][1:]]
        index1_status=self.HashTable[tablename1][attr1] or self.BTreeTbale[tablename1][attr1]
        index2_status=self.HashTable[tablename2][attr2] or self.BTreeTbale[tablename2][attr2]
        if index1_status and index2_status:
            for key1 in index1_status.keys():
                for key2 in index2_status.keys():
                    if self.map_func[symbol](key1,key2):
                        for idx1 in index1_status[key1]:
                            for idx2 in index2_status[key2]:
                                joined.append([row_idx]+table1[idx1][1:]+table2[idx2][1:])
                                row_idx+=1
        elif index1_status and not index2_status:
            for key1 in index1_status.keys():
                for row2 in table2[1:]:
                    if self.map_func[symbol](key1,row2[self.map_col[tablename2][attr2]]):
                        for idx1 in index1_status[key1]:
                            joined.append([row_idx]+table1[idx1][1:]+row2[1:])
                            row_idx+=1
        elif not index1_status and index2_status:
            for key2 in index2_status.keys():
                for row1 in table1[1:]:
                    if self.map_func[symbol](row1[self.map_col[tablename1][attr1]],key2):
                        for idx2 in index2_status[key2]:
                            joined.append([row_idx]+row1[1:]+table2[idx2][1:])
                            row_idx+=1
        else:
            for row1 in table1[1:]:
                for row2 in table2[1:]:
                    if self.map_func[symbol](row1[self.map_col[tablename1][attr1]],row2[self.map_col[tablename2][attr2]]):
                        joined.append([row_idx]+row1[1:]+row2[1:])
                        row_idx+=1
        return joined


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
            paras=list(filter(None,re.split(":=|\)|\(|\\s+|,|(=)|(>)|(<)|(!=)|(≥)|(≤)",strs)))
            returnTable=paras[0]
            func=paras[1]
            if func=='inputfromfile':
                self.tables[returnTable]=self.inputfromfile(returnTable,paras[2]+'.txt')
            elif func=='select':
                self.tables[returnTable]=self.select(paras[2],self.tables[paras[2]],paras[3:])
            elif func=='project':
                self.tables[returnTable]=self.project(self.tables[paras[2]],paras[3:])
            elif func=='avg':
                self.tables[returnTable]=self.avg(self.tables[paras[2]],paras[3:])
            elif func=='sumgroup':
                self.tables[returnTable]=self.sumgroup(self.tables[paras[2]],paras[3:])
            elif func=='avggroup':
                self.tables[returnTable]=self.avggroup(self.tables[paras[2]], paras[3:])
            elif func=='join':
                self.tables[returnTable]=self.join(paras[2],self.tables[paras[2]],paras[3],self.tables[paras[3]],paras[4:])
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
            elif paras[0]=='Hash':
                self.Hash(paras[1],self.tables[paras[1]],paras[2])
            elif paras[0]=='BTree':
                self.BTree(paras[1],self.tables[paras[1]],paras[2])


a=Solution()
a.ReadFromInput('test.txt')

# raw=a.inputfromfile('sales1.txt')
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

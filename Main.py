from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree
import numpy as np
import pandas as pd
import time

class Solution:
    def __init__(self):
        self.HashTable=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
        self.BTreeTbale=defaultdict(lambda:defaultdict(lambda:OOBTree()))
        self.map_func={'=':lambda x,y:x==y,'<':lambda x,y:x<y,'>':lambda x,y:x>y,'!=':lambda x,y:x!=y,'>=':lambda x,y:x>=y,'<=':lambda x,y:x<=y}
        self.map_ops={'$': lambda x,y: x,'+': lambda x,y:x+y,'-':lambda x,y:x-y,'*':lambda x,y:x*y,'/':lambda x,y:x/y}
        self.map_col=defaultdict(dict)
        self.tables={}

    def inputfromfile(self,filename):
        raw,currow=[],0
        f=open(filename,"r")
        for strs in f.readlines():
            strs=strs.strip('\n')
            data=strs.split('|')
            raw.append([currow]+[float(d) if d.isdigit() else d for d in data])
            currow+=1
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

    def parse_select(self,query):
        parsed_q=[]
        pairs,idx,symbol=["",""],0,None
        for c in query:
            if c in self.map_func.keys():
                symbol=c
                idx+=1
            elif c in ('and','or'):
                parsed_q.append((pairs[0],pairs[1],symbol))
                pairs,idx,symbol=["", ""],0,None
            else:
                pairs[idx]+=c
        parsed_q.append((pairs[0],pairs[1],symbol))
        return parsed_q

    def parse_select2(self,expr):
        expr_parse = list(filter(None, re.split("(\+)|(-)|(\*)|(/)", expr)))
        if len(expr_parse) > 1:
            attr,symbol,num = expr_parse[0], expr_parse[1], float(expr_parse[2])
        else:
            attr,symbol,num = float(expr_parse[0]) if expr_parse[0].isdigit() else expr_parse[0], '$', 0
        return attr,symbol,num

    def filter_and(self,tablename,table,parsed_lm):
        equal_rows=[]
        for k,v,symbol in parsed_lm:
            parsed_k,symbol_k,num_k=self.parse_select2(k)
            parsed_v,symbol_v,num_v=self.parse_select2(v)
            if symbol=='=':
                if parsed_k in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value,num_k),parsed_v):
                            equal_rows.append(self.HashTable[parsed_k][tablename][value])
                elif parsed_k in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value,num_k),parsed_v):
                            equal_rows.append(self.BTreeTbale[parsed_k][tablename][value])
                elif parsed_v in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](value,num_v)):
                            equal_rows.append(self.HashTable[parsed_v][tablename][value])
                elif parsed_v in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[parsed_v][tablename]:
                        if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](value,num_v)):
                            equal_rows.append(self.BTreeTbale[parsed_v][tablename][value])
                else:
                    if isinstance(parsed_v,float):
                        table=[table[0]]+[row for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]],num_k),parsed_v)]
                    else:
                        table=[table[0]]+[row for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))]
            else:
                if isinstance(parsed_v,float):
                    table=[table[0]]+[row for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)]
                else:
                    table=[table[0]]+[row for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))]
        if equal_rows:
            target_rows=set.intersection(*equal_rows)
            return [table[0]]+[r for r in table[1:] if r[0] in target_rows]
        else:
            return table

    def filter_or(self,tablename,table,parsed_lm):
        rows=set()
        for k,v,symbol in parsed_lm:
            parsed_k,symbol_k,num_k=self.parse_select2(k)
            parsed_v,symbol_v,num_v=self.parse_select2(v)
            if symbol=='=':
                if parsed_k in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value,num_k),parsed_v):
                            rows|=self.HashTable[parsed_k][tablename][value]
                elif parsed_k in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value,num_k),parsed_v):
                            rows|=self.BTreeTbale[parsed_k][tablename][value]
                elif parsed_v in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](value,num_v)):
                            rows|self.HashTable[parsed_v][tablename][value]
                elif parsed_v in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[parsed_v][tablename]:
                        if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](value,num_v)):
                            rows|=self.BTreeTbale[parsed_v][tablename][value]
                else:
                    if isinstance(parsed_v, float):
                        rows|=set([row[0] for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)])
                    else:
                        rows|=set([row[0] for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))])
            else:
                if isinstance(parsed_v, float):
                    rows|=set([row[0] for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)])
                else:
                    rows|=set([row[0] for row in table[1:] if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]], num_v))])
        return [table[0]]+[r for r in table if r[0] in rows]

    def select(self,tablename,table,query):
        if 'and' in query:
            parsed_lm=self.parse_select(query)
            return self.filter_and(tablename,table,parsed_lm)
        else:
            parsed_lm=self.parse_select(query)
            return self.filter_or(tablename,table,parsed_lm)

    def project(self,table_name,table,query):
        projected = []
        projected.append([row[0] for row in table])
        for s in query:
            projected.append([row[self.map_col[table_name][s]] for row in table])
        projected = [[row[col] for row in projected] for col in range(len(projected[0]))]
        return projected

    def avg(self,table_name,table,query):
        res=[[0,"AVG_"+query[0]]]
        data=[row[self.map_col[table_name][query[0]]] for row in table[1:]]
        res.append([1,sum(data)/len(data)])
        return res

    def sumgroup(self,table_name,table,query):
        first=query[0]
        keys=[]
        #find all the key combinations
        for row in range(1,len(table)):
            comb=[]
            for var in query[1:]:
                comb.append(table[row][self.map_col[table_name][var]])
            keys.append(comb)
        keys=np.array(list(set([tuple(t) for t in keys])))
        dict={}
        for i in range(len(keys)):
            dict[tuple(keys[i])]=0
        for row in range(1, len(table)):
            k=[]
            for var in query[1:]:
                k.append(table[row][self.map_col[table_name][var]])
            key = tuple(k)
            if key in dict:
                dict[key]+=table[row][self.map_col[table_name][first]]
        header=[[0]+query[1:]+['sum_'+first]]
        table,row_idx=header,1
        for key in dict:
            row=[]
            for k in key:
                row.append(k)
            row.append(dict.get(key))
            table.append([row_idx]+row)
            row_idx+=1
        return table

    def avggroup(self,table_name,table,query):
        first = query[0]
        keys = []
        # find all the key combinations
        for row in range(1, len(table)):
            comb = []
            for var in query[1:]:
                comb.append(table[row][self.map_col[table_name][var]])
            keys.append(comb)
        keys = np.array(list(set([tuple(t) for t in keys])))
        dict = {}
        count = {}
        for i in range(len(keys)):
            dict[tuple(keys[i])] = 0
            count[tuple(keys[i])] = 0
        for row in range(1, len(table)):
            k = []
            for var in query[1:]:
                k.append(table[row][self.map_col[table_name][var]])
            key = tuple(k)
            if key in dict:
                count[key] += 1
                dict[key] += float(table[row][self.map_col[table_name][first]])
        header=[[0]+query[1:]+['avg_'+first]]
        table, row_idx = header, 1
        for key in dict:
            row = []
            for k in key:
                row.append(k)
            row.append(dict.get(key)/count.get(key))
            table.append([row_idx]+row)
            row_idx+=1
        return table

    def parse_join(self,tablename1,tablename2,query):
        def parse_helper(attr1,symbol,attr2):
            if attr1.split('.')[0]==tablename1:
                return (attr1.split('.')[1],attr2.split('.')[1],symbol)
            else:
                if symbol=='>=':
                    return (attr2.split('.')[1],attr1.split('.')[1],'<=')
                elif symbol=='<=':
                    return (attr2.split('.')[1],attr1.split('.')[1],'>=')
                elif symbol=='>':
                    return (attr2.split('.')[1],attr1.split('.')[1],'<')
                elif symbol=='<':
                    return (attr2.split('.')[1],attr1.split('.')[1],'>')
                else:
                    return (attr2.split('.')[1], attr1.split('.')[1], symbol)
        parsed_q=[]
        pairs,idx,symbol=["",""],0,None
        for c in query:
            if c in self.map_func.keys():
                symbol=c
                idx+=1
            elif c=='and':
                parsed_q.append(parse_helper(pairs[0],symbol,pairs[1]))
                pairs,idx,symbol=["", ""],0,None
            else:
                pairs[idx]+=c
        parsed_q.append(parse_helper(pairs[0],symbol,pairs[1]))
        return parsed_q

    def parse_join2(self,expr):
        expr_parse=list(filter(None, re.split("(\+)|(-)|(\*)|(/)", expr)))
        if len(expr_parse)>1:
            attr,symbol,num=expr_parse[0],expr_parse[1],float(expr_parse[2])
        else:
            attr,symbol,num=expr_parse[0],'$',0
        return attr,symbol,num

    def join(self,tablename1,table1,tablename2,table2,query):
        parsed_q=self.parse_join(tablename1,tablename2,query)
        expr1,expr2,symbol=parsed_q[0][0],parsed_q[0][1],parsed_q[0][2]
        attr1,inner_symbol1,num1=self.parse_join2(expr1)
        attr2,inner_symbol2,num2=self.parse_join2(expr2)
        joined,row_idx=[],1
        joined+=[[table1[0][0]]+[tablename1+'_'+w for w in table1[0][1:]]+[tablename2+'_'+w for w in table2[0][1:]]]
        index1_status=self.HashTable[tablename1][attr1] or self.BTreeTbale[tablename1][attr1]
        index2_status=self.HashTable[tablename2][attr2] or self.BTreeTbale[tablename2][attr2]
        if index1_status and index2_status:
            for key1 in index1_status.keys():
                for key2 in index2_status.keys():
                    if self.map_func[symbol](self.map_ops[inner_symbol1](key1,num1),self.map_ops[inner_symbol2](key2,num2)):
                        for idx1 in index1_status[key1]:
                            for idx2 in index2_status[key2]:
                                joined.append([row_idx]+table1[idx1][1:]+table2[idx2][1:])
                                row_idx+=1
        elif index1_status and not index2_status:
            for key1 in index1_status.keys():
                for row2 in table2[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](key1,num1),self.map_ops[inner_symbol2](row2[self.map_col[tablename2][attr2]],num2)):
                        for idx1 in index1_status[key1]:
                            joined.append([row_idx]+table1[idx1][1:]+row2[1:])
                            row_idx+=1
        elif not index1_status and index2_status:
            for key2 in index2_status.keys():
                for row1 in table1[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](row1[self.map_col[tablename1][attr1]],num1),self.map_ops[inner_symbol2](key2,num2)):
                        for idx2 in index2_status[key2]:
                            joined.append([row_idx]+row1[1:]+table2[idx2][1:])
                            row_idx+=1
        else:
            for row1 in table1[1:]:
                for row2 in table2[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](row1[self.map_col[tablename1][attr1]],num1),self.map_ops[inner_symbol2](row2[self.map_col[tablename2][attr2]],num2)):
                        joined.append([row_idx]+row1[1:]+row2[1:])
                        row_idx+=1
        temp={name:i+1 for i, name in enumerate(joined[0][1:])}
        for expr1,expr2,symbol in parsed_q[1:]:
            new_joined=[joined[0]]
            attr1,inner_symbol1,num1 = self.parse_join2(expr1)
            attr2,inner_symbol2,num2 = self.parse_join2(expr2)
            attr1,attr2=tablename1+'_'+attr1,tablename2+'_'+attr2
            for data in joined[1:]:
                if self.map_func[symbol](self.map_ops[inner_symbol1](data[temp[attr1]],num1),self.map_ops[inner_symbol2](data[temp[attr2]],num2)):
                    new_joined.append(data)
            joined=new_joined
        return joined

    def sort(self,tablename,table,query):
        header,data=table[0],table[1:]
        for col in query[::-1]:
            data.sort(key=lambda x:-x[self.map_col[tablename][col]])
        return [header]+data

    def movavg(self,tablename,table,query):
        col,avg_para=query[0],int(query[1])
        output=[[0,'movavg'+'_'+query[0]+'_'+query[1]]]
        col_num=self.map_col[tablename][col]
        data=[row[col_num] for row in table[1:]]
        for i in range(len(data)):
            curdata=data[max(0,i+1-avg_para):i+1]
            curavg=sum(curdata)/len(curdata)
            output.append([i+1,curavg])
        return output

    def movsum(self,tablename,table,query):
        col,sum_para=query[0],int(query[1])
        output=[[0,'movsum'+'_'+query[0]+'_'+query[1]]]
        col_num=self.map_col[tablename][col]
        data=[row[col_num] for row in table[1:]]
        for i in range(len(data)):
            output.append([i+1,sum(data[max(0,i+1-sum_para):i+1])])
        return output

    def concat(self,table1,table2):
        return table1+table2[1:]

    def outputfile(self,table,filename):
        tablefile=open(filename+'.txt','w')
        for row in table:
            tablefile.write('|'.join(map(str,row[1:]))+'\n')
        tablefile.close()

    def ReadFromInput(self,testfile):
        f=open(testfile,"r")
        for strs in f.readlines():
            strs=strs.strip('\n')
            paras=list(filter(None,re.split(":=|\)|\(|\\s+|,|(=)|(>)|(<)|(!=)|(>=)|(<=)",strs)))
            returnTable=paras[0]
            func=paras[1]
            start=time.perf_counter()
            oper=func
            if func=='inputfromfile':
                self.tables[returnTable]=self.inputfromfile(paras[2]+'.txt')
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
            elif func=='select':
                self.tables[returnTable]=self.select(paras[2],self.tables[paras[2]],paras[3:])
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # select=self.tables[returnTable]
                # test_select=pd.DataFrame(data=select)
                # test_select.to_csv('C:/Users/asus/Desktop/test_select.csv',index=False)
            elif func=='project':
                self.tables[returnTable]=self.project(paras[2],self.tables[paras[2]],paras[3:])
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # project=self.tables[returnTable]
                # test_select=pd.DataFrame(data=project)
                # test_select.to_csv('C:/Users/asus/Desktop/test_project.csv',index=False)
            elif func=='avg':
                self.tables[returnTable]=self.avg(paras[2],self.tables[paras[2]],paras[3:])
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # avg=self.tables[returnTable]
                # test_avg=pd.DataFrame(data=avg)
                # test_avg.to_csv('C:/Users/asus/Desktop/test_avg.csv',index=False)
            elif func=='sumgroup':
                self.tables[returnTable]=self.sumgroup(paras[2],self.tables[paras[2]],paras[3:])
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # sumgroup=self.tables[returnTable]
                # test_sumgroup=pd.DataFrame(data=sumgroup)
                # test_sumgroup.to_csv('C:/Users/asus/Desktop/test_sumgroup.csv',index=False)
            elif func=='avggroup':
                self.tables[returnTable]=self.avggroup(paras[2],self.tables[paras[2]], paras[3:])
                self.map_col[returnTable]={name:i+1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # avggroup=self.tables[returnTable]
                # test_avggroup=pd.DataFrame(data=avggroup)
                # test_avggroup.to_csv('C:/Users/asus/Desktop/test_avggroup.csv',index=False)
            elif func=='join':
                self.tables[returnTable]=self.join(paras[2],self.tables[paras[2]],paras[3],self.tables[paras[3]],paras[4:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                # join=self.tables[returnTable]
                # test_join=pd.DataFrame(data=join)
                # test_join.to_csv('C:/Users/asus/Desktop/test_join.csv',index=False)
            elif func=='sort':
                self.tables[returnTable]=self.sort(paras[2],self.tables[paras[2]], paras[3:])
                self.map_col[returnTable]={name:i+1 for i,name in enumerate(self.tables[returnTable][0][1:])}
                # sort=self.tables[returnTable]
                # test_sort=pd.DataFrame(data=sort)
                # test_sort.to_csv('C:/Users/asus/Desktop/test_sort.csv',index=False)
            elif func=='movavg':
                self.tables[returnTable]=self.movavg(paras[2],self.tables[paras[2]], paras[3:])
                self.map_col[returnTable]={name:i+1 for i,name in enumerate(self.tables[returnTable][0][1:])}
                # movavg=self.tables[returnTable]
                # test_movavg=pd.DataFrame(data=movavg)
                # test_movavg.to_csv('C:/Users/asus/Desktop/test_movavg.csv',index=False)
            elif func=='movsum':
                self.tables[returnTable]=self.movsum(paras[2],self.tables[paras[2]], paras[3:])
                self.map_col[returnTable]={name:i+1 for i,name in enumerate(self.tables[returnTable][0][1:])}
                # movsum=self.tables[returnTable]
                # test_movsum=pd.DataFrame(data=movsum)
                # test_movsum.to_csv('C:/Users/asus/Desktop/test_movsum.csv',index=False)
            elif func=='concat':
                self.tables[returnTable]=self.concat(self.tables[paras[2]], self.tables[paras[3]])
                self.map_col[returnTable]={name:i+1 for i,name in enumerate(self.tables[returnTable][0][1:])}
                # concat=self.tables[returnTable]
                # test_concat=pd.DataFrame(data=concat)
                # test_concat.to_csv('C:/Users/asus/Desktop/test_concat.csv',index=False)
            elif paras[0]=='outputfile':
                oper = paras[0]
                self.outputfile(self.tables[paras[1]],paras[2])
            elif paras[0]=='Hash':
                oper = paras[0]
                self.Hash(paras[1],self.tables[paras[1]],paras[2])
            elif paras[0]=='BTree':
                oper = paras[0]
                self.BTree(paras[1],self.tables[paras[1]],paras[2])
            end=time.perf_counter()
            print('Running time of %s: %f ms' % (oper, round((end - start)*1000,3)))


a=Solution()
a.ReadFromInput('test.txt')
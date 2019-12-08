from collections import defaultdict
import re
from BTrees.OOBTree import OOBTree
import time
import numpy as np
import copy


class Solution:
    # this is the __init__ function in python, we define our global variables here.
    def __init__(self):
        self.HashTable = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.BTreeTbale = defaultdict(lambda: defaultdict(lambda: OOBTree()))
        self.map_func = {'=': lambda x, y: x == y, '<': lambda x, y: x < y, '>': lambda x, y: x > y,'!=': lambda x, y: x != y, '>=': lambda x, y: x >= y, '<=': lambda x, y: x <= y}
        self.map_ops = {'$': lambda x, y: x, '+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y,'/': lambda x, y: x / y}
        self.map_col = defaultdict(dict)
        self.tables = {}

    # command example: R := inputfromfile(sales1)
    # inputfromfile function is used to accpet file and transfer it into table in list
    def inputfromfile(self, filename):
        raw, currow = [], 0
        f = open(filename, "r")
        for strs in f.readlines():
            strs = strs.strip('\n')
            data = strs.split('|')
            raw.append([currow] + [float(d) if d.isdigit() else d for d in data])
            currow += 1
        return raw

    # command example: Hash (S, Q)
    # this is the fuction used to hash a attribute if a table.
    # tablename is name of opreated table,table is data of opreated table, column is the name of column which we hash.
    # there is no output, the corresponding attribute will be hashed and stored in self.HashTable
    def Hash(self, tablename, table, column):
        curHash = defaultdict(set)
        col = self.map_col[tablename][column]
        for data in table[1:]:
            curHash[data[col]].add(data[0])
        self.HashTable[tablename][column] = curHash

    # command example: Btree (R, qty)
    # this is the fuction used to Btree a attribute if a table.
    # tablename is name of opreated table,table is data of opreated table, column is the name of column which we Btree.
    # there is no output, the corresponding Btree attribute will be stored in self.BTreeTbale
    def BTree(self, tablename, table, column):
        curBTree = OOBTree()
        col = self.map_col[tablename][column]
        for data in table[1:]:
            d = data[col]
            if d in curBTree.keys():
                curBTree[d].add(data[0])
            else:
                curBTree[d] = {data[0]}
        self.BTreeTbale[tablename][column] = curBTree

    # parse_select is a helper function for select, which tranform the select condition into individual several conditions. exp:  ((time > 50) or (qty < 30))=> [(time,50,>),(qty,30,<)]
    # query is the input condition in form of string
    # the output will be all individual condition stored in list. exp: [(time,50,>),(qty,30,<)]
    def parse_select(self, query):
        parsed_q = []
        pairs, idx, symbol = ["", ""], 0, None
        for c in query:
            if c in self.map_func.keys():
                symbol = c
                idx += 1
            elif c in ('and', 'or'):
                parsed_q.append((pairs[0], pairs[1], symbol))
                pairs, idx, symbol = ["", ""], 0, None
            else:
                pairs[idx] += c
        parsed_q.append((pairs[0], pairs[1], symbol))
        return parsed_q

    # parse_select2 is a helper function for select, which deal with the Operation in single condition. exp: 10+time, time/3
    # expr is individual condition
    # the output will be splited indidual condition in form of tuple, exp: (time,+,10), if there is no operation (only time), output will be (time,$,0)
    def parse_select2(self, expr):
        expr_parse = list(filter(None, re.split("(\+)|(-)|(\*)|(/)", expr)))
        if len(expr_parse) > 1:
            attr, symbol, num = expr_parse[0], expr_parse[1], float(expr_parse[2])
        else:
            try:
                attr = float(expr_parse[0])
            except:
                attr = expr_parse[0]
            symbol, num = '$', 0
        return attr, symbol, num

    # filter_and is a helper function for select, which deal with condition of and in select
    # tablename is name of operated table, table is the operated table, parsed_lm is all conditions needed in select
    # the output will be all rows which mathch the conditions in select, stored in form of table
    def filter_and(self, tablename, table, parsed_lm):
        equal_rows = []
        for k, v, symbol in parsed_lm:
            parsed_k, symbol_k, num_k = self.parse_select2(k)
            parsed_v, symbol_v, num_v = self.parse_select2(v)
            if symbol == '=':
                if parsed_k in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value, num_k), parsed_v):
                            equal_rows.append(self.HashTable[tablename][parsed_k][value])
                elif parsed_k in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value, num_k), parsed_v):
                            equal_rows.append(self.BTreeTbale[tablename][parsed_k][value])
                elif parsed_v in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](value, num_v)):
                            equal_rows.append(self.HashTable[tablename][parsed_v][value])
                elif parsed_v in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](value, num_v)):
                            equal_rows.append(self.BTreeTbale[tablename][parsed_v][value])
                else:
                    if isinstance(parsed_v, float):
                        table = [table[0]] + [row for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)]
                    else:
                        table = [table[0]] + [row for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))]
            else:
                if isinstance(parsed_v, float):
                    table = [table[0]] + [row for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)]
                else:
                    table = [table[0]] + [row for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))]
        if equal_rows:
            target_rows = set.intersection(*equal_rows)
            return [table[0]] + [r for r in table[1:] if r[0] in target_rows]
        else:
            return table

    # filter_or is a helper function for select, which deal with condition of or in select
    # tablename is name of operated table, table is the operated table, parsed_lm is all conditions needed in select
    # the output will be all rows which mathch the conditions in select, stored in form of table
    def filter_or(self, tablename, table, parsed_lm):
        rows = set()
        for k, v, symbol in parsed_lm:
            parsed_k, symbol_k, num_k = self.parse_select2(k)
            parsed_v, symbol_v, num_v = self.parse_select2(v)
            if symbol == '=':
                if parsed_k in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value, num_k), parsed_v):
                            rows |= self.HashTable[tablename][parsed_k][value]
                elif parsed_k in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_k]:
                        if self.map_func[symbol](self.map_ops[symbol_k](value, num_k), parsed_v):
                            rows |= self.BTreeTbale[tablename][parsed_k][value]
                elif parsed_v in self.HashTable[tablename]:
                    for value in self.HashTable[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](value, num_v)):
                            rows | self.HashTable[tablename][parsed_v][value]
                elif parsed_v in self.BTreeTbale[tablename]:
                    for value in self.BTreeTbale[tablename][parsed_v]:
                        if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](value, num_v)):
                            rows |= self.BTreeTbale[tablename][parsed_v][value]
                else:
                    if isinstance(parsed_v, float):
                        rows |= set([row[0] for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)])
                    else:
                        rows |= set([row[0] for row in table[1:] if self.map_func[symbol](parsed_k,self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]],num_v))])
            else:
                if isinstance(parsed_v, float):
                    rows |= set([row[0] for row in table[1:] if self.map_func[symbol](self.map_ops[symbol_k](row[self.map_col[tablename][parsed_k]], num_k), parsed_v)])
                else:
                    rows |= set([row[0] for row in table[1:] if self.map_func[symbol](parsed_k, self.map_ops[symbol_v](row[self.map_col[tablename][parsed_v]], num_v))])
        return [table[0]] + [r for r in table if r[0] in rows]

    # command example: R1 := select(R, (time > 50) or (qty < 30))
    # the function is used to select target rowss which match the query from table
    # tablename is name of operated table, table is the operated table query includes the query conditions
    # the output will be table with header and valid rows , and the resulting table will be stored in global variable self.tables
    def select(self, tablename, table, query):
        if 'and' in query:
            parsed_lm = self.parse_select(query)
            return self.filter_and(tablename, table, parsed_lm)
        else:
            parsed_lm = self.parse_select(query)
            return self.filter_or(tablename, table, parsed_lm)

    # command example: R2 := project(R1, saleid, qty, pricerange)
    # the function is used to select target columns from table
    # table_name is name of operated table, table is the operated table and query, query includes the column name
    # the output will be table with header and target columns, and the resulting table will be stored in global variable self.tables
    def project(self, table_name, table, query):
        projected = []
        projected.append([row[0] for row in table])
        for s in query:
            projected.append([row[self.map_col[table_name][s]] for row in table])
        projected = [[row[col] for row in projected] for col in range(len(projected[0]))]
        return projected

    # command example: R8 := sum(R1, qty)
    # the function is used to calculate the sum of certain column in table
    # the input for the function includes operated table_name, table and column name
    # the output will be header of column and sum of target attribute in float format
    def sum(self, table_name, table, query):
        res = [[0, "SUM_" + query[0] + '_'+ table_name]]
        data = [row[self.map_col[table_name][query[0]]] for row in table[1:]]
        res.append([1, sum(data)])
        return res

    # command example: R3 := avg(R1, qty)
    # the function is used to calculate the average of certain column in table
    # the input for the function includes operated table_name, table and column name
    # the output will be header of column and average of target attribute in floar format
    def avg(self, table_name, table, query):
        res = [[0, "AVG_" + query[0] + '_'+ table_name]]
        data = [row[self.map_col[table_name][query[0]]] for row in table[1:]]
        res.append([1, sum(data) / len(data)])
        return res

    # command example: R3 := count(R1)
    # the function is used to calculate the number of rows in a table
    # the input for the function includes operated table_name and table
    # the output will be header of column and count of rows in a table
    def count(self, table_name, table):
        res = [[0, "COUNT_" + table_name]]
        res.append([1, len(table) - 1])
        return res

    # command example: R4 := sumgroup(R1, time, qty)
    # the function is used to calculate the sum of certain attribute in group of other selected attributes
    # it accept new table_name, original table, query includes one attribute used to get sum and one or more attributes used to divide group
    # the output will be a new table with different group and corresponding sum
    def sumgroup(self, table_name, table, query):
        first = query[0]
        keys = []
        # find all the key combinations
        for row in range(1, len(table)):
            comb = []
            for var in query[1:]:
                comb.append(table[row][self.map_col[table_name][var]])
            keys.append(comb)
        dict = {}
        # initialization
        for i in range(len(keys)):
            dict[tuple(keys[i])] = 0
        for row in range(1, len(table)):
            k = []
            for var in query[1:]:
                k.append(table[row][self.map_col[table_name][var]])
            key = tuple(k)
            # update sum by group
            if key in dict.keys():
                dict[key] += table[row][self.map_col[table_name][first]]
        header = [[0] + query[1:] + ['sum_' + first]]
        table, row_idx = header, 1
        # build resulting table
        for key in dict:
            row = []
            for k in key:
                row.append(k)
            row.append(dict[key])
            table.append([row_idx] + row)
            row_idx += 1

        return table

    # command example: R6 := avggroup(R1, qty, pricerange)
    # the function is used to calculate the average of certain attribute in group of other selected attributes
    # it accept new table_name, original table, query includes one attribute used to get average and one or more attributes used to divide group
    # the output will be a new table with different group and corresponding average
    def avggroup(self, table_name, table, query):
        first = query[0]
        keys = []
        # find all the key combinations
        for row in range(1, len(table)):
            comb = []
            for var in query[1:]:
                comb.append(table[row][self.map_col[table_name][var]])
            keys.append(comb)
        # find the sum
        dict = {}
        # find the number
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
        header = [[0] + query[1:] + ['avg_' + first]]
        table, row_idx = header, 1
        for key in dict:
            row = []
            for k in key:
                row.append(k)
            # avg=sum/num
            row.append(dict.get(key) / count.get(key))
            table.append([row_idx] + row)
            row_idx += 1
        return table

    # command example: R8 := countgroup(R1, qty)
    # the function is used to count the number of certain attributes by group
    # it accept new table_name, original table, query includes one or more attributes providing combination key
    # the output will be a new table with header and the numbe of rows in different groups
    def countgroup(self, table_name, table, query):
        keys = []
        # find all the key combinations
        for row in range(1, len(table)):
            comb = []
            for var in query:
                comb.append(table[row][self.map_col[table_name][var]])
            keys.append(comb)
        dict = {}
        for i in range(len(keys)):
            dict[tuple(keys[i])] = 0
        for row in range(1, len(table)):
            k = []
            for var in query:
                k.append(table[row][self.map_col[table_name][var]])
            key = tuple(k)
            # count number of rows
            if key in dict.keys():
                dict[key] += 1
        header = [[0] + query + ['count']]
        table, row_idx = header, 1

        for key in dict:
            row = []
            for k in key:
                row.append(k)
            row.append(dict[key])
            table.append([row_idx] + row)
            row_idx += 1

        return table

    # parse_join is a helper function for join, which tranform the join condition into individual several conditions. exp:  (R1.qty > S.Q) and (R1.saleid = S.saleid)=> [(qty,Q,>),(saleid,saleid,=)]
    # tablename1,tablename2 is tablename of 2 operated tables, query is the input condition in form of string
    # the output will be all individual condition stored in list. exp: [(qty,Q,>),(saleid,saleid,=)]
    def parse_join(self, tablename1, tablename2, query):
        def parse_helper(attr1, symbol, attr2):
            if attr1.split('.')[0] == tablename1:
                return (attr1.split('.')[1], attr2.split('.')[1], symbol)
            else:
                if symbol == '>=':
                    return (attr2.split('.')[1], attr1.split('.')[1], '<=')
                elif symbol == '<=':
                    return (attr2.split('.')[1], attr1.split('.')[1], '>=')
                elif symbol == '>':
                    return (attr2.split('.')[1], attr1.split('.')[1], '<')
                elif symbol == '<':
                    return (attr2.split('.')[1], attr1.split('.')[1], '>')
                else:
                    return (attr2.split('.')[1], attr1.split('.')[1], symbol)

        parsed_q,equals= [],[]
        pairs, idx, symbol = ["", ""], 0, None
        for c in query+['and']:
            if c in self.map_func.keys():
                symbol = c
                idx += 1
            elif c == 'and':
                parse_lm=parse_helper(pairs[0], symbol, pairs[1])
                if parse_lm[2]=='=':
                    equals.append(parse_lm)
                else:
                    parsed_q.append(parse_lm)
                pairs, idx, symbol = ["", ""], 0, None
            else:
                pairs[idx] += c
        return equals + parsed_q

    # parse_join2 is a helper function for select, which deal with the Operation in single condition. exp: R1.qty+10, S.Q/2
    # expr is individual condition
    # the output will be splited indidual condition in form of tuple, exp: (qty,+,10), if there is no operation (R1.qty), output will be (qty,$,0)
    def parse_join2(self, expr):
        expr_parse = list(filter(None, re.split("(\+)|(-)|(\*)|(/)", expr)))
        if len(expr_parse) > 1:
            attr, symbol, num = expr_parse[0], expr_parse[1], float(expr_parse[2])
        else:
            attr, symbol, num = expr_parse[0], '$', 0
        return attr, symbol, num

    # command example: T1 := join(R1, S, (R1.saleid = S.saleid))
    # the function is used to calculate the join result of 2 tables.
    # the input tablename and table are name of opreated 2 tables and data of 2 tables. query is conditions of join.
    # the output will be the resulting table of join on 2 input tables
    def join(self, tablename1, table1, tablename2, table2, query):
        parsed_q = self.parse_join(tablename1, tablename2, query)
        expr1, expr2, symbol = parsed_q[0][0], parsed_q[0][1], parsed_q[0][2]
        attr1, inner_symbol1, num1 = self.parse_join2(expr1)
        attr2, inner_symbol2, num2 = self.parse_join2(expr2)
        joined, row_idx = [], 1
        joined += [[table1[0][0]] + [tablename1 + '_' + w for w in table1[0][1:]] + [tablename2 + '_' + w for w in table2[0][1:]]]
        index1_status = self.HashTable[tablename1][attr1] or self.BTreeTbale[tablename1][attr1]
        index2_status = self.HashTable[tablename2][attr2] or self.BTreeTbale[tablename2][attr2]
        if index1_status and index2_status:
            for key1 in index1_status.keys():
                for key2 in index2_status.keys():
                    if self.map_func[symbol](self.map_ops[inner_symbol1](key1, num1),self.map_ops[inner_symbol2](key2, num2)):
                        for idx1 in index1_status[key1]:
                            for idx2 in index2_status[key2]:
                                joined.append([row_idx] + table1[idx1][1:] + table2[idx2][1:])
                                row_idx += 1
        elif index1_status and not index2_status:
            for key1 in index1_status.keys():
                for row2 in table2[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](key1, num1),self.map_ops[inner_symbol2](row2[self.map_col[tablename2][attr2]], num2)):
                        for idx1 in index1_status[key1]:
                            joined.append([row_idx] + table1[idx1][1:] + row2[1:])
                            row_idx += 1
        elif not index1_status and index2_status:
            for key2 in index2_status.keys():
                for row1 in table1[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](row1[self.map_col[tablename1][attr1]], num1),self.map_ops[inner_symbol2](key2, num2)):
                        for idx2 in index2_status[key2]:
                            joined.append([row_idx] + row1[1:] + table2[idx2][1:])
                            row_idx += 1
        else:
            for row1 in table1[1:]:
                for row2 in table2[1:]:
                    if self.map_func[symbol](self.map_ops[inner_symbol1](row1[self.map_col[tablename1][attr1]], num1),self.map_ops[inner_symbol2](row2[self.map_col[tablename2][attr2]], num2)):
                        joined.append([row_idx] + row1[1:] + row2[1:])
                        row_idx += 1
        temp = {name: i + 1 for i, name in enumerate(joined[0][1:])}
        for expr1, expr2, symbol in parsed_q[1:]:
            new_joined = [joined[0]]
            attr1, inner_symbol1, num1 = self.parse_join2(expr1)
            attr2, inner_symbol2, num2 = self.parse_join2(expr2)
            attr1, attr2 = tablename1 + '_' + attr1, tablename2 + '_' + attr2
            for data in joined[1:]:
                if self.map_func[symbol](self.map_ops[inner_symbol1](data[temp[attr1]], num1),self.map_ops[inner_symbol2](data[temp[attr2]], num2)):
                    new_joined.append(data)
            joined = new_joined
        return joined

    # command example: T2prime := sort(T1, R1_time, S_C)
    # the function is used to sort table by different attributes
    # tablename is name of opreated table,table is data of opreated table, query contains columns on which sorting based on.
    # the output will be the resulting sorted table.
    def sort(self, tablename, table, query):
        header, data = table[0], table[1:]
        for col in query[::-1]:
            data.sort(key=lambda x: -x[self.map_col[tablename][col]])
        return [header] + data

    # command example: T3 := movavg(T2prime, R1_qty, 3)
    # the function is used to calculate moving average result of a column in table
    # tablename is name of opreated table,table is data of opreated table, query contains the column name and average num.
    # the output will be the original table + a moving average column.
    def movavg(self, tablename, table, query):
        curtable = copy.deepcopy(table)
        col, avg_para = query[0], int(query[1])
        curtable[0].append('movavg' + '_' + query[0] + '_' + query[1])
        col_num = self.map_col[tablename][col]
        data = [row[col_num] for row in curtable[1:]]
        for i in range(len(data)):
            curdata = data[max(0, i + 1 - avg_para):i + 1]
            curavg = sum(curdata) / len(curdata)
            curtable[i + 1].append(curavg)
        return curtable

    # command example: T4 := movsum(T2prime, R1_qty, 5)
    # the function is used to calculate moving sum result of a column in table
    # tablename is name of opreated table,table is data of opreated table, query contains the column name and sum num.
    # the output will be the original table + a moving sum column.
    def movsum(self, tablename, table, query):
        curtable = copy.deepcopy(table)
        col, sum_para = query[0], int(query[1])
        curtable[0].append('movsum' + '_' + query[0] + '_' + query[1])
        col_num = self.map_col[tablename][col]
        data = [row[col_num] for row in curtable[1:]]
        for i in range(len(data)):
            curtable[i + 1].append(sum(data[max(0, i + 1 - sum_para):i + 1]))
        return curtable

    # command example: Q5 := concat(Q4, Q2)
    # the function is used to combine 2 tables together.
    # table1 and table2 are data of 2 operated table.
    # the output will be the combination of 2 input tables.
    def concat(self, table1, table2):
        return table1 + table2[1:]

    # command example:outputtofile(Q5, Q5)
    # used to reach table in tables list from program and output it into txt file
    # it accepts the table created before and defines the name of output file
    # the output will be a txt file containing table, each attribute in the table is divided by "|"
    def outputfile(self, table, filename):
        tablefile = open('ll4123_yf1357_'+ filename + '.txt', 'w')
        for row in table:
            tablefile.write('|'.join(map(str, row[1:])) + '\n')
        tablefile.close()

    # the function is used to read form test file, it extracts commands seperately
    # once the command is matched with function name, the corresponding operation will be made
    def ReadFromInput(self, testfile):

        def writeops(returnTable,query):
            out.write("tablename:" + returnTable + '\n')
            out.write("operation:" + query + '\n')
            out.write('|'.join([returnTable+'.'+col for col in self.tables[returnTable][0][1:]]) + '\n')
            for row in self.tables[returnTable][1:]:
                out.write('|'.join(map(str, row[1:])) + '\n')
            out.write("------------------------------------------------------------------------" + '\n')
            out.write("------------------------------------------------------------------------" + '\n')

        f = open(testfile, "r")
        out = open("ll4123_yf1357_AllOperations"+'.txt', 'a')
        for strs in f.readlines():
            strs = strs.strip('\n')
            paras = list(filter(None, re.split(":=|\)|\(|\\s+|,|(=)|(>)|(<)|(!=)|(>=)|(<=)", strs)))
            returnTable = paras[0]
            func = paras[1]
            start = time.perf_counter()
            oper = func
            if func == 'inputfromfile':
                self.tables[returnTable] = self.inputfromfile(paras[2] + '.txt')
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'select':
                self.tables[returnTable] = self.select(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'project':
                self.tables[returnTable] = self.project(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'sum':
                self.tables[returnTable] = self.sum(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'avg':
                self.tables[returnTable] = self.avg(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'count':
                self.tables[returnTable] = self.count(paras[2], self.tables[paras[2]])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'sumgroup':
                self.tables[returnTable] = self.sumgroup(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'avggroup':
                self.tables[returnTable] = self.avggroup(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'countgroup':
                self.tables[returnTable] = self.countgroup(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'join':
                self.tables[returnTable] = self.join(paras[2], self.tables[paras[2]], paras[3], self.tables[paras[3]],paras[4:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'sort':
                self.tables[returnTable] = self.sort(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'movavg':
                self.tables[returnTable] = self.movavg(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'movsum':
                self.tables[returnTable] = self.movsum(paras[2], self.tables[paras[2]], paras[3:])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif func == 'concat':
                self.tables[returnTable] = self.concat(self.tables[paras[2]], self.tables[paras[3]])
                self.map_col[returnTable] = {name: i + 1 for i, name in enumerate(self.tables[returnTable][0][1:])}
                writeops(returnTable,strs)
            elif paras[0] == 'outputtofile':
                oper = paras[0]
                self.outputfile(self.tables[paras[1]], paras[2])
            elif paras[0] == 'Hash':
                oper = paras[0]
                self.Hash(paras[1], self.tables[paras[1]], paras[2])
            elif paras[0] == 'Btree':
                oper = paras[0]
                self.BTree(paras[1], self.tables[paras[1]], paras[2])
            end = time.perf_counter()
            print('Running time of %s: %f ms' % (oper, round((end - start) * 1000, 3)))


a = Solution()
a.ReadFromInput('test.txt')


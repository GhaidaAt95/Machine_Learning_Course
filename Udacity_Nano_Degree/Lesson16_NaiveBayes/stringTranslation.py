import string

str1  = "wy"
str2 = "gf"
str3 = 'u'

trg = "weeksyourweeks!----.ghaida*?,"

table = trg.maketrans(str1,str2,str3)
print(table)
print(trg.translate(table))
table2 = str.maketrans('','',string.punctuation)
print(table2)
print(trg.translate(table2))



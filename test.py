dict = {}
a = [1,2,3,4]
for index, item in enumerate(a):
    dict.update({index: item})
print(type(dict[0]))
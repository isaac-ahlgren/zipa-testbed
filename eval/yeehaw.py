from eval_tools import *
from matplotlib import pyplot as plt

dir = '/Volumes/testbed/'
keyword1 = 'lux_id_10.0.0.231_date_20240620*'
keyword2 = 'lux_id_10.0.0.232_date_20240620*'

thing1 = Signal_File(dir, keyword1)
thing2 = Signal_File(dir, keyword2)

# 1 second = 48,000 | need 5 secs of data
# data1 = thing1.read(240000)
print(thing1.read(240000))
# print(len(data1))
#data2 = thing2.read(240000)
# print(len(data2))
print(thing2.read(240000))

# plt.plot(data1)
# plt.plot(data2)
# plt.show()



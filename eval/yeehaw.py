from eval_tools import *

dir = '/mnt/data/'
keyword = 'lux_id_10.0.0.232_date_20240620*'

test = Signal_File(dir, keyword)
test.switch_files()

# 1 second = 48,000 | need 5 secs of data
print(test.read(240000))

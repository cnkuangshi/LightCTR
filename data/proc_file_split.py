import sys
import random

file_origin = open(sys.argv[1], 'r')
shardings = int(sys.argv[2])

names = []
for i in xrange(shardings):
    name = sys.argv[1] + '_' + str(i + 1) # start from 1
    names.append(name)

file_io_handle = []
for i in xrange(shardings):
    file_io_handle.append(open(names[i], 'w'))

rand_stand = 1.0 / shardings

for line in file_origin:
    v = random.random()
    part = int(v / rand_stand)
    assert part < shardings

    file_io_handle[part].write(line.strip())
    file_io_handle[part].write('\n')

for file in file_io_handle:
    file.close()

file_origin.close()

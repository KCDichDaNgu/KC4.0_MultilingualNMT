import os

raw = '/home/huy/nlp/NMT-LaVi/data/raw/'
file_list_lo = ['dev2023.lo',
                'train2023.lo']
file_list_vi = ['dev2023.vi',
                'train2023.vi']
file_list = file_list_lo + file_list_vi
dup = '/home/huy/nlp/NMT-LaVi/data/dup/'
pre_processed = '/home/huy/nlp/NMT-LaVi/data/pre_processed/'
print('File list: ', file_list)

# add '.' to end of all lines
for file in file_list:
    with open(raw + file,'r') as f, open(pre_processed + 'adddot_' + file,'w+') as f2:
        lines = f.readlines()
        for line in lines:
            if line[-2] != '.':
                line = line[:-1] + '.\n'
            f2.write(line)
print('Add dot done!')

# remove duplicate lines
file_name = ['dev2023', 
             'train2023']
for file in file_name:
    file_lo = pre_processed + 'adddot_' + file + '.lo'
    file_vi = pre_processed + 'adddot_' + file + '.vi'
    with open(file_lo,'r') as flo, open(file_vi,'r') as fvi, open(pre_processed + file + '.lo','w+') as flo2, open(pre_processed + file + '.vi','w+') as fvi2:
        lines_lo = flo.readlines()
        lines_vi = fvi.readlines()
        n = min(len(lines_lo),len(lines_vi))
        line_set_lo = set()
        line_set_vi = set()
        for i in range(0,n):
            if((lines_lo[i] not in line_set_lo) and (lines_vi[i] not in line_set_vi)):
                line_set_lo.add(lines_lo[i])
                line_set_vi.add(lines_vi[i])
                flo2.write(lines_lo[i])
                fvi2.write(lines_vi[i])



print('Remove duplicate done!')



# print duplicated lines
for file in file_list:
    with open(pre_processed + file,'r') as f:
        # max_len = 0
        # for line in f:
        #     max_len = max(max_len,len(line))
        # print(max_len)
        line_set = set()
        mi = 0
        lines = f.readlines()
        for i in range(0,len(lines)):
            if(len(lines[i]) > len(lines[mi])):
                mi = i
            if(lines[i] not in line_set):
                line_set.add(lines[i])
        print(file)
        print(f'Max len at {mi}, has total {len(lines[mi])} char')
        print(f'The file has total {len(line_set)} unique lines out of {len(lines)} lines')


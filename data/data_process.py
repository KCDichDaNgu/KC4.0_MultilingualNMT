# import os
# import regex

raw = '/home/huy/nlp/NMT-LaVi/data/raw/'
file_list_lo = ['dev2023.lo',
                'train2023.lo']
file_list_vi = ['dev2023.vi',
                'train2023.vi']
file_list = file_list_lo + file_list_vi
dup = '/home/huy/nlp/NMT-LaVi/data/dup/'
pre_processed = '/home/huy/nlp/NMT-LaVi/data/pre_processed/'
print('File list: ', file_list)

# # add '.' to end of all lines
# for file in file_list:
#     with open(raw + file,'r') as f, open(pre_processed + 'adddot_' + file,'w+') as f2:
#         lines = f.readlines()
#         for line in lines:
#             if line[-2] != '.':
#                 line = line[:-1] + '.\n'
#             f2.write(line)
# print('Add dot done!')

# # remove duplicate lines
# file_name = ['dev2023',     
#              'train2023']
# for file in file_name:
#     file_lo = pre_processed + 'adddot_' + file + '.lo'
#     file_vi = pre_processed + 'adddot_' + file + '.vi'
#     with open(file_lo,'r') as flo, open(file_vi,'r') as fvi, open(pre_processed + 'nodup_' + file + '.lo','w+') as flo2, open(pre_processed + 'nodup_' + file + '.vi','w+') as fvi2:
#         lines_lo = flo.readlines()
#         lines_vi = fvi.readlines()
#         n = min(len(lines_lo),len(lines_vi))
#         line_set_lo = set()
#         line_set_vi = set()
#         for i in range(0,n):
#             if((lines_lo[i] not in line_set_lo) and (lines_vi[i] not in line_set_vi)):
#                 line_set_lo.add(lines_lo[i])
#                 line_set_vi.add(lines_vi[i])
#                 flo2.write(lines_lo[i])
#                 fvi2.write(lines_vi[i])
# print('Remove duplicate done!')

# # # remove numbers and latin characters
# # for file in file_list_lo:
# #     with open(pre_processed + 'nodup_' + file,'r') as f, open(pre_processed + file,'w+') as f2:
# #         lines = f.readlines()
# #         for line in lines:
# #             line = regex.sub(r"(\s*[A-Za-z0-9])+",'',line)
# #             f2.write(line)
# # print('Remove numbers and latin characters done!')

# # remove line with emoji, links, html tags
# file_name = ['dev2023',     
#              'train2023']
# for file in file_name:
#     file_lo = pre_processed + 'nodup_' + file + '.lo'
#     file_vi = pre_processed + 'nodup_' + file + '.vi'
#     # file_lo_out = pre_processed + 'notrash_' + file + '.lo'
#     # file_vi_out = pre_processed + 'notrash_' + file + '.vi'
#     file_lo_out = pre_processed + file + '.lo'
#     file_vi_out = pre_processed + file + '.vi'
#     with open(file_lo,'r') as flo, open(file_vi,'r') as fvi, open(file_lo_out,'w+') as flo2, open(file_vi_out,'w+') as fvi2:
#         lines_lo = flo.readlines()
#         lines_vi = fvi.readlines()
#         n = min(len(lines_lo),len(lines_vi))
#         for i in range(0,n):
#             # no emoji 
                
#                 flo2.write(lines_lo[i])
#                 fvi2.write(lines_vi[i])
# print('Remove emoji, html, links done!')

# create 1000 liné version
file_lo = pre_processed + 'train2023.lo'
file_vi = pre_processed + 'train2023.vi'

with open(file_lo,'r') as flo, open(file_vi,'r') as fvi, open(pre_processed + 'train1000.lo','w+') as flo2, open(pre_processed + 'train1000.vi','w+') as fvi2:
    lines_lo = flo.readlines()
    lines_vi = fvi.readlines()
    n = min(len(lines_lo),len(lines_vi))
    for i in range(0,1000):
        flo2.write(lines_lo[i])
        fvi2.write(lines_vi[i])
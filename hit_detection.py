import csv
import queue
import copy
import numpy as np
from tkinter import filedialog

csv_file_path = filedialog.askopenfilename()
origin = queue.Queue()
vector = queue.Queue()
position_buffer = queue.Queue()

# 打開CSV文件
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as input_file:
    reader = csv.reader(input_file)

    title = next(reader)
    origin.put(title)

    for row in reader:
        while len(row) <= 4:
            row.append('')
        origin.put(copy.deepcopy(row))
        position_buffer.put(copy.deepcopy(row))
            
        if(position_buffer.qsize()>3):
            before = position_buffer.get()
            if row[1]=='1':
                if before[1]=='1':
                    vector_x = int(row[2])-int(before[2])
                    vector_y = int(row[3])-int(before[3])
                    row[2] = str(vector_x)
                    row[3] = str(vector_y)
                    row[4] = str(round(np.linalg.norm([vector_x, vector_y])))
                else:
                    row[2] = str(0)
                    row[3] = str(0)
                    row[4] = str(0)

        vector.put(row)

    while not position_buffer.empty():
        garbage = position_buffer.get()


'''
while not out.empty():
    output = out.get()
    print(output)
'''

QSize = 5
count_threshold = 2
change_threshold = 29

with open('test_output.csv', mode='w', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    new_row = origin.get()
    if(len(new_row)==4):
        new_row.append("Count")
    else:
        new_row[4] = "Count"
    writer.writerow(new_row)

    hit = 0
    reset = 0
    vector_buffer = queue.Queue()
    while not vector.empty():
        vector_value = vector.get()
        new_row = origin.get()
        if reset==0:
            if(vector_buffer.qsize()<QSize):
                if(int(vector_value[1])==1):
                    vector_buffer.put(vector_value)
                else:
                    while not vector_buffer.empty():
                        garbage = vector_buffer.get()
            else:

                if(int(vector_value[1])==1):
                    if (vector_value[4] != ''):
                        vector_buffer.put(vector_value)
                        count = 0
                        for i in range(QSize):
                            before = list(vector_buffer.queue)[i]
                            if ((int(vector_value[2])*int(before[2])<0) or (int(vector_value[3])*int(before[3])<0)):
                                count += 1
                        if(count>count_threshold):
                            if(int(vector_value[4])>change_threshold):
                                hit += 1
                                print(f'row : {vector_value[0]}')
                                reset = 7
                                while not vector_buffer.empty():
                                    garbage = vector_buffer.get()
                            else:
                                garbage = vector_buffer.get()
                        else:
                            garbage = vector_buffer.get()
                    else:
                        print(f'row : {vector_value[0]} don\'t have value')
                else:  
                    while not vector_buffer.empty():
                        garbage = vector_buffer.get()   

        else:
            reset -= 1
        new_row[4] = str(hit)
        writer.writerow(new_row)

    while not vector_buffer.empty():
        garbage = vector_buffer.get()
import main
import os

# point_list = []
# file_path = "SIM_DATA/EP1/"
# dirs = os.listdir(file_path)
# for file_name in dirs:
#     with open(file_path + "/" + file_name) as file_object:
#         for line in file_object:
#             line = line.rstrip()
#             line = line.split(",")
#             p = main.Point(float(line[3]), float(line[4]))
#             point_list.append(p)
#
# file_object = open("data.txt","w")
# for p in point_list:
#     file_object.write(str(p.x)+","+str(p.y))
#     file_object.write("\n")

num = 0
file_path = "SIM_DATA/EP1/"
with open("SIM_DATA/range_sim1.txt") as query_file:
    for line in query_file:
        line = line.rstrip()
        line = line.split(",")
        rang = main.Rang(float(line[0]),float(line[1]),float(line[2]),float(line[3]))
        dirs = os.listdir(file_path)
        for file_name in dirs:
            with open(file_path + "/" + file_name) as file_object:
                for line in file_object:
                    line = line.rstrip()
                    line = line.split(",")
                    if rang.x_min<=float(line[3])<=rang.x_max and rang.y_min<=float(line[4])<=rang.y_max:
                        num += 1
print(num)

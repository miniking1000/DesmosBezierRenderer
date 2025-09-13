import os
print("open this proggram and have a look you will understand")
fromnow_count = int(input("from now count: ")) # frame123 ... frame 435 meens its 123
how_many = int(input("how many: ")) #frame 123 ... frame 435 meens its 435 -123 + 1 =313
count = int(input("count: "))
try:
    print("default: frames")
    folder = input("folder: ")
    os.chdir(folder)
except:
    os.chdir("frames")
while how_many != 0:
    count += 1
    os.rename("frame" + str(fromnow_count) + ".png","frame" + str(count) + ".png")
    how_many -= 1
    fromnow_count += 1
    
# place it to folder where images

from PyDesmos import Graph
with open("timing.txt","r") as f:
    xc = []
    yc = []
    line = f.readline()
    while line != "":
        xc.append(float(line[line.index("x")+2:][:-1]))
        yc.append(float(line[2:line.index(" ")]))
        line = f.readline()
with Graph("pointcloud") as G:
    f, x, y = G.f, G.x, G.y
    f[x] = 0.0000017*(x ** 2)
    G.new_table({x: xc, y: yc})  # where x, y = G.x, G.y, OR

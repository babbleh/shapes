import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, affinity, MultiPolygon, ops, MultiPoint, Point, validation, difference, MultiLineString
import math
from tqdm import tqdm
import io
from PIL import Image

from tesselationGenerators import simple_generators

def shapefillTess_simplePrim_smart(boundGeom, tile, translations):
    #place the tile at the centroid of the boundGeom
    translation = (boundGeom.centroid.x - tile.centroid.x, boundGeom.centroid.y - tile.centroid.y)
    tile = affinity.translate(tile, xoff=translation[0], yoff=translation[1])

    tiles = [tile]
    tileset = validation.make_valid(MultiPolygon(tiles))
    threshold_dist = 2

    def check_translation(translation):
        new_tile = affinity.translate(tile, xoff=translation[0], yoff=translation[1])
        if new_tile.centroid.within(tileset):
            return False
        elif not boundGeom.contains(Point(new_tile.centroid)) and \
            boundGeom.exterior.distance(new_tile) > threshold_dist:
                return False
        return True
    
    #for each tile in tiles, translate it by each translation in translations
    i = 0
    while i < len(tiles):
        tile = tiles[i]
        for translation in translations:
            if check_translation(translation):
                new_tile = affinity.translate(tile, xoff=translation[0], yoff=translation[1])
                tiles.append(new_tile)
                tileset = ops.unary_union([tileset, new_tile])
        i += 1

    return MultiPolygon(tiles), boundGeom

np.random.seed(0) #ARG N

# Number of coordinates to generate
n = 4 #ARG N
canv_size = 25 #ARG N

# Generate random x and y values
x_coords = np.random.uniform(0, canv_size, n)
y_coords = np.random.uniform(0, canv_size, n)

# Pair them into coordinates
coords = list(zip(x_coords, y_coords))

#coords = ((1,1), (1,4), (4,2))
tiles = []
canvas = Polygon([(0,0), (0,canv_size), (canv_size,canv_size), (canv_size,0), (0,0)])
origin_dividing_voronoi = ops.voronoi_diagram(MultiPoint(coords))
#print voronoi diagram size
origin_dividing_voronoi = MultiPolygon([poly.intersection(canvas) for poly in origin_dividing_voronoi.geoms])
voronoi_gens = []

#TODO scale the voronoi diagram so that it fits the canvas, also scaling the origin points 

all_dict = {}
for origin in tqdm(coords):
    #randomly pick 1/2 for hex/square
    idx = np.random.randint(1,7) #ARG N
    print(simple_generators[idx][0])
    prim = simple_generators[idx][1]
    translx = simple_generators[idx][2]

    voronoi = None
    for geom in origin_dividing_voronoi.geoms:
        if geom.contains(Point(origin)):
            voronoi = geom
            break

    keepers = []
    outies = []
    tileset, voronoi = shapefillTess_simplePrim_smart(voronoi, prim, translx)
    for tile in tileset.geoms:
        if voronoi.contains(tile):
            keepers.append(tile)
        else:
            outies.append(tile.centroid)

    all_dict[origin] = (keepers, voronoi)

    #fig, ax = plt.subplots()
    for tile in keepers:
        x,y = tile.exterior.xy
        #ax.plot(tile.exterior.xy


    voronoi_gens = voronoi_gens + outies
    tiles = tiles + keepers



outside_voronoi = ops.voronoi_diagram(MultiPoint(voronoi_gens))

xlim = (canvas.bounds[0], canvas.bounds[2])
ylim = (canvas.bounds[1], canvas.bounds[3])


fig, ax = plt.subplots()
for tile in tqdm(tiles):
    x,y = tile.exterior.xy
    #fill the tiles
    ax.fill(x, y, color='black')
    ax.plot(x,y)

#set the axes
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.axis('off')  # Turn off the axes
plt.savefig('tesselation1.png', bbox_inches='tight', pad_inches=0) #ARGN, MKDIR if not exists


fig, ax = plt.subplots()
for tile in outside_voronoi.geoms:
    x,y = tile.exterior.xy
    #fill the tiles
    ax.fill(x, y, color='black')
    ax.plot(x,y)

#set the axes
ax.set_xlim(xlim)
ax.set_ylim(ylim)

#save the figure
#plt.savefig('tesselation.png')
ax.axis('off')  # Turn off the axes
plt.savefig('tesselation2.png', bbox_inches='tight', pad_inches=0) #ARGN, MKDIR if not exists


fig, ax = plt.subplots()
for tile in outside_voronoi.geoms:
    x,y = tile.exterior.xy
    #fill the tiles
    ax.fill(x, y, color='red')
    ax.plot(x,y, color="white")

for tile in tqdm(tiles):
    x,y = tile.exterior.xy
    ax.fill(x, y, color='red')
    ax.plot(x,y, color="white")

#set the axes
ax.set_xlim(xlim)
ax.set_ylim(ylim)

#save the figure
ax.axis('off')  # Turn off the axes
plt.savefig('tesselation3.png', bbox_inches='tight', pad_inches=0) #ARGN, MKDIR if not exists




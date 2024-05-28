import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, affinity, MultiPolygon, ops, MultiPoint, Point, validation, difference
import math
from tqdm import tqdm

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


# Number of coordinates to generate
n = 2
canv_size = 10
# Range for x and y values
x_min, x_max = 0, canv_size
y_min, y_max = 0, canv_size

# Generate random x and y values
x_coords = np.random.uniform(x_min, x_max, n)
y_coords = np.random.uniform(y_min, y_max, n)

# Pair them into coordinates
coords = list(zip(x_coords, y_coords))

#coords = ((1,1), (1,4), (4,2))
tiles = []
canvas = Polygon([(0,0), (0,canv_size), (canv_size,canv_size), (canv_size,0), (0,0)])
origin_dividing_voronoi = ops.voronoi_diagram(MultiPoint(coords))
#print voronoi diagram size
origin_dividing_voronoi = MultiPolygon([poly.intersection(canvas) for poly in origin_dividing_voronoi.geoms])

#TODO scale the voronoi diagram so that it fits the canvas, also scaling the origin points 

all_dict = {}
for origin in tqdm(coords):
    #randomly pick 1/2 for hex/square
    idx = np.random.randint(1,5)
    print(simple_generators[idx][0])
    prim = simple_generators[idx][1]
    translx = simple_generators[idx][2]

    voronoi = None
    for geom in origin_dividing_voronoi.geoms:
        if geom.contains(Point(origin)):
            voronoi = geom
            break

    tileset, voronoi = shapefillTess_simplePrim_smart(voronoi, prim, translx)

    all_dict[origin] = (tileset, voronoi)

    tiles.append([tile for tile in tileset.geoms])

#flatten the list of lists
tiles = [item for sublist in tiles for item in sublist]
"""
#plot the voronoi diagram
fig, ax = plt.subplots()
for geom in origin_dividing_voronoi.geoms:
    x,y = geom.exterior.xy
    ax.plot(x,y)

#plot the tesselation
for tile in tiles:
    x,y = tile.exterior.xy
    ax.plot(x,y)

#make the origins visible
for origin in coords:
    ax.plot(origin[0], origin[1], 'ro')
plt.show()
"""

#find the list of all tiles that intersect with the boundaries of the voronoi components
bound_tiles = []
voronoi_border = origin_dividing_voronoi.envelope.exterior

for tile in tiles:
    for origin in coords:
        if all_dict[origin][1].exterior.intersects(tile) and not voronoi_border.intersects(tile):
            bound_tiles.append(tile)
            break

#make another voronoi using the centroids of the bound tiles as and their union as the envelope
bound_centroids = MultiPoint([tile.centroid for tile in bound_tiles])
bound_union = ops.unary_union(bound_tiles)

#check if bound_union is a  polygon or a multipolygon
if bound_union.geom_type == 'Polygon':
    bound_union = MultiPolygon([bound_union])

bound_vors = []
for bound_segment in tqdm(bound_union.geoms):
    segment_centroids = bound_centroids.intersection(bound_segment)
    #plot the segment and centroids
    segment_voronoi = ops.voronoi_diagram(segment_centroids)
    vor_segs = [poly.intersection(bound_segment) for poly in segment_voronoi.geoms]
    vor_segs = [poly for poly in vor_segs if poly.geom_type == 'Polygon']
    bound_vors.append(vor_segs)

print(len(bound_vors))
#flatten the list of lists
bound_voronoi = [item for sublist in bound_vors for item in sublist]
adj_tiles = [poly.difference(bound_union) for poly in tiles]
adj_tiles = [poly for poly in adj_tiles if poly.geom_type == 'Polygon']
#plot the tileset
fig, ax = plt.subplots()
for tile in adj_tiles:
    x,y = tile.exterior.xy
    ax.plot(x,y)

for tile in bound_voronoi:
    x,y = tile.exterior.xy
    ax.plot(x,y)
plt.show()

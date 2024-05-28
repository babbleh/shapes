import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, affinity, MultiPolygon, ops, MultiPoint, Point, validation
from tqdm import tqdm
import argparse
from tesselationGenerators import simple_generators

def shapefillTess_simplePrim_smart(boundGeom, tile, translations):
    translation = (boundGeom.centroid.x - tile.centroid.x, boundGeom.centroid.y - tile.centroid.y)
    tile = affinity.translate(tile, xoff=translation[0], yoff=translation[1])
    tiles = [tile]
    tileset = validation.make_valid(MultiPolygon(tiles))
    threshold_dist = 2

    def check_translation(translation):
        new_tile = affinity.translate(tile, xoff=translation[0], yoff=translation[1])
        if new_tile.centroid.within(tileset):
            return False
        elif not boundGeom.contains(Point(new_tile.centroid)) and boundGeom.exterior.distance(new_tile) > threshold_dist:
            return False
        return True
    
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

def save_tessellation(tiles, xlim, ylim, filename):
    fig, ax = plt.subplots()
    for tile in tiles:
        if tile.geom_type == 'Polygon':
            x, y = tile.exterior.xy
            ax.fill(x, y, color='black')
            ax.plot(x, y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate and save tessellations based on Voronoi diagrams.")
    parser.add_argument('--seed', type=int, default=0, help='Initial seed for coordinate generation.')
    parser.add_argument('--num_coords_min', type=int, default=4, help='Minimum number of coordinates to generate.')
    parser.add_argument('--num_coords_max', type=int, default=10, help='Maximum number of coordinates to generate.')
    parser.add_argument('--canvas_size_min', type=int, default=25, help='Minimum size of the canvas for the tessellation.')
    parser.add_argument('--canvas_size_max', type=int, default=100, help='Maximum size of the canvas for the tessellation.')
    parser.add_argument('--output_prefix', type=str, default='tesselation', help='Prefix for the output file names.')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to run, incrementing the seed each time.')

    args = parser.parse_args()

    for batch in tqdm(range(args.num_batches)):
        current_seed = args.seed + batch
        np.random.seed(current_seed)
        
        num_coords = np.random.randint(args.num_coords_min, args.num_coords_max + 1)
        canv_size = np.random.randint(args.canvas_size_min, args.canvas_size_max + 1)
        
        x_min, x_max = 0, canv_size
        y_min, y_max = 0, canv_size

        x_coords = np.random.uniform(x_min, x_max, num_coords)
        y_coords = np.random.uniform(y_min, y_max, num_coords)
        coords = list(zip(x_coords, y_coords))

        tiles = []
        voronoi_gens = []
        canvas = Polygon([(0, 0), (0, canv_size), (canv_size, canv_size), (canv_size, 0), (0, 0)])
        origin_dividing_voronoi = ops.voronoi_diagram(MultiPoint(coords))
        origin_dividing_voronoi = MultiPolygon([poly.intersection(canvas) for poly in origin_dividing_voronoi.geoms])

        for origin in coords:
            idx = np.random.randint(1, 7)
            prim = simple_generators[idx][1]
            translx = simple_generators[idx][2]
            
            voronoi = None
            for geom in origin_dividing_voronoi.geoms:
                if geom.contains(Point(origin)):
                    voronoi = geom
                    break

            tileset, voronoi = shapefillTess_simplePrim_smart(voronoi, prim, translx)
            
            keepers = []
            outies = []

            for tile in tileset.geoms:
                if voronoi.contains(tile):
                    keepers.append(tile)
                else:
                    outies.append(tile.centroid)

            voronoi_gens = voronoi_gens + outies
            tiles += keepers

        filename = f"{args.output_prefix}/seed_{current_seed}.png"
        xlim = (0, canv_size)
        ylim = (0, canv_size)

        outside_voronoi = ops.voronoi_diagram(MultiPoint(voronoi_gens))
        
        save_tessellation(outside_voronoi.geoms, xlim, ylim, filename)

if __name__ == "__main__":
    main()

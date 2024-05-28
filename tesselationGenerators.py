
from shapely import Polygon
import math

simple_generators = {}

simple_generators[1] = (['square',
                          Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))),
                            ((1,0), (0,1), (-1,0), (0,-1))])

simple_generators[2] = (('hexagon',
                            Polygon(((0., 0.),
                                        (1.0, 0),
                                        (1.5, math.sqrt(3.0)/2.0),
                                        (1.0, math.sqrt(3.0)),
                                        (0., math.sqrt(3.0)),
                                        (-0.5, math.sqrt(3.0)/2.0),
                                        (0., 0.))),
                        ((1.5,math.sqrt(3)/2),
                           (0,math.sqrt(3)),
                             (-1.5,math.sqrt(3)/2),
                               (0,-math.sqrt(3)),
                                 (-1.5,-math.sqrt(3)/2),
                                   (1.5,-math.sqrt(3)/2))))

simple_generators[3] = (('basket',Polygon(((0,0),
                           (1.5,0),
                           (1.5,1),
                           (1,1),
                           (1,2),
                           (.5,2),
                           (.5,1),
                           (0,1),
                           (0,0))),((1,1),
                           (-1,1),
                           (1,-1),
                           (-1,-1))))
simple_generators[4] = ('equilateral_triangle',
        Polygon([(0, 0),
                          (1, 0),
                          (0.5, math.sqrt(3)/2),
                          (0, 0)]),
        ((1, 0), (-1, 0), (0.5, -math.sqrt(3)/2), (0.5, math.sqrt(3)/2),
                    (-0.5, math.sqrt(3)/2),
                    (-0.5, -math.sqrt(3)/2)))

simple_generators[5] = ('octagon',
        Polygon([(0, 0),
                          (1, 0),
                          (2, 1),
                          (2, 2),
                          (1, 3),
                          (0, 3),
                          (-1, 2),
                          (-1, 1),
                          (0, 0)]),  # Octagon
        ((3, 0),
                    (0, 3),
                    (-3, 0),
                    (0, -3)))

simple_generators[5] = ('rhombus',
                        Polygon([(0, 0), (1, 0), (1.5, 0.5), (0.5, 0.5), (0, 0)]),
                        ((1, 0), (0, .5), (-1, 0), (0, -.5)))

simple_generators[6] = ('cairo_pentagon',
                        Polygon([(0, 0), (1, 0), (1.5, 0.5), (1, 1), (0.5, 0.5), (0, 0)]),
                        ((1, 0), (1, 1), (-1, 1), (-1, 0), (-1, -1), (1, -1)))

# Additional Tilings
simple_generators[7] = ('dodecagon',
                        Polygon([(math.cos(math.pi * i / 6), math.sin(math.pi * i / 6)) for i in range(12)]),
                        ((2, 0), (1, math.sqrt(3)), (-1, math.sqrt(3)), (-2, 0), (-1, -math.sqrt(3)), (1, -math.sqrt(3))))





'''
tri_coords = ((0., 0.), (1.0, 0), (0.5, math.sqrt(3.0)/2.0), (0., 0.))
t1 = Polygon(tri_coords)
t2 = Polygon(zip(t1.exterior.coords.xy[0],
                 [y-math.sqrt(3.0)/2.0 for y in t1.exterior.coords.xy[1]]))
t2 = affinity.rotate(t2,angle=180)

triangle_primitive = MultiPolygon([t1, t2])
ts = [triangle_primitive]
triangle_translations = ((1,0), (0.5,math.sqrt(3)/2))

hex_coords = ((0., 0.), (1.0, 0), (1.5, math.sqrt(3.0)/2.0), (1.0, math.sqrt(3.0)), (0., math.sqrt(3.0)), (-0.5, math.sqrt(3.0)/2.0), (0., 0.))
hex_prim = Polygon(hex_coords)

hexes =[hex_prim]
hex_translations = ((1.5,math.sqrt(3)/2), (0,math.sqrt(3)))
'''
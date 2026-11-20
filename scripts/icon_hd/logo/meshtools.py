"""Mesh and spatial-processing actions, drawn from functional differences.

Use coarse/fine tessellation, smooth profiles, closed surfaces and spatial cells
rather than tracing the incidental model shown by the source raster artwork.
"""
from . import paraview_style as P


def _mesh(fill='url(#mesh)', dense=False):
    boundary = P.fill('M 2.5 8 L 9 2.5 L 20.5 5.5 L 22 15.5 '
                      'L 14 21.5 L 3 18.5 Z', fill)
    edges = 'M 2.5 8 L 11 11 L 9 2.5 M 11 11 L 20.5 5.5 '
    edges += 'M 11 11 L 22 15.5 M 11 11 L 14 21.5 M 11 11 L 3 18.5'
    if dense:
        edges += (' M 5.75 5.25 L 6.75 9.5 L 5.2 14.3 L 2.75 13.25 '
                  'M 14.75 4 L 15.75 8.25 L 21.25 10.5 '
                  'M 16.5 13.25 L 18 18.5 L 12.5 16.25 L 8.5 20 '
                  'M 6.75 9.5 L 10 6.75 L 15.75 8.25 '
                  'M 5.2 14.3 L 12.5 16.25 L 16.5 13.25')
    return boundary + P.stroke(edges, sw=.65)


def _blue():
    return P.linear('mesh', [(0, P.BLUE_PALE), (.6, P.BLUE_MID), (1, P.BLUE_STRONG)])


def decimate():
    # Fewer, larger facets; minus refers to polygon count, not deleting an entity.
    body = [_mesh(dense=False),
            P.circle(18.7, 19, 3.6, P.YELLOW),
            P.stroke('M 16.6 19 H 20.8', sw=1.5)]
    return P.svg('<title>Decimate mesh — fewer facets</title>\n'+''.join(body), defs=_blue())


def subdivide():
    body = [_mesh(dense=True), P.circle(18.7, 19, 3.6, P.GREEN_LT),
            P.stroke('M 16.6 19 H 20.8 M 18.7 16.9 V 21.1', sw=1.3)]
    return P.svg('<title>Subdivide mesh — more facets</title>\n'+''.join(body), defs=_blue())


def laplacian():
    defs = P.linear('smooth', [(0, P.BLUE_PALE), (1, P.BLUE_MID)])
    body = [P.fill('M 2 12 C 5 11 6 3 11 3 C 16 3 16 11 22 9 '
                   'V 18 C 16 20 6 20 2 18 Z', 'url(#smooth)'),
            P.stroke('M 2 12 L 5 9 L 8 2 L 12 5 L 15 3 L 18 9 L 22 9',
                     colour=P.GREY_DK, sw=.7).replace('/>', ' stroke-dasharray="1.2 1"/>'),
            P.stroke('M 3 15 C 9 18 15 13 21 14 '
                     'M 8 7 C 7 10 8 15 9 18 M 14 6 C 13 10 15 14 15 18', sw=.6),
            # L = Laplacian. Outlined form avoids the host font dependency.
            P.stroke('M 17 17 V 22 H 21', sw=1.3)]
    return P.svg('<title>Laplacian mesh smoothing</title>\n'+''.join(body), defs=defs)


def smooth_filter():
    defs = P.radial('smooth', [(0, P.BLUE_PALE), (.55, P.BLUE_MID), (1, P.BLUE_STRONG)],
                    cx=.3, cy=.25, r=.9)
    body = [P.circle(12, 9.5, 7.5, 'url(#smooth)'),
            P.stroke('M 3 10 C 7 6 17 14 21 9', colour=P.GREY_DK, sw=.7),
            P.stroke('M 3 18 L 4.8 22 L 6.6 18 '
                     'M 8.6 18 H 13 M 10.8 18 V 22 '
                     'M 15.7 18 V 22 M 15.7 20 L 19.2 18 M 15.7 20 L 19.2 22', sw=1.05)]
    return P.svg('<title>VTK smooth filter — smoothed surface</title>\n'+''.join(body), defs=defs)


def surface_render():
    defs = P.linear('surface', [(0, P.GREEN_LT), (.45, P.GREEN), (1, P.GREEN_DK)])
    body = [P.fill('M 2 10 C 5 10 7 2 11 2 C 15 2 17 9 22 7 '
                   'V 17 C 17 19 15 13 11 13 C 7 13 5 22 2 21 Z', 'url(#surface)'),
            P.stroke('M 2 15 C 6 14 7 7.5 11 7.5 C 16 7.5 17 14 22 12 '
                     'M 7 5 C 6 10 8 13 7 17 M 16 6 C 15 10 17 14 16 17',
                     colour=P.INK, sw=.55)]
    return P.svg('<title>Surface rendering — continuous shaded sheet</title>\n'+''.join(body), defs=defs)


def mesh_menu():
    # A readable triangular sheet, not a shaded polyhedron mistaken for a solid.
    vertices = [(2.5, 6), (11, 3), (21, 6), (3.5, 13), (12, 11),
                (21.5, 14), (5, 21), (13, 20), (22, 22)]
    faces = [(0,1,4),(0,4,3),(1,2,4),(2,5,4),
             (3,4,6),(4,7,6),(4,5,7),(5,8,7)]
    colors = [P.BLUE_PALE,P.BLUE_MID,P.BLUE_MID,P.BLUE_PALE,
              P.BLUE_MID,P.BLUE_PALE,P.BLUE_PALE,P.BLUE_MID]
    body = [P.poly([vertices[i] for i in face], color)
            for face,color in zip(faces,colors)]
    for x,y in vertices:
        body.append(P.circle(x,y,.65,P.INK,outline=None))
    return P.svg('<title>Mesh tools — connected vertices and triangular faces</title>\n'+''.join(body))


def poisson():
    # Before/after: disconnected oriented samples become a watertight surface.
    # Both depict the same round subject; the arrow communicates reconstruction.
    defs=P.radial('closed',[(0,P.GREEN_LT),(.7,P.GREEN),(1,P.GREEN_DK)],
                  cx=.3,cy=.25,r=.9)
    samples=[(5,3.8,0,-1.5),(2.1,6,-1.2,-.7),(1.5,10.2,-1.3,0),
             (2.4,14.4,-1.2,.7),(5.3,16.5,0,1.5),(8,14.2,1.2,.8),
             (8.5,6.4,1.1,-.8)]
    body=[]
    for x,y,dx,dy in samples:
        body.append(P.stroke(f'M {x} {y} l {dx} {dy}',colour=P.BLUE_STRONG,sw=.65))
        body.append(P.circle(x,y,.65,P.BLUE_STRONG,outline=P.INK))
    body += [f'<ellipse cx="18.1" cy="10.2" rx="4.8" ry="6.7" '
             f'fill="url(#closed)" stroke="{P.INK}"/>',
             P.stroke('M 18.1 3.5 C 14.7 6.3 14.7 14.1 18.1 16.9 '
                      'M 18.1 3.5 C 21.5 6.3 21.5 14.1 18.1 16.9 '
                      'M 13.4 10.2 Q 18.1 13 22.8 10.2',sw=.55),
             P.fill('M 8.2 9 H 11 V 7.6 L 14 10.2 L 11 12.8 '
                    'V 11.4 H 8.2 Z',P.YELLOW,sw=.55),
             # P is a compact functional discriminator from generic meshing.
             P.stroke('M 10 22 V 18.7 H 13 V 20.5 H 10',sw=1)]
    return P.svg('<title>Poisson — oriented point samples to closed reconstructed surface</title>\n'+''.join(body),defs=defs)


def convex_hull():
    # A taut enclosing boundary around a deliberately non-convex point set.
    # No internal mesh spokes: those previously confused this with triangulation.
    outer=[(2,7),(8,2),(18,3),(22,12),(17,22),(3,19)]
    body=[P.poly(outer,P.BLUE_PALE),
          P.stroke('M 2 7 L 8 2 L 18 3 L 22 12 L 17 22 L 3 19 Z',
                   colour=P.GREEN_DK,sw=1.5)]
    for x,y in outer+[(7,8),(10,6),(8.5,13),(7,17),(13,19),(16,15),(19,12)]:
        body.append(P.circle(x,y,.8,P.BLUE_STRONG,outline=P.INK))
    return P.svg('<title>Convex hull — tight outer boundary enclosing scattered points</title>\n'+''.join(body))


def voxel_sampling():
    # Uniform cells with many muted input samples and one retained green sample
    # each. This makes density reduction visible, unlike the former solid cube.
    body=[P.rect(2,4,20,18,P.BLUE_PALE,r=0),
          P.stroke('M 8.667 4 V 22 M 15.333 4 V 22 M 2 13 H 22',sw=.75)]
    for cx in [5.333,12,18.667]:
        for cy in [8.5,17.5]:
            for dx,dy in [(-1.8,-2.1),(1.7,-1.8),(-1.6,2)]:
                body.append(P.circle(cx+dx,cy+dy,.48,P.GREY_MID,outline=None))
            body.append(P.circle(cx,cy,1.05,P.GREEN,outline=P.INK))
    body.append(P.fill('M 5 1 H 17 V .4 L 20 2.3 L 17 4.2 V 3.5 H 5 Z',
                       P.YELLOW,sw=.4))
    return P.svg('<title>Voxel downsample — many input points to one representative per equal cell</title>\n'+''.join(body))


def octree():
    # Literal one-to-eight hierarchy; avoid another voxel-looking wireframe.
    body=[P.cube(12,5,3.3,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.BLUE_STRONG),
          P.stroke('M 12 8.3 V 10.5 M 3.4 10.5 H 20.6 '
                   'M 3.4 10.5 V 12.5 M 9.1 10.5 V 12.5 '
                   'M 14.9 10.5 V 12.5 M 20.6 10.5 V 12.5 '
                   'M 3.4 16.5 V 18.5 M 9.1 16.5 V 18.5 '
                   'M 14.9 16.5 V 18.5 M 20.6 16.5 V 18.5',sw=.65)]
    for y in [14.5,20.5]:
        for x in [3.4,9.1,14.9,20.6]:
            body.append(P.cube(x,y,1.75,top=P.GREEN_LT,left=P.GREEN,right=P.GREEN_DK))
    return P.svg('<title>Octree — one parent spatial volume and eight child cells</title>\n'+''.join(body))


ICONS={
 'decimate.png':('filter/filterDecimate.svg',decimate,'网格减面'),
 'subdivide.png':('mesh/meshSubdivide.svg',subdivide,'网格细分'),
 'meshSmooth.png':('mesh/meshSmoothLaplacian.svg',laplacian,'Laplacian 平滑'),
 'vtkSmooth.png':('filter/filterSmoothVTK.svg',smooth_filter,'VTK 平滑滤波'),
 'surface.svg':('surface/surfaceRender.svg',surface_render,'曲面渲染'),
 'mesh4.png':('mesh/meshTools.svg',mesh_menu,'网格工具'),
 'poisson.png':('mesh/meshPoissonReconstruction.svg',poisson,'Poisson 重建'),
 'convex_concave_hull.png':('mesh/meshConvexHull.svg',convex_hull,'凸包'),
 'voxelSampling.jpg':('mesh/meshVoxelSampling.svg',voxel_sampling,'体素降采样'),
 'octree.png':('mesh/meshOctree.svg',octree,'八叉树工具'),
}

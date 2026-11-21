"""qPCL-owned, function-recognisable icon drawings in the ParaView palette."""
from . import paraview_style as P
INK=P.INK
PAPER='#FAFAF6'

def dot(x,y,c=P.BLUE_STRONG,r=.78):return P.circle(x,y,r,c,outline=INK)
def cloud(points,color=P.BLUE_STRONG,r=.75):return ''.join(dot(x,y,color,r) for x,y in points)
def line(d,c=P.GREY_DK,w=.7):return P.stroke(d,colour=c,sw=w)

def convex_hull():
 pts=[(3,8),(8,3),(18,2),(22,9),(20,18),(14,22),(4,19),(1,13)]
 body=[P.stroke('M 3 8 L 8 3 L 18 2 L 22 9 L 20 18 L 14 22 L 4 19 L 1 13 Z',colour=P.GREEN_DK,sw=1.4)]
 body+= [dot(x,y,P.BLUE_STRONG,.8) for x,y in pts+[(7,9),(12,6),(10,14),(16,13)]]
 return P.svg('<title>Convex or concave hull around a point set</title>'+''.join(body))

def correspondence():
 a=[(3,5),(4,11),(3,18),(7,8),(8,15)];b=[(20,5),(18,11),(21,18),(16,8),(15,15)]
 links=[(0,0),(1,2),(2,4),(3,3),(4,1)]
 body=[line(' '.join(f'M {a[i][0]} {a[i][1]} L {b[j][0]} {b[j][1]}' for i,j in links),P.GREY_MID,.55).replace('/>',' stroke-dasharray="1.4 1"/>'),cloud(a,P.BLUE_STRONG,.9),cloud(b,P.GREEN,.9)]
 return P.svg('<title>Correspondence grouping between two feature sets</title>'+''.join(body))

def euclidean_clusters():
 sets=[([(3,5),(5,4),(4,7),(7,6)],P.BLUE_STRONG),([(15,4),(18,5),(16,7),(20,7)],P.GREEN),([(6,15),(8,17),(5,19),(9,20)],P.TAN),([(16,15),(19,16),(17,19)],P.ROSE)]
 body=[]
 for pts,c in sets:
  cx=sum(x for x,y in pts)/len(pts);cy=sum(y for x,y in pts)/len(pts)
  d='M '+' L '.join(f'{cx+(x-cx)*1.65:.1f} {cy+(y-cy)*1.65:.1f}' for x,y in pts)+' Z'
  body += [line(d,c,.65).replace('/>',' stroke-dasharray="1 0.8"/>'),cloud(pts,c,.82)]
 return P.svg('<title>Euclidean clustering: separate neighboring groups</title>'+''.join(body))

def projection_filter():
 pts=[(3,3),(7,4),(12,2),(17,5),(21,3),(5,7),(11,6),(18,8)]
 body=[P.poly([(2,16),(19,11),(22,13),(5,19)],P.BLUE_PALE),line('M 2 16 L 19 11 M 5 19 L 22 13',P.INK,.65)]
 body += [dot(x,y,P.BLUE_STRONG,.7) for x,y in pts]
 for x,y in pts[::2]:body.append(line(f'M {x} {y+1} L {x-1} 16',P.GREY_MID,.5))
 body += [dot(x,y,P.GREEN,1) for x,y in [(4,15.3),(10,13.2),(17,11.2)]]
 return P.svg('<title>Project points onto a reference plane</title>'+''.join(body))

def greedy_triangulation():
 pts=[(3,6),(9,3),(16,5),(21,9),(5,13),(12,11),(19,16),(8,20),(15,22)]
 tris=[(0,1,4),(1,2,5),(2,3,5),(4,5,6),(4,7,5),(5,7,8),(5,8,6)]
 body=[P.poly([pts[i] for i in t],P.BLUE_PALE) for t in tris]
 body += [line('M '+' L '.join(f'{x} {y}' for x,y in [pts[i] for i in t]),P.INK,.55) for t in tris]
 body += [dot(x,y,P.GREEN if i in (0,5) else P.BLUE_STRONG,.6) for i,(x,y) in enumerate(pts)]
 return P.svg('<title>Greedy triangulation joins neighboring surface samples</title>'+''.join(body))

def mls():
 defs=P.linear('smooth',[(0,P.BLUE_PALE),(1,P.BLUE_MID)])
 body=[line('M 2 17 L 5 13 L 8 19 L 11 8 L 14 16 L 17 5 L 22 12',P.GREY_DK,.65).replace('/>',' stroke-dasharray="1 1"/>'),
 P.fill('M 2 19 C 5 15 7 15 9 15 C 13 15 13 10 17 10 C 19 10 20 11 22 12 L 22 18 C 17 19 7 21 2 21 Z','url(#smooth)'),
 line('M 2 19 C 5 16 7 16 9 16 C 13 16 13 11 17 11 C 19 11 20 12 22 13',P.BLUE_STRONG,.9),P.arrow(5,5,18,5,P.GREEN,head=1.5,sw=.8)]
 return P.svg('<title>Moving least squares smoothing and optional upsampling</title>'+''.join(body),defs=defs)

def normal_curvature():
 defs=P.linear('patch',[(0,P.BLUE_PALE),(1,P.BLUE_MID)])
 body=[P.fill('M 2 17 Q 7 6 12 12 Q 17 18 22 5 L 22 19 Q 17 23 12 17 Q 7 11 2 21 Z','url(#patch)'),
 line('M 5 13 L 3.8 8 M 9 13 L 9 7 M 13 15 L 14 9 M 18 14 L 20 8',P.GREEN_DK,1),
 P.poly([(3.8,8),(2.7,9.8),(4.7,9.4)],P.GREEN_DK,outline=None),
 P.poly([(9,7),(8,8.8),(10,8.8)],P.GREEN_DK,outline=None),
 line('M 2 21 Q 7 11 12 17 Q 17 23 22 10',P.INK,.55)]
 return P.svg('<title>Surface normals and curvature samples</title>'+''.join(body),defs=defs)

def poisson():
 defs=P.radial('surface',[(0,P.GREEN_LT),(.7,P.GREEN),(1,P.GREEN_DK)],cx=.3,cy=.25,r=.9)
 pts=[(3,5),(6,3),(2,9),(4,13),(7,18),(3,20)]
 body=[cloud(pts,P.BLUE_STRONG,.7),line('M 4 4 L 6 2 M 2 8 L 1 6 M 4 12 L 5 10 M 6 17 L 8 16',P.BLUE_STRONG,.55),
 P.fill('M 8 10 H 11 V 8 L 14 11.5 L 11 15 V 13 H 8 Z',P.YELLOW,sw=.55),P.circle(18,11.5,5.5,'url(#surface)'),
 line('M 18 6 C 15 8 15 15 18 17 M 18 6 C 21 8 21 15 18 17 M 13 11.5 H 23',P.INK,.5)]
 return P.svg('<title>Poisson reconstructs a closed surface from oriented points</title>'+''.join(body),defs=defs)

def region_growing():
 pts=[(4,6),(8,4),(12,6),(16,4),(19,7),(6,10),(10,9),(14,10),(18,11),(4,15),(8,15),(12,14),(16,16),(20,17),(7,20),(12,20),(17,21)]
 selected={1,2,3,6,7,8,11,12,15};body=[]
 for i,(x,y) in enumerate(pts):body.append(dot(x,y,P.GREEN if i in selected else P.GREY_MID,.85))
 body.append(dot(12,9,P.YELLOW,.55))
 return P.svg('<title>Region growing expands a connected region from a seed point</title>'+''.join(body))

def sac():
 pts=[(3,16),(6,14.5),(9,13),(12,11.5),(15,10),(18,8.5),(21,7)];out=[(4,5),(9,19),(16,4),(20,17)]
 body=[line('M 2 16.5 L 22 6.5',P.BLUE_STRONG,1.25),line('M 2 18 L 22 8',P.BLUE_PALE,.6).replace('/>',' stroke-dasharray="1.1 1"/>'),
 cloud(pts,P.GREEN,.85),cloud(out,P.GREY_MID,.8),line('M 3 20 H 21',P.GREY_DK,.55)]
 return P.svg('<title>SAC segmentation fits a model and separates inliers from outliers</title>'+''.join(body))

def save():
 defs=P.linear('disk',[(0,P.BLUE_PALE),(1,P.BLUE_STRONG)])
 body=[P.fill('M 3 2 H 19 L 22 5 V 22 H 2 V 3 Z','url(#disk)'),P.rect(6,2,10,6,P.GREY_LT,r=.2),P.rect(8,2,2,5,P.GREY_DK,r=0),P.rect(5,14,14,8,PAPER,r=.35),P.rect(5,20,14,2,P.GREEN,r=0)]
 return P.svg('<title>Save PCL result</title>'+''.join(body),defs=defs)

def sift():
 body=[P.rect(2,2,20,20,P.GREY_LT,r=.4)]
 for x in [7,12,17]:body.append(line(f'M {x} 3 V 21',P.GREY_MID,.4))
 for y in [7,12,17]:body.append(line(f'M 3 {y} H 21',P.GREY_MID,.4))
 for x,y,r in [(7,7,2.8),(16.8,9.3,4),(11,16.8,2.1)]:body.extend([P.circle(x,y,r,'none',outline=P.BLUE_STRONG),dot(x,y,P.YELLOW,.8)])
 return P.svg('<title>SIFT detects scale-space keypoints</title>'+''.join(body))

def sor():
 body=[dot(x,y,P.BLUE_MID,.75) for x,y in [(4,7),(7,5),(8,9),(5,11),(11,7),(8,14),(11,12),(13,9),(10,17)]]
 body += [line('M 4 20 H 19',P.GREY_DK,.65),line('M 5 19 V 15 M 8 19 V 12 M 11 19 V 10 M 14 19 V 12 M 17 19 V 16',P.BLUE_STRONG,.6),dot(20,4,P.ROSE,1.15),line('M 18.7 2.7 L 21.3 5.3 M 21.3 2.7 L 18.7 5.3',P.RED,.9)]
 return P.svg('<title>Statistical outlier removal rejects isolated distant points</title>'+''.join(body))

def template_alignment():
 a=[(3,5),(5,9),(3,14),(7,6),(8,12),(6,18)];b=[(17,4),(20,7),(18,12),(22,15),(16,18),(20,21)]
 body=[line('M 5 5 L 19 5 M 7 9 L 18 12 M 5 14 L 21 15 M 7 18 L 18 19',P.GREY_MID,.5).replace('/>',' stroke-dasharray="1 1"/>'),cloud(a,P.BLUE_STRONG,.9),cloud(b,P.GREEN,.9),P.arrow(8,21,15,21,P.YELLOW,head=1.4,sw=.9)]
 return P.svg('<title>Align template and target point clouds</title>'+''.join(body))

def mincut():
 body=[P.fill('M 2 3 H 22 V 21 H 2 Z',P.BLUE_PALE),P.fill('M 3 4 H 10 V 20 H 3 Z',P.BLUE_MID,outline=None),P.fill('M 14 4 H 21 V 20 H 14 Z',P.GREEN_LT,outline=None),line('M 12 4 L 10.7 7 L 13.2 9 L 10.7 12 L 13.2 15 L 10.7 18 L 12 20',P.RED,1.5),dot(5.5,12,P.BLUE_STRONG,1.25),dot(18,12,P.GREEN,1.25)]
 return P.svg('<title>Minimum cut separates foreground and background on a graph</title>'+''.join(body))

def point_picking():
 body=[line('M 12 2 V 7 M 12 17 V 22 M 2 12 H 7 M 17 12 H 22',P.INK,1.1),P.circle(12,12,4.3,'none',outline=P.BLUE_STRONG),dot(12,12,P.YELLOW,1.05),P.fill('M 14 14 L 21.5 20 L 18.5 20.2 L 20 22 L 18 23 L 16.7 20.5 L 14.7 22 Z',P.GREY_LT,sw=.75)]
 return P.svg('<title>Pick a point with cursor and crosshair</title>'+''.join(body))

def don():
 body=[P.fill('M 3 13 C 5 8 9 7 12 11 C 15 7 19 8 21 13 L 18 19 H 6 Z',P.BLUE_PALE),line('M 4 14 L 7 10 L 10 15 M 14 15 L 17 10 L 20 14',P.INK,.55),P.arrow(8,7,5,3,P.GREEN,head=1.1,sw=.8),P.arrow(16,7,19,3,P.YELLOW,head=1.1,sw=.8),dot(12,12,P.BLUE_STRONG,.95)]
 return P.svg('<title>Difference of normals compares surface orientation at two scales</title>'+''.join(body))

def general_filter():
 defs=P.linear('funnel',[(0,P.BLUE_PALE),(1,P.BLUE_STRONG)])
 return P.svg('<title>General PCL filter funnel</title>'+P.fill('M 2 3 H 22 L 14.5 12 V 19 L 10 22 V 12 Z','url(#funnel)')+line('M 5 6 H 19 M 6.5 8.5 H 17.5',P.GREEN_DK,.6),defs=defs)

def bspline_curve():
 pts=[(3,18),(7,5),(12,19),(17,4),(21,16)]
 body=[line('M 3 18 L 7 5 L 12 19 L 17 4 L 21 16',P.GREY_MID,.65).replace('/>',' stroke-dasharray="1 1"/>'),line('M 3 18 C 7 18 5 10 9 9 C 13 8 13 16 16 12 C 19 8 18 5 21 4',P.BLUE_STRONG,1.2)]
 body += [dot(x,y,P.YELLOW,1) for x,y in pts]
 return P.svg('<title>B-spline curve fit from control points</title>'+''.join(body))

def bspline_surface():
 body=[P.fill('M 2 15 C 6 15 7 5 12 7 C 16 9 17 3 22 4 L 22 13 C 17 12 16 18 12 16 C 7 14 6 22 2 22 Z',P.BLUE_PALE)]
 for x,y in [(4,17),(9,14),(14,12),(19,8)]:body.append(line(f'M {x} {y+3} C {x+1} {y} {x} {y-5} {x+2} {y-7}',P.BLUE_STRONG,.48))
 for y in [7,10,13,16,19]:body.append(line(f'M 2 {y} Q 12 {y-3} 22 {y-7}',P.INK,.38))
 return P.svg('<title>B-spline surface fit with a curved control net</title>'+''.join(body))

def marching_cubes():
 body=[P.poly([(3,7),(11,3),(21,6),(13,10)],P.BLUE_PALE),P.poly([(3,7),(13,10),(13,21),(3,18)],P.BLUE_MID),P.poly([(13,10),(21,6),(21,17),(13,21)],P.BLUE_STRONG),P.fill('M 4 14 L 8 12 L 12 14 L 16 12 L 20 14 L 20 16 L 16 14 L 12 17 L 8 14 L 4 16 Z',P.YELLOW,sw=.5),line('M 8 12 L 8 14 M 12 14 L 12 17 M 16 12 L 16 14',P.INK,.5)]
 return P.svg('<title>Marching Cubes extracts an isosurface from voxel cells</title>'+''.join(body))

def fast_global_registration():
 a=[(3,7),(5,11),(3,16),(7,8),(8,14)];b=[(16,5),(19,8),(16,13),(21,15),(18,19)]
 body=[line('M 3 6 L 17 5 M 5 11 L 20 8 M 3 16 L 17 13 M 7 8 L 22 15',P.GREY_MID,.45).replace('/>',' stroke-dasharray="1 1"/>'),cloud(a,P.BLUE_STRONG,.82),cloud(b,P.GREEN,.82),P.arrow(7,21,18,21,P.YELLOW,head=1.6,sw=1.1)]
 return P.svg('<title>Fast global registration aligns clouds without initial pose</title>'+''.join(body))

# These two legacy registered illustrations have no C++/UI/runtime consumer.
# Keep their redesign in review archives, not in the shipped plugin QRC.
SOURCE_ONLY = {'pclSave.svg', 'pclPointPicking.svg'}

PLUGIN_ICONS={
 'pclConvexConcaveHull.png':('pclConvexConcaveHull.svg',convex_hull,'Convex/concave hull'),
 'pclCorrespondenceGrouping.png':('pclCorrespondenceGrouping.svg',correspondence,'Correspondence grouping'),
 'pclEuclideanClusterSegmentation.png':('pclEuclideanClusterSegmentation.svg',euclidean_clusters,'Euclidean clustering'),
 'pclProjectionFilter.png':('pclProjectionFilter.svg',projection_filter,'Projection onto plane'),
 'pclGreedyTriangulation.png':('pclGreedyTriangulation.svg',greedy_triangulation,'Greedy triangulation'),
 'pclMLSSmoothing.svg':('pclMLSSmoothing.svg',mls,'MLS smoothing and upsampling'),
 'pclNormalCurvature.svg':('pclNormalCurvature.svg',normal_curvature,'Normal and curvature estimation'),
 'pclPoissonReconstruction.png':('pclPoissonReconstruction.svg',poisson,'Poisson surface reconstruction'),
 'pclRegionGrowing.png':('pclRegionGrowing.svg',region_growing,'Region growing segmentation'),
 'pclSACSegmentation.png':('pclSACSegmentation.svg',sac,'Sample-consensus segmentation'),
 'pclSave.svg':('pclSave.svg',save,'Save result'),
 'pclSIFT.svg':('pclSIFT.svg',sift,'SIFT keypoint extraction'),
 'pclStatisticalOutliers.svg':('pclStatisticalOutliers.svg',sor,'Statistical outlier removal'),
 'pclTemplateAlignment.png':('pclTemplateAlignment.svg',template_alignment,'Template alignment'),
 'pclMinimumCut.png':('pclMinimumCut.svg',mincut,'Minimum-cut segmentation'),
 'pclPointPicking.svg':('pclPointPicking.svg',point_picking,'Point picking'),
 'pclDifferenceOfNormals.png':('pclDifferenceOfNormals.svg',don,'Difference of normals'),
 'pclGeneralFilter.png':('pclGeneralFilter.svg',general_filter,'General filtering'),
 'pclBSplineCurve.png':('pclBSplineCurve.svg',bspline_curve,'B-spline curve fitting'),
 'pclBSplineSurface.png':('pclBSplineSurface.svg',bspline_surface,'B-spline surface fitting'),
 'pclMarchingCubes.png':('pclMarchingCubes.svg',marching_cubes,'Marching-cubes isosurface extraction'),
 'pclFastGlobalRegistration.svg':('pclFastGlobalRegistration.svg',fast_global_registration,'Fast global registration'),
}

ICONS={
  'convex_concave_hull.png':('pclConvexHull.svg',convex_hull,'Convex/concave hull'),
  'correspondence_grouping.png':('pclCorrespondenceGrouping.svg',correspondence,'Correspondence grouping'),
  'EuclideanClusterSegmentation.png':('pclEuclideanClusters.svg',euclidean_clusters,'Euclidean clusters'),
  'filter_projection.png':('pclProjectionFilter.svg',projection_filter,'Projection filter'),
  'greedy_triangulation.png':('pclGreedyTriangulation.svg',greedy_triangulation,'Greedy triangulation'),
  'mls_smoothing.svg':('pclMLSSmoothing.svg',mls,'MLS smoothing'),
  'normal_curvature.svg':('pclNormalCurvature.svg',normal_curvature,'Normals and curvature'),
  'poisson.png':('pclPoissonReconstruction.svg',poisson,'Poisson reconstruction'),
  'Regiongrowing.png':('pclRegionGrowing.svg',region_growing,'Region growing'),
  'SAC_Segmentation.png':('pclSACSegmentation.svg',sac,'SAC segmentation'),
  'save.svg':('pclSave.svg',save,'Save result'),
  'sift.svg':('pclSIFT.svg',sift,'SIFT keypoints'),
  'sor_outlier_remover.svg':('pclSOR.svg',sor,'Statistical outliers'),
  'templateAlignment.png':('pclTemplateAlignment.svg',template_alignment,'Template alignment'),
  'mincut.png':('pclMinimumCut.svg',mincut,'Minimum cut'),
  'PointPicking.svg':('pclPointPicking.svg',point_picking,'Point picking'),
  'DoN.png':('pclDifferenceOfNormals.svg',don,'Difference of normals'),
  'generalFilter.png':('pclGeneralFilter.svg',general_filter,'General filter'),
  'bspline_curve.png':('pclBSplineCurve.svg',bspline_curve,'B-spline curve'),
  'bspline_surface.png':('pclBSplineSurface.svg',bspline_surface,'B-spline surface'),
  'MarchingCubeReconstruction.png':('pclMarchingCubes.svg',marching_cubes,'Marching Cubes'),
  'fastGlobalRegistration.svg':('pclFastGlobalRegistration.svg',fast_global_registration,'Fast global registration'),
}


# Fixed order is the review identifier; do not renumber between overview/boards.
LABELS = {
 'pclConvexConcaveHull.svg':'HULL',
 'pclCorrespondenceGrouping.svg':'CORR',
 'pclEuclideanClusterSegmentation.svg':'EC',
 'pclProjectionFilter.svg':'PROJ',
 'pclGreedyTriangulation.svg':'GP3',
 'pclMLSSmoothing.svg':'MLS',
 'pclNormalCurvature.svg':'N/CURV',
 'pclPoissonReconstruction.svg':'POISSON',
 'pclRegionGrowing.svg':'RG',
 'pclSACSegmentation.svg':'SAC',
 'pclSave.svg':'SAVE',
 'pclSIFT.svg':'SIFT',
 'pclStatisticalOutliers.svg':'SOR',
 'pclTemplateAlignment.svg':'(R,t)',
 'pclMinimumCut.svg':'MINCUT',
 'pclPointPicking.svg':'PICK',
 'pclDifferenceOfNormals.svg':'DoN',
 'pclGeneralFilter.svg':'FILTER',
 'pclBSplineCurve.svg':'B-CURVE',
 'pclBSplineSurface.svg':'B-SURF',
 'pclMarchingCubes.svg':'MC',
 'pclFastGlobalRegistration.svg':'FGR',
}

def _labeled(build,label):
    from .iconlabels import with_label
    return lambda: with_label(build(),label)

UNLABELED_ICONS = PLUGIN_ICONS.copy()
PLUGIN_ICONS = {key:(name,_labeled(build,LABELS[name]),title)
                for key,(name,build,title) in UNLABELED_ICONS.items()}

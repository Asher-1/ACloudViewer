"""Measured geometry for the two product logos.

All numbers are derived from the 1024x1024 high-resolution masters recovered
from the legacy SVG wrappers (git HEAD), not invented.  The measurement pass
lives in ``measure.py``; this module holds its conclusions as named constants so
the drawing code never hard-codes a magic number.

Coordinates are in the original master's 0..1024 space.  ``export.py`` maps them
into the icon canvas; nothing here assumes a particular output size.

Why measure instead of redraw by eye
------------------------------------
The brief was "refine the existing marks, do not replace them".  A logo's
equity lives in its proportions -- the exact taper of the C, the exact width of
the cube's faces.  Re-deriving those proportions from pixels is the only way to
refine without drifting.
"""

# ---------------------------------------------------------------------------
# ACloudViewer -- isometric cube, deliberately NOT a regular isometric.
# ---------------------------------------------------------------------------
# Measured extremes of the opaque region:
#   top vertex       (512,   20)
#   bottom vertex    (512, 1002)
#   right edge x = 924, spanning y 256..762
#   left  edge x =  99, spanning y 252..760
# A true isometric cube would want half-width = half-height * sin(60deg) = 425.3
# for a half-height of 491.  The artwork uses 412.5, i.e. 3% narrower: the cube
# reads taller and more assertive.  That optical choice is preserved.

ACV_CANVAS = 1024

ACV_CUBE = {
    "top": (512.0, 20.0),
    "tr": (924.0, 256.0),    # top-right
    "br": (924.0, 762.0),    # bottom-right
    "bot": (512.0, 1002.0),
    "bl": (99.0, 762.0),     # bottom-left
    "tl": (99.0, 252.0),     # top-left
    "center": (512.0, 511.0),  # the three faces meet here
}

# Core: the amber "solve" point sits on the cube centre, not the canvas centre.
ACV_CORE_CENTER = (505.0, 500.0)
ACV_CORE_RADIUS = 210.0       # outer reach of the warm bloom
ACV_CORE_HOT_RADIUS = 42.0    # the blown-out centre

# The dissolving plane: the left face is torn into a lattice of thin struts
# that run off the left edge and break into loose points.
ACV_DISSOLVE_ORIGIN = (99.0, 500.0)
ACV_DISSOLVE_EXTENT = (330.0, 480.0)   # (dx to the right, dy each way)

# Particle field: 80 isolated blobs of 20..600 px were measured in the master.
# Their positions are stored as normalised (x, y, r, kind) tuples and are
# emitted verbatim so the scatter keeps the original composition.
ACV_PARTICLE_COUNT = 80

# ---------------------------------------------------------------------------
# CloudViewer -- a tapered crescent, NOT a constant-width ring.
# ---------------------------------------------------------------------------
# Horizontal cuts through the master (y, list of opaque spans):
#   y=220  [(284, 738)]                    top arc, still closed
#   y=430  [(59,137),(165,282),(384,638),(865,963)]
#   y=500  [(20,64),(156,269),(356,666),(951,1002)]
#   y=570  [(162,276),(366,653)]
#   y=710  [(216,367),(661,801)]
#   y=780  [(277,745)]                     bottom, closed again
# The opening faces +x and sits high, so the crescent is asymmetric: the upper
# arm reaches further (to x=1002) than the lower one.
#
# Ring radii derived from the y=500 cut on the left side:
#   outer edge 511-156 = 355      inner edge 511-269 = 242
# and the beige disc spans x 356..666, i.e. radius 155 about x=511.

CV_CANVAS = 1024
CV_CENTER = (511.0, 511.0)
CV_R_OUTER = 355.0
CV_R_INNER = 242.0
CV_DISC_RADIUS = 155.0
CV_DISC_CENTER = (511.0, 510.0)

# Angular extent of the ribbon.  The gap is centred a little above +x, which is
# what makes the mark read as a C and not as an O.
CV_GAP_CENTER_DEG = 8.0
CV_GAP_HALF_DEG = 34.0

# Taper: the ribbon is thickest at the lower right, thinnest at the upper tip.
CV_TAPER = {"max_thickness": 118.0, "min_thickness": 74.0}

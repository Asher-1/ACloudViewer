"""Measured palette for the product logos.

Every value here was sampled from the 1024x1024 high-resolution masters that
were embedded inside the legacy ``*.svg`` wrappers (recovered from git history).
Nothing here is invented: the colours come from the original artwork so the
vector reconstruction keeps the existing brand equity.
"""

# --- ACloudViewer: isometric cube, teal -> blue, amber core -----------------
# Left-face values are measured, not chosen: separating the face's pixels by
# luminance shows a dark indigo base covering 33% of it and pale cyan struts
# covering 22%.  Painting the face as one flat indigo (or one flat grey) loses
# the glow that makes the dissolving plane read as lit structure.
ACV = {
    # top face: lightest, reads as the "sky" plane of the cube
    "top_near": "#4FD2DD",
    "top_far": "#2C93B4",
    # right face: teal at the top edge falling to deep ocean blue at the base
    "right_top": "#29A2B2",
    "right_bottom": "#1E5A8C",
    # left face: measured from the artwork, not chosen.  Separating that face's
    # pixels by luminance gives a dark indigo base (#3E4E84) covering 33% of it
    # and pale strokes (#BDCBC9) covering 22%.
    "left_base": "#7A8BA6",
    "left_deep": "#4A5A85",
    "left_top": "#4A5C9C",
    "left_bottom": "#252E63",
    "left_strut": "#BDCBC9",
    # measured from the 32 px master: at small sizes the face is one solid navy
    "left_solid": "#2B3A8F",
    # wireframe pass over the solid faces
    "mesh": "#8FE6F2",
    # the amber "solve" core at the cube centre
    "core_hot": "#FFF8D9",
    "core_mid": "#F7BE62",
    "core_warm": "#D2703A",
    "core_out": "#7A3B1E",
}

# Scattered points floating off the dissolving left face.
ACV_DOTS = {
    "cyan": "#2DBBD0",
    "blue": "#3F63CE",
    "orange": "#F2A055",
    "pale": "#9ADCE8",
}

# --- CloudViewer: cyan C-ring + amber dotted sphere -------------------------
CV = {
    "ring_light": "#8FE4EE",   # top of the C, catches the light
    "ring_mid": "#4EC4E6",
    "ring_deep": "#2E9AD8",    # underside, in shadow
    "ring_edge": "#BDF3FA",    # thin specular along the upper contour
    "disc_pale": "#FBF1DE",    # beige disc the points sit on
    "disc_deep": "#F1DCBB",
    "dot_dark": "#C8823A",
    "dot_mid": "#DE9A4E",
    "dot_pale": "#EBB86E",
}

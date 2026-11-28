"""Functional contracts for batch 9, not a claim of aesthetic acceptance."""
import unittest
import xml.etree.ElementTree as ET
from scripts.icon_hd.logo import meshtools as M, spatialtools as S
from scripts.icon_hd.logo.icon import fit, _png

CATALOG = {**M.ICONS, **S.ICONS}


class MeshSpatialTests(unittest.TestCase):
    def test_twenty_unique_module_named_destinations(self):
        self.assertEqual(len(CATALOG), 20)
        paths = [row[0] for row in CATALOG.values()]
        self.assertEqual(len(set(paths)), 20)
        self.assertEqual(sum(old.startswith('svg/main/') for old in CATALOG), 2)
        for path in paths:
            module, name = path.split('/')
            self.assertTrue(name.startswith('toolMain' if module == 'main' else module))
        self.assertFalse({'qt.png', 'donate.png', 'dbContainerSymbol.svg', 'frame.png'} & set(CATALOG))

    def test_twenty_icons_are_vector_only_with_ink_margins(self):
        for old, (_, build, _) in CATALOG.items():
            with self.subTest(icon=old):
                data = fit(build())
                node = ET.fromstring(data)
                tags = {n.tag.rsplit('}', 1)[-1] for n in node.iter()}
                self.assertFalse(tags & {'text', 'image', 'filter', 'script'})
                self.assertEqual(node.attrib['viewBox'], '0 0 24 24')
                im = _png(data, 192)
                box = im.getchannel('A').point(lambda a: 255 if a > 127 else 0).getbbox()
                self.assertGreater(min(box[:2]), 0)
                self.assertLess(max(box[2:]), 192)

    def test_subdivision_has_more_internal_edges_than_decimation(self):
        # Both icons share the same mesh boundary, not arbitrary different shapes.
        self.assertIn('M 2.5 8 L 9 2.5', M.decimate())
        self.assertIn('M 2.5 8 L 9 2.5', M.subdivide())
        self.assertGreater(M._mesh(dense=True).count(' L '), M._mesh(dense=False).count(' L '))

    def test_voxels_keep_one_representative_per_cell(self):
        node = ET.fromstring(M.voxel_sampling())
        circles = [n for n in node.iter() if n.tag.rsplit('}', 1)[-1] == 'circle']
        retained = [n for n in circles if n.attrib['fill'] == '#72BE44']
        self.assertEqual(len(circles), 24)  # eighteen muted inputs, six retained
        self.assertEqual(len(retained), 6)
        self.assertNotIn('<circle', M.octree())

    def test_octree_has_one_parent_and_eight_distinct_child_volumes(self):
        root = ET.fromstring(M.octree())
        faces = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'path'
                 and n.attrib.get('fill', 'none') != 'none']
        self.assertEqual(len(faces), 27)  # three faces each, nine volumes
        children = [n for n in faces if n.attrib['fill'] in {'#A7D28C','#72BE44','#4E8C2E'}]
        self.assertEqual(len(children), 24)

    def test_mesh_is_triangles_not_another_closed_polyhedron(self):
        root = ET.fromstring(M.mesh_menu())
        faces = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'path']
        nodes = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'circle']
        self.assertEqual(len(faces), 8)
        self.assertEqual(len(nodes), 9)
        self.assertTrue(all(n.attrib['d'].count(' L ') == 2 for n in faces))

    def test_rotation_centre_family_shares_rotation_ring(self):
        ring = S._rotation_ring()
        self.assertIn(ring, S.pick_center())
        self.assertIn(ring, S.reset_center())
        self.assertIn(ring, S.show_center())
        self.assertNotEqual(S.pick_center(), S.reset_center())
        self.assertNotEqual(S.reset_center(), S.show_center())

    def test_remove_last_point_keeps_list_and_single_minus(self):
        node = ET.fromstring(S.remove_last_point())
        rects = [n for n in node.iter() if n.tag.rsplit('}', 1)[-1] == 'rect']
        self.assertEqual(len(rects), 3)
        self.assertIn('M 15.8 18.5 H 20.2', S.remove_last_point())
        self.assertIn('last registration point', S.remove_last_point())


if __name__ == '__main__':
    unittest.main()

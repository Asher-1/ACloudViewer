"""Batch 8 contracts: distinct modules, retained symbols and shared resources."""
import unittest
import xml.etree.ElementTree as ET

from scripts.icon_hd.logo import contexttools, processingtools, exporttools
from scripts.icon_hd.logo.icon import fit, _png

CATALOG = {**contexttools.ICONS, **processingtools.ICONS, **exporttools.ICONS}


class ProcessingIconTests(unittest.TestCase):
    def test_twenty_unique_resources_without_reprocessing_database_container(self):
        self.assertEqual(len(CATALOG), 20)
        paths = [row[0] for row in CATALOG.values()]
        self.assertEqual(len(set(paths)), 20)
        self.assertNotIn('dbContainerSymbol.svg', CATALOG)
        self.assertIn('pqZoomToBox@2x.png', CATALOG)
        for path in paths:
            directory, filename = path.split('/')
            self.assertTrue(directory in {'main','misc','small','view','filter','annotation','surface'})
            if directory=='main':self.assertTrue(filename.startswith('toolMain'))
            self.assertTrue(filename.endswith('.svg'))

    def test_all_svg_are_fontless_and_vector_only(self):
        for old, (_, build, _) in CATALOG.items():
            with self.subTest(icon=old):
                root = ET.fromstring(fit(build()))
                self.assertEqual(root.attrib['viewBox'], '0 0 24 24')
                tags = {n.tag.rsplit('}', 1)[-1] for n in root.iter()}
                self.assertFalse(tags & {'text', 'image', 'script', 'filter'})

    def test_intersection_colors_only_the_overlap(self):
        im = _png(processingtools.intersect_selections(), 240)
        self.assertEqual(im.getpixel((110, 110)), (114, 190, 68, 255))
        self.assertEqual(im.getpixel((40, 40))[3], 0)
        self.assertEqual(im.getpixel((185, 185))[3], 0)

    def test_property_gear_has_a_real_central_hole(self):
        im = _png(fit(contexttools.property_editor()), 32)
        self.assertEqual(im.getpixel((16, 16))[3], 0)

    def test_exports_keep_same_down_arrow_but_distinct_palettes(self):
        a, b = ET.fromstring(exporttools.export_plane()), ET.fromstring(exporttools.export_cloud())
        paths = lambda root: [n.attrib['d'] for n in root.iter()
                              if n.tag.rsplit('}', 1)[-1] == 'path']
        self.assertEqual(paths(a), paths(b))
        colors = lambda root: [n.attrib['stop-color'] for n in root.iter()
                               if n.tag.rsplit('}', 1)[-1] == 'stop']
        self.assertEqual(colors(a), ['#D0E5F6', '#2896D3'])
        self.assertEqual(colors(b), ['#A7D28C', '#72BE44'])

    def test_icp_has_two_congruent_sample_sets_and_correspondences(self):
        root = ET.fromstring(processingtools.icp())
        circles = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'circle']
        self.assertEqual(len(circles), 12)
        reference, moving = circles[:6], circles[6:]
        for a, b in zip(reference, moving):
            self.assertAlmostEqual(float(b.attrib['cx'])-float(a.attrib['cx']), 3)
            self.assertAlmostEqual(float(b.attrib['cy'])-float(a.attrib['cy']), -2.6)
        self.assertTrue(any('stroke-dasharray' in n.attrib for n in root.iter()))
        self.assertLess(len(processingtools.icp()), 5000)

    def test_distance_ends_on_analytic_surface_not_sphere_centre(self):
        root = ET.fromstring(processingtools.cloud_primitive_distance())
        circles = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'circle']
        surface, endpoint = circles[:2]
        dx = float(endpoint.attrib['cx'])-float(surface.attrib['cx'])
        dy = float(endpoint.attrib['cy'])-float(surface.attrib['cy'])
        self.assertAlmostEqual((dx*dx+dy*dy)**.5, float(surface.attrib['r']), delta=.01)
        self.assertEqual(len(circles), 8)  # sphere, footpoint and six cloud samples
        self.assertFalse(any(n.tag.rsplit('}', 1)[-1] == 'rect' for n in root.iter()))

    def test_clean_keeps_point_cloud_samples_beside_broom(self):
        root = ET.fromstring(processingtools.clean_menu())
        circles = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'circle']
        self.assertEqual(len(circles), 9)
        colors = {n.attrib['stop-color'] for n in root.iter()
                  if n.tag.rsplit('}', 1)[-1] == 'stop'}
        self.assertTrue({'#D0E5F6', '#2896D3', '#C1A561', '#F3C017'} <= colors)
        self.assertLess(len(processingtools.clean_menu()), 3000)

    def test_filter_retains_three_sample_shapes_above_funnel(self):
        root = ET.fromstring(processingtools.filter_tools())
        # Mouth + two square input samples; the third sample is a diamond path.
        rects = [n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'rect']
        self.assertEqual(len(rects), 3)


if __name__ == '__main__':
    unittest.main()

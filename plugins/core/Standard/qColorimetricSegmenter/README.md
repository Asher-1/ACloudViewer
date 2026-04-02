# Colorimetric Segmenter (plugin)

Color-based segmentation of point clouds (see [ptrans](https://gitlab.univ-nantes.fr/E164955Z/ptrans)).

## Introduction

The Colorimetric Segmenter plugin provides three modes for segmenting point clouds based on color or scalar field values:

- **RGB mode**: Filter points by red, green, and blue channel ranges
- **HSV mode**: Filter points by hue, saturation, and value ranges
- **Scalar field mode**: Filter points by scalar field value range

This allows extraction of points matching specific color criteria (e.g. isolating vegetation by green color, or extracting objects of a particular hue).

## ACloudViewer CLI

Three separate commands are available, one for each segmentation mode:

### RGB segmentation

```bash
ACloudViewer -SILENT -O cloud.las -COLOR_SEG_RGB [OPTIONS] -SAVE_CLOUDS
```

| Token | Type | Description |
|-------|------|-------------|
| `-COLOR_SEG_RGB` | command | Segment by RGB range |
| `-R_MIN` / `-R_MAX` | int | Red channel range (0–255) |
| `-G_MIN` / `-G_MAX` | int | Green channel range (0–255) |
| `-B_MIN` / `-B_MAX` | int | Blue channel range (0–255) |

### HSV segmentation

```bash
ACloudViewer -SILENT -O cloud.las -COLOR_SEG_HSV [OPTIONS] -SAVE_CLOUDS
```

| Token | Type | Description |
|-------|------|-------------|
| `-COLOR_SEG_HSV` | command | Segment by HSV range |
| `-H_MIN` / `-H_MAX` | float | Hue range |
| `-S_MIN` / `-S_MAX` | float | Saturation range |
| `-V_MIN` / `-V_MAX` | float | Value range |

### Scalar field segmentation

```bash
ACloudViewer -SILENT -O cloud.las -COLOR_SEG_SCALAR [OPTIONS] -SAVE_CLOUDS
```

| Token | Type | Description |
|-------|------|-------------|
| `-COLOR_SEG_SCALAR` | command | Segment by scalar field value range |
| `-SCALAR_MIN` / `-SCALAR_MAX` | float | Scalar field value range |

## Build

```cmake
-DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON
```

## References

- Source: [gitlab.univ-nantes.fr/E164955Z/ptrans](https://gitlab.univ-nantes.fr/E164955Z/ptrans)
- CloudCompare wiki: [Colorimetric Segmenter (plugin)](https://www.cloudcompare.org/doc/wiki/index.php/Colorimetric_Segmenter_(plugin))

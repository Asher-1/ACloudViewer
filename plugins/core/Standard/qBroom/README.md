# qBroom — CEA Virtual Broom

![qBroom icon](https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qBroom/images/qBroom.png)

Interactive cloud cleaning tool: sweep a virtual broom across a point cloud
to remove unwanted points, with a multi-level undo stack and an explicit
Apply/Cancel step. Port of the CloudCompare
[Virtual Broom plugin](https://www.cloudcompare.org/doc/wiki/index.php?title=Virtual_broom_(plugin))
by Wesley Grimes (Collision Engineering Associates).

## Usage

1. Select **one point cloud** in the DB tree — the action enables only for a
   single cloud selection.
2. **Plugins → CEA Virtual Broom** — a disclaimer dialog shows on first use.
3. In the broom dialog, sweep points away with the mouse; each stroke is
   recorded, so **Undo** / **Undo 10** restore previous states.
4. **Apply** commits the cleaned cloud; closing the dialog without applying
   leaves the original cloud untouched.

## Build

```bash
-DPLUGIN_STANDARD_QBROOM=ON
```

Registered via `AddPlugin` in `plugins/cmake/Plugins.cmake`; the option is
documented in
[BUILD.md](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md).

## Credits

- Author / maintainer: Wesley Grimes
  ([Collision Engineering Associates](https://cea-az.com/))
- Upstream documentation: the CloudCompare wiki page linked above
- Source files carry the MIT license notice (`SPDX-License-Identifier: MIT`).

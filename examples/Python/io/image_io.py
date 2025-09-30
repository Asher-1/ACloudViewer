# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    img_data = cv3d.data.JuneauImage()
    print(f"Reading image from file: Juneau.jpg stored at {img_data.path}")
    img = cv3d.io.read_image(img_data.path)
    print(img)
    print("Saving image to file: copy_of_Juneau.jpg")
    cv3d.io.write_image("copy_of_Juneau.jpg", img)

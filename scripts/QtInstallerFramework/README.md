# Windows

1，put application data in : [data](./windows/packages/com.vendor.product/data)

2, modify [config.xml](./windows/config/config.xml) and [package.xml](./windows/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE](./windows) 

4, binarycreator.exe -c config/config.xml -p packages CloudViewer-3.8.0-2021-10-10-win-amd64.exe

5, binarycreator.exe -c config/config.xml -p packages ErowCloudViewer-3.8.0-2021-10-10-win-amd64.exe


# Linux
1，put application data in: [data for ErowCloudViewer](./linux/ErowCloudViewer/packages/com.vendor.product/data) or [data for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/data)

2, modify [config.xml for ErowCloudViewer](./linux/ErowCloudViewer/config/config.xml) or  [config.xml for CloudViewer](./linux/CloudViewer/config/config.xml) and [package.xml for ErowCloudViewer](./linux/ErowCloudViewer/packages/com.vendor.product/meta/package.xml) or [package.xml for CloudViewer](./linux/CloudViewer/packages/com.vendor.product/meta/package.xml) 

3, cd [WORKSPACE for CloudViewer](./linux/CloudViewer) && binarycreator -c config/config.xml -p packages CloudViewer-3.8.0-2021-10.10-ubuntu1804-amd64.run

4, cd [WORKSPACE for ErowCloudViewer](./linux/ErowCloudViewer) && binarycreator -c config/config.xml -p packages ErowCloudViewer-3.8.0-2021-10.10-ubuntu1804-amd64.run
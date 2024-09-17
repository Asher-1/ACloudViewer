#!/bin/bash
if [ -x "$(command -v sw_vers)" ]; then
  echo 'MacOS sed call'
  sed -i "" "s/..\/data\/4050\/lut_apa\/configs/.\/4050\/lut_apa\/configs/g" $1
else
  echo 'Linux sed call'
  sed -i "s/..\/data\/4050\/lut_apa\/configs/.\/4050\/lut_apa\/configs/g" $1
fi

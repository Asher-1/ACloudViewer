#!/bin/sh
appname=`basename $0 | sed s,\.sh$,,`  

dirname=`dirname $0`
tmp="${dirname#?}"  

if [ "${dirname%$tmp}" != "/" ]; then
	dirname=$PWD/$dirname
fi
LD_LIBRARY_PATH="$dirname:$dirname/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Bundled CUDA runtime (AICore_BUNDLE_CUDA_RUNTIME=ON): driver-only deployment.
if [ -d "$dirname/lib/cuda-runtime" ]; then
	LD_LIBRARY_PATH="$dirname/lib/cuda-runtime:$LD_LIBRARY_PATH"
fi

# Otherwise optional ggml CUDA modules use the target machine's toolkit.
# Preserve an explicit user path first, then add the first installed runtime.
# Do NOT use "set --" here: it would replace "$@" and make ACloudViewer try to
# load libcudart.so.* as point-cloud files on startup (suffix "0", "89", …).
for _cuda_root in "${CUDA_PATH:-}" "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
	[ -d "$dirname/lib/cuda-runtime" ] && break
	[ -n "$_cuda_root" ] || continue
	for _cuda_lib in "$_cuda_root/lib64" \
		"$_cuda_root/targets/$(uname -m)-linux/lib" \
		"$_cuda_root/targets/x86_64-linux/lib"; do
		[ -d "$_cuda_lib" ] || continue
		for _cudart in "$_cuda_lib"/libcudart.so.*; do
			[ -e "$_cudart" ] || continue
			LD_LIBRARY_PATH="$_cuda_lib:$LD_LIBRARY_PATH"
			break 3
		done
	done
done
export LD_LIBRARY_PATH

export PYTHONPATH=$dirname/plugins/Python

$dirname/$appname "$@"

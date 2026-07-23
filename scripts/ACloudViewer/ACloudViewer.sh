#!/bin/sh
appname=`basename $0 | sed s,\.sh$,,`  

dirname=`dirname $0`
tmp="${dirname#?}"  

if [ "${dirname%$tmp}" != "/" ]; then
	dirname=$PWD/$dirname
fi
LD_LIBRARY_PATH="$dirname:$dirname/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Optional ggml CUDA modules use the target machine's toolkit. Preserve an
# explicit user path first, then add the first installed toolkit runtime found.
for _cuda_root in "${CUDA_PATH:-}" "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
	[ -n "$_cuda_root" ] || continue
	for _cuda_lib in "$_cuda_root/lib64" \
		"$_cuda_root/targets/$(uname -m)-linux/lib" \
		"$_cuda_root/targets/x86_64-linux/lib"; do
		[ -d "$_cuda_lib" ] || continue
		set -- "$_cuda_lib"/libcudart.so.*
		if [ -e "$1" ]; then
			LD_LIBRARY_PATH="$_cuda_lib:$LD_LIBRARY_PATH"
			break 2
		fi
	done
done
export LD_LIBRARY_PATH

export PYTHONPATH=$dirname/plugins/Python

$dirname/$appname "$@"

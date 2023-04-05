#!/bin/bash
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$script_path/colmap gui

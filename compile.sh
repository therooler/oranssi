#!/usr/bin/env bash
echo "Compiling Oranssi"
export PATH=/pkgs/anaconda3/bin:$PATH
echo $PATH
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate pennylane
python setup.py install

set -e
# for the parameter accepted values, see demo_main.m header
METHOD=${METHOD:-1}
DATA=${DATA:-1}
MODEL=${MODEL:-1}
BURNITERS=${BURNITERS:-8}  # paper mentions 8 iterations, they're probably only doing burn iters as they're only interested in last sample
SAMPLINGITERS=${SAMPLINGITERS:-0}
IT=${IT:-400000}
/home/tools/matlab/mathworks_r2019b/bin/glnxa64/MATLAB -nodisplay -nosplash -nodesktop -prefersoftwareopengl \
-r "try demo_main($METHOD, $DATA, $MODEL, $BURNITERS, $SAMPLINGITERS, $IT); catch; end; exit;"

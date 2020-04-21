set -e
# for the parameter accepted values, see demo_main.m header
METHOD=${METHOD:-1}
DATA=${DATA:-1}
MODEL=${MODEL:-1}
BURNITERS=${BURNITERS:-0}  # paper doesn't mention anything specific here, IIUC
SAMPLINGITERS=${SAMPLINGITERS:-8}  # paper mentions 8 here
/home/tools/matlab/mathworks_r2019b/bin/glnxa64/MATLAB -nodisplay -nosplash -nodesktop -prefersoftwareopengl \
-r "try demo_main($METHOD, $DATA, $MODEL, $BURNITERS, $SAMPLINGITERS); catch; end; exit;"

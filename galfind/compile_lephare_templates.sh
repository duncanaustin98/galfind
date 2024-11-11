#!/bin/bash

CONFIG_PATH=$1
TYPE=$2
FILTERS=$3
FILT_NAME=$4
BINARY_NAME=$5
SAVE_SUFFIX=$6

OUT_NAME=${BINARY_NAME}_${FILT_NAME}${SAVE_SUFFIX}
FILT_NAME="$FILT_NAME.filt"

# assume binaries and templates have already been compiled

if [ "$TYPE" == "STAR" ]; then
    # make stellar template set
    $LEPHAREDIR/source/mag_star -c $CONFIG_PATH -FILTER_LIST $FILTERS -FILTER_FILE $FILT_NAME -STAR_LIB_OUT $OUT_NAME
elif [ "$TYPE" == "QSO" ]; then
    # make QSO template set
    $LEPHAREDIR/source/mag_gal -t Q -c $CONFIG_PATH -FILTER_LIST $FILTERS -FILTER_FILE $FILT_NAME -QSO_LIB_OUT $OUT_NAME
elif [ "$TYPE" == "GAL" ]; then
    # make galaxy template set (takes by far the longest)
    $LEPHAREDIR/source/mag_gal -t G -c $CONFIG_PATH -FILTER_LIST $FILTERS -FILTER_FILE $FILT_NAME -GAL_LIB_OUT $OUT_NAME
fi
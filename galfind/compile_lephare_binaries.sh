#!/bin/bash

CONFIG_PATH=$1
TYPE=$2
INPUT_PATH=$3
BINARY_NAME=$4
AGES_PATH=$5

if [ "$TYPE" == "STAR" ]; then
    # compile stellar binaries
    $LEPHAREDIR/source/sedtolib -t S -c ${CONFIG_PATH} -STAR_SED ${INPUT_PATH} -STAR_LIB ${BINARY_NAME}
elif [ "$TYPE" == "QSO" ]; then
    # compile QSO binaries
    $LEPHAREDIR/source/sedtolib -t Q -c ${CONFIG_PATH} -QSO_SED ${INPUT_PATH} -QSO_LIB ${BINARY_NAME}
elif [ "$TYPE" == "GAL" ]; then
    if [ "$AGES_PATH" == "" ]; then
        # compile galaxy binaries
        $LEPHAREDIR/source/sedtolib -t G -c ${CONFIG_PATH} -GAL_SED ${INPUT_PATH} -GAL_LIB ${BINARY_NAME} -AGE_RANGE 0.,13.e9
    else
        # compile galaxy binaries with ages
        $LEPHAREDIR/source/sedtolib -t G -c ${CONFIG_PATH} -GAL_SED ${INPUT_PATH} -GAL_LIB ${BINARY_NAME} -SEL_AGE ${AGES_PATH}
    fi
fi





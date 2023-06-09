#!/bin/bash

DATA_DIR="/nvme/scratch/work/austind"
export SURVEY=$1
INSTRUMENT=$2
FILTERS=("f115W" "f150W" "f200W" "f277W" "f356W" "f410M" "f444W")  #("f435w" "f606w" "f775w" "f814w" "f850lp") # ceers
export FORCED_PHOT_FILT=$3 #"f444W"
export VERSION=$4
PIXEL_SCALE=0.03
MAG_ZEROPOINT=28.08
CALC_SEG_MAPS=true
RUN_FORCED_PHOT=true
MATCH_SEX_CATS=true
echo $VERSION

export OUTPUT_CAT_DIR=${DATA_DIR}/SExtractor/${INSTRUMENT}/${SURVEY}
mkdir -p $OUTPUT_CAT_DIR

# calculate appropriate segmentation maps
filter_length=${#FILTERS[@]}
if $CALC_SEG_MAPS
then
    for ((j=0; j<${filter_length}; j++));
    do
        export FILT=${FILTERS[$j]}
        INPUT_IMAGE_PATH=($(python "/nvme/scratch/work/austind/SExtractor/return_survey_image_path.py"))
        echo ${INPUT_IMAGE_PATH}
        OUTPUT_CAT_PATH=${OUTPUT_CAT_DIR}/${SURVEY}_${FILT}_${FILT}_sel_cat_${VERSION}
        echo ${OUTPUT_CAT_PATH}
        sex ${INPUT_IMAGE_PATH}'[1]' -c ${DATA_DIR}/SExtractor/nircam.sex -WEIGHT_TYPE MAP_RMS -PIXEL_SCALE ${PIXEL_SCALE} -WEIGHT_IMAGE ${INPUT_IMAGE_PATH}'[2]' \
        -CATALOG_NAME ${OUTPUT_CAT_PATH}.fits -MAG_ZEROPOINT ${MAG_ZEROPOINT} \
        -CHECKIMAGE_NAME ${OUTPUT_CAT_PATH}_seg.fits,${OUTPUT_CAT_PATH}_bkg.fits \
        -CHECKIMAGE_TYPE SEGMENTATION,-BACKGROUND
    done
fi

# NO NEED TO RE-RUN FORCED_PHOT_BANDxFORCED_PHOT_BAND AGAIN!
# perform forced photometry for the catalogues
export FILT=${FORCED_PHOT_FILT}
FORCED_PHOT_IMAGE_PATH=($(python "/nvme/scratch/work/austind/SExtractor/return_survey_image_path.py"))
if $RUN_FORCED_PHOT
then
    for ((j=0; j<${filter_length}; j++));
    do
        export FILT=${FILTERS[$j]}
        INPUT_IMAGE_PATH=($(python "/nvme/scratch/work/austind/SExtractor/return_survey_image_path.py"))
        OUTPUT_CAT_PATH=${OUTPUT_CAT_DIR}/${SURVEY}_${FILT}_${FORCED_PHOT_FILT}_sel_cat_${VERSION}
        sex ${FORCED_PHOT_IMAGE_PATH}'[1]' ${INPUT_IMAGE_PATH}'[1]' -c ${DATA_DIR}/SExtractor/nircam.sex -WEIGHT_TYPE MAP_RMS -PIXEL_SCALE ${PIXEL_SCALE} -WEIGHT_IMAGE ${FORCED_PHOT_IMAGE_PATH}'[2]',${INPUT_IMAGE_PATH}'[2]' \
        -CATALOG_NAME ${OUTPUT_CAT_PATH}.fits -MAG_ZEROPOINT ${MAG_ZEROPOINT} \
        -CHECKIMAGE_TYPE NONE
    done
fi

# produce a matched catalogue
if $MATCH_SEX_CATS
then
    python "/nvme/scratch/work/austind/SExtractor/match_sex_cats.py"
    # copy to Catalogue folder
    NEW_CAT_DIR="/nvme/scratch/work/austind/Catalogues/${SURVEY}"
    CAT_NAME="${SURVEY}_MASTER_Sel-${FORCED_PHOT_FILT}_${VERSION}.fits"
    mkdir -p $NEW_CAT_DIR
    cp "${OUTPUT_CAT_DIR}/${CAT_NAME}" "${NEW_CAT_DIR}/${CAT_NAME}"
fi



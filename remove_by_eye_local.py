#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:20:05 2023

@author: u92876da
"""

# remove_by_eye_local.py

"""
1) messy cutout (non-blob like)
2) hot pixel/cosmic ray
3) wisp issue
4) ICL contamination
5) bright nearby source

"""

# may well be the wrong MACS-0416 cutouts
# GLASS done
# SMACS-0723 some things appear to have not dropped out, couple of high-z 'plusses' - cluster not masked
# CLIO 5271 doesnt appear to drop-out
# cluster issue here
# remove_IDs_reasons = {
# "NGDEEP": {11148: [2], 13723: [2], 13938: [1, 2], 14796: [2], 17157: [2]}, \
# "MACS-0416": {6624: [1], 7933: [1], 9296: [1], 9937: [1]}, \
# "GLASS": {}, \
# "SMACS-0723": {519: [1], 5964: [1]}, \
# "El-Gordo": {604: [2, 5], 2307: [1, 2], 4208: [1], 4352: [1, 5], 9218: [1, 4]}, \
# "CLIO": {3752: [1], 3974: [1, 5], 6049: [2], 7752: [5]}, \
# "CEERSP1": {7953: [2]}, \
# "CEERSP2": {4532: [1, 5]}, \
# "CEERSP3": {999: [5], 4899: [1], 6682: [1, 5], 8134: [1], 8428: [1, 5]}, \
# "CEERSP4": {}, \
# "CEERSP5": {}, \
# "CEERSP6": {}, \
# "CEERSP7": {}, \
# "CEERSP8": {}, \
# "CEERSP9": {}, \
# "CEERSP10": {}
# }

# CEERSP1
CEERSP1_reasons = {}
CEERSP1_remove_IDs = []
CEERSP1_dodgy_IDs = []
CEERSP1_close_pairs = []

# CEERSP2
CEERSP2_reasons = {}
CEERSP2_remove_IDs = []
CEERSP2_dodgy_IDs = []
CEERSP2_close_pairs = []

# CEERSP3
CEERSP3_reasons = {}
CEERSP3_remove_IDs = []
CEERSP3_dodgy_IDs = []
CEERSP3_close_pairs = []

# CEERSP4
CEERSP4_reasons = {}
CEERSP4_remove_IDs = []
CEERSP4_dodgy_IDs = []
CEERSP4_close_pairs = []

# CEERSP5
CEERSP5_reasons = {}
CEERSP5_remove_IDs = []
CEERSP5_dodgy_IDs = []
CEERSP5_close_pairs = []

# CEERSP6
CEERSP6_reasons = {}
CEERSP6_remove_IDs = []
CEERSP6_dodgy_IDs = []
CEERSP6_close_pairs = []

# CEERSP7
CEERSP7_reasons = {}
CEERSP7_remove_IDs = []
CEERSP7_dodgy_IDs = []
CEERSP7_close_pairs = []

# CEERSP8
CEERSP8_reasons = {}
CEERSP8_remove_IDs = []
CEERSP8_dodgy_IDs = []
CEERSP8_close_pairs = []

# CEERSP9
CEERSP9_reasons = {}
CEERSP9_remove_IDs = []
CEERSP9_dodgy_IDs = []
CEERSP9_close_pairs = []

# CEERSP10
CEERSP10_reasons = {}
CEERSP10_remove_IDs = []
CEERSP10_dodgy_IDs = []
CEERSP10_close_pairs = []
 
# CLIO
CLIO_reasons = {3752: "clumpy", 5271: "detected in F090W", 6049: "dithered hot pixel", 7752: "potential F090W detection"}
CLIO_remove_IDs = [6049]
CLIO_dodgy_IDs  = [3752, 5271, 7752]

# El-Gordo
El_Gordo_reasons = {604: "near bright source", 4352: "hot pixel/blended", 6809: "potential F090W detection", 9218: "blended/ICL contamination?"}
El_Gordo_remove_IDs = [4352]
El_Gordo_dodgy_IDs = [604, 2307, 6809, 9218]
El_Gordo_close_pairs = [6908, 9616]

# GLASS
GLASS_reasons = {1908: "Looks legit but z=20.6? Brown dwarf or EM line combination?", 3404: "potential F090W detection"}
GLASS_dodgy_IDs = [1908, 3404]
GLASS_close_pairs = [11800, 13962, 17154]

# MACS-0416
MACS_0416_reasons = {5709: "clumpy", 6079: "clumpy + potential blue detections", 6996: "Detected in F090W", 7024: "dithered hot pixel", \
                     10506: "ICL contamination", 14999: "F090W detection", 17148: "hot pixel?", 17197: "near bright source + blue detection?"}
MACS_0416_remove_IDs = [7024, 17148]
MACS_0416_dodgy_IDs = [5709, 6079, 6624, 6996, 9296, 9937, 10506, 11343, 14999, 17197]
MACS_0416_close_pairs = [13857, 20481]

# SMACS-0723
SMACS_0723_reasons = {519: "clumpy", 692: "near bright source", 1273: "Adams+2022 merger candidate", 2735: "near bright source", 4970: "blended", \
                      8041: "F090W detection", 5903: "dithered hot pixel?", 9818: "F090W detection", 10132: "strange LW morphology"}
SMACS_0723_dodgy_IDs = [519, 692, 2735, 4970, 8041, 5903, 9818, 10132]
SMACS_0723_close_pairs = [831, 1273, 6636]

# NGDEEP
NGDEEP_reasons = {11148: "F115W hot pixel/cosmic ray", 13723: "F115W+F200W hot pixel/cosmic ray", 13938: "extended+clumpy+F200W cosmic ray", \
                  14796: "F155W contamination, cosmic ray?", 17157: "F115W+F200W hot pixel/cosmic ray"}
NGDEEP_remove_IDs = [13938]
NGDEEP_dodgy_IDs = [11148, 13723, 14796, 17157]
NGDEEP_close_pairs = [939, 2330, 2430, 2897, 4250, 7183, 11441, 13117, 16486]

# NEP-1
NEP_1_reasons = {305: "clumpy+noisy", 808: "strange SED", 824: "strange SED", 993: "clumpy + near bright source", 1221: "slightly offset F606W detection", \
                 9453: "bright nearby contaminating source", 111257: "clumpy", 11490: "potential blue detection", 11917: "strange LW morphology", \
                 13022: "near bright source + contaminated", 16857: "potential F606W detection", 17160: "strange LW morhpology", 18668: "blue SW detections", \
                 19060: "strange LW morphology + near bright source"}
NEP_1_remove_IDs = [808, 824, 11917, 17160, 19060]
NEP_1_dodgy_IDs = [305, 993, 1221, 9453, 11257, 11490, 13022, 13199, 14185, 14390, 16857, 18668]
NEP_1_close_pairs = [2494, 6470, 6613, 15222, 16227, 18432]

# NEP-2
NEP_2_reasons = {490: "strange SED + LW morphology", 694: "strange SED + LW morphology", 785: "strange SED + LW morphology", 1245: "faint F606W detection", \
                 1267: "strange SED + LW morphology", 1290: "strange SED + LW morphology", 2679: "LW wide band hot pixels", 3797: "severely blended", \
                 3980: "large F115W excess flux + minor F606W detection", 4231: "faint F090W detection", 5107: "blended", 5906: "clumpy", 10708: "clumpy", \
                 12443: "stange LW morphology and unexpected F410M non-detection", 12524: "Yellow F606W", 13566: "strange SED + LW morphology", \
                 14422: "strong F606W detection", 14846: "F115W + F150W detections", 15249: "high-z secondary solution", 16330: "possible SW detections", \
                 16562: "clumpy", 18240: "strange SED + LW morphology"}
NEP_2_remove_IDs = [490, 694, 785, 1267, 1290, 2679, 3797, 12443, 13566, 14422, 18240]
NEP_2_dodgy_IDs = [1245, 3980, 4231, 5107, 5906, 10198, 10708, 12524, 14422, 14846, 15249, 16330, 16562]
NEP_2_close_pairs = [6227, 13924, 13935, 17134]

# NEP-3
NEP_3_reasons = {506: "strange SED + LW morphology", 723: "strange SED + LW morphology", 1072: "strange SED + LW morphology", 4149: "F356W hot pixel", \
                 7559: "severely blended", 9992: "strong F606W detection", 10295: "strong F606W detection", 10791: "strong F606W detection", \
                 11630: "several bright contaminants", 12107: "slight F606W detection", 12431: "blended", 12898: "strange SED + LW morphology", \
                 13381: "slight F606W detection", 14255: "strange SED + LW morphology", 16558: "slight F606W detection", 18439: "strong F606W detection", \
                 19023: "slight F606W detection", 19196: "slight F606W detection"}
NEP_3_remove_IDs = [506, 723, 1072, 4149, 7559, 11630, 12898, 14255, 18439]
NEP_3_dodgy_IDs = [9992, 10141, 10791, 11179, 12107, 12431, 13381, 16558, 19023, 19196]
NEP_3_close_pairs = [1702, 3188, 3447, 4724, 5200, 6122, 7168, 13984, 14540, 16814, 16932, 18740, 19289, 19339]

# NEP-4
NEP_4_reasons = {787: "strange SED + LW morphology", 968: "slight F606W detection", 994: "strange SED + LW morphology", 1527: "strong F606W detection", \
                 2811: "clumpy", 6560: "slight F606W detection", 8286: "slight F606W detection", 8478: "off-centred F090W + F606W detection", 10333: "diffuse cutouts", \
                 10619: "close contaminant", 10803: "offset F606W", 11150: "slight F606W detection", 11750: "very diffuse", 12684: "strange SED + LW morphology", \
                 13194: "very strong F606W detection", 14034: "strange LW morphology", 14365: "slight F606W detection", 15558: "slight F606W detection", \
                 17371: "strange LW morphology", 17627: "very diffuse", 18207: "strange SED + LW morphology", 18846: "strange SED + LW morphology", 19353: "Faint F606W detection"}
NEP_4_remove_IDs = [787, 994, 8478, 10333, 12684, 13194, 14034, 18207, 18846]
NEP_4_dodgy_IDs = [968, 1527, 2811, 6560, 8286, 10619, 10803, 11150, 11750, 14365, 15558, 17627, 19353]
NEP_4_close_pairs = [1673, 1750, 8740, 10781]



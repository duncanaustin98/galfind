# make_brown_dwarf_fit_files.py

import os

from galfind import config

def get_dir(templates):
    return f"{config['Brown_Dwarf_Fitting']['TEMPLATES_DIR']}/{templates}"

def make_templates_file(templates, template_names):
    dir = get_dir(templates)
    out_name = f"{config['Brown_Dwarf_Fitting']['CONFIG_FILENAME']}.txt"
    out_path = f"{dir}/{out_name}"
    # delete old file should it exist
    if os.path.exists(out_path):
        os.remove(out_path)
    # write each template to outfile
    with open(out_path, "a") as txt_file:
        for name in template_names:
            txt_file.write(name)
            if name != template_names[-1]:
                txt_file.write("\n")
        txt_file.close()

def make_sonora_cholla_in(temp_arr, log_gs_arr, log_kzzs_arr):
    dir = get_dir("sonora_cholla")
    templates_size = int(len(temp_arr) * len(log_gs_arr) * len(log_kzzs_arr))
    template_names = [f"{dir}/logkzz{log_kzz}/{log_g}g/{temp}K_{log_g}g_logkzz{log_kzz}_resample.dat" \
        for log_kzz in log_kzzs_arr for log_g in log_gs_arr for temp in temp_arr]
    template_names = [template_name.replace(f"{dir}/", "") for template_name in template_names]
    assert len(template_names) == templates_size, print(f"{len(template_names)=}!={templates_size=}")
    make_templates_file("sonora_cholla", template_names)

def make_sonora_bobcat_in(temp_arr, log_g_arr, metallicity_arr, co_ratio_arr):
    dir = get_dir("sonora_bobcat")
    templates_size = int(len(temp_arr) * len(log_g_arr) * len(metallicity_arr))
    template_names = [f"{dir}/m{metallicity}/{log_g}g/sp_t{temp}g{log_g}nc_m{metallicity}_resample.dat" \
        for metallicity in metallicity_arr for log_g in log_g_arr for temp in temp_arr]
    print(f"{dir}/m{metallicity_arr[0]}/{log_g_arr[0]}g/sp_t{temp_arr[0]}g{log_g_arr[0]}nc_m{metallicity_arr[0]}_resample.dat")
    if "0.0" in metallicity_arr and 1000 in log_g_arr:
        templates_size += int(len(co_ratio_arr) * len(temp_arr))
        template_names += [f"{dir}/m0.0/1000g/co{co_ratio}/sp_t{temp}g1000nc_m0.0_resample.dat" \
            for temp in temp_arr for co_ratio in co_ratio_arr]
    print(template_names)
    template_names = [template_name.replace(f"{dir}/", "") for template_name in template_names]
    assert len(template_names) == templates_size, print(f"{len(template_names)=}!={templates_size=}")
    make_templates_file("sonora_bobcat", template_names)

if __name__ == "__main__":
    # make sonora_cholla templates file
    temp_arr = [500, 550, 600, 650, 700, 750, 800, 850, \
        900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
    log_g_arr = [31, 56, 100, 178, 316, 562, 1000, 1780, 3162]
    log_kzz_arr = [2, 4, 7]
    make_sonora_cholla_in(temp_arr, log_g_arr, log_kzz_arr)

    # make sonora_bobcat templates file
    temp_arr = [200, 225, 250, 275, 300, 325, 350, 375, 400, \
        425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, \
        800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, \
        1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
    log_g_arr = [10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160]
    metallicity_arr = ["-0.5", "0.0", "+0.5"]
    co_ratio_arr = [0.5, 1.5] # only relevant for log(g)=1000, metallicity="0.0"
    make_sonora_bobcat_in(temp_arr, log_g_arr, metallicity_arr, co_ratio_arr)




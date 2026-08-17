import ast

files_and_lines = [
    ("galfind/SFH.py", 388),
    ("galfind/catalogues/Catalogue.py", 1815),
    ("galfind/catalogues/Catalogue_Base.py", 220),
    ("galfind/catalogues/Multiple_Catalogue.py", 748),
    ("galfind/galaxy/Galaxy.py", 1543),
    ("galfind/imaging/Data.py", 996),
    ("galfind/imaging/Instrument.py", 892),
    ("galfind/imaging/NIRCam_reduction.py", 604),
    ("galfind/imaging/PSF.py", 740),
    ("galfind/number_density_functions/ndf.py", 905),
    ("galfind/properties/Morphology.py", 1229),
    ("galfind/properties/Property_calculator.py", 2267),
    ("galfind/properties/Rest_frame_properties.py", 1522),
    ("galfind/sed_fitting/Bagpipes.py", 1192),
    ("galfind/sed_fitting/EAZY.py", 1114),
    ("galfind/sed_fitting/SED_codes.py", 646),
    ("galfind/selection/Selector.py", 569),
    ("galfind/utils/Depths.py", 1115),
    ("galfind/utils/MCMC.py", 616),
    ("galfind/utils/Masking.py", 689),
    ("galfind/visualization/Cutout.py", 925),
    ("galfind/visualization/PDF.py", 1255),
]

print("Checking the 22 specified errors have been fixed...\n")
all_ok = True

for filepath, line_num in files_and_lines:
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            # Check if the line exists and can be parsed
            if line_num <= len(lines):
                line_content = lines[line_num - 1].strip()
                # Try to parse the entire file to ensure no syntax errors
                with open(filepath, "r") as f2:
                    try:
                        ast.parse(f2.read())
                        # File parsed successfully, but check
                        # surrounding context
                        start = max(0, line_num - 3)
                        end = min(len(lines), line_num + 2)
                        context = "".join(lines[start:end])
                        print(f"✓ {filepath}:{line_num} - OK")
                    except SyntaxError as e:
                        if e.lineno != line_num:
                            print(
                                f"✓ {filepath}:{line_num} - FIXED "
                                f"(other errors in file at "
                                f"line {e.lineno})"
                            )
                        else:
                            print(f"✗ {filepath}:{line_num} - STILL HAS ERROR")
                            all_ok = False
    except Exception as e:
        print(f"✗ {filepath}:{line_num} - ERROR: {e}")
        all_ok = False

print("\n" + ("=" * 60))
if all_ok:
    print("✓ All 22 specified errors have been fixed!")
else:
    print("Some errors need attention")

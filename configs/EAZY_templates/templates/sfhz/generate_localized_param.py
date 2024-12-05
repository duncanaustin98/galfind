import glob
import pathlib

def generate_localized_param():
    print('Generating localized param files')

    # Look for all the .param files in the directory and subdirectories
    param_files = glob.glob('*.param', recursive=True)

    print(param_files)

    # Loop through all the .param files
    for param_file in param_files:
        print(f'Processing {param_file}')

        # Read the contents of the .param file
        with open(param_file, 'r') as f:
            param_contents = f.read()

        # Loop over each line in the file. Second item in each row(space seperated) will be relative path of the file.
        # Find absolute path and replace the relative path with the absolute path.
        for line in param_contents.split('\n'):
            if len(line.split()) > 1:
                relative_path = line.split()[1]
                # relative path may be relative to a higher directory in the current hierarchy
                # so we need to find the absolute path of the file
                absolute_path = pathlib.Path(param_file).parent / relative_path
                absolute_path = absolute_path.resolve()
                # Replace the relative path with the absolute path
                param_contents = param_contents.replace(relative_path, str(absolute_path))
                
        # Write the contents back to the file - but replace .param with _galfind.param
        with open(param_file.replace('.param', '_galfind.param'), 'w') as f:
            f.write(param_contents)


if __name__ == '__main__':
    generate_localized_param()


        
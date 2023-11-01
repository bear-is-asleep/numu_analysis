import nbformat
from nbconvert import PythonExporter
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Select input and output (optional) files")

# Add the arguments
parser.add_argument('-i', type=str, help='Input file name',required=True)
parser.add_argument('-o', type=str, help='Output file name',default='output.py',required=False)

# Parse the arguments
args = parser.parse_args()

def convert_ipynb_to_py(ipynb_file_name, py_file_name='output.py'):
    # load notebook
    with open(ipynb_file_name) as f:
        nb = nbformat.read(f, as_version=4)

    # create python exporter
    exporter = PythonExporter()
    
    # process the notebook we loaded
    body, _ = exporter.from_notebook_node(nb)
    
    # write to python file
    with open(py_file_name, 'w') as f:
        f.write(body)
        
convert_ipynb_to_py(args.i, args.o)
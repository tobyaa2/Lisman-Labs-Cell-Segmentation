import os
import sys

# Make the project root importable so `import cell_counter` works under pytest
# regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

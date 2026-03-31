import os
import glob

header_py = """# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""

header_r = """# Copyright (c) 2026 Moses Boudourides
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""

def prepend_to_file(filepath, header):
    with open(filepath, 'r') as f:
        content = f.read()
    if "Copyright (c)" not in content:
        with open(filepath, 'w') as f:
            f.write(header + content)

for file in glob.glob("/home/ubuntu/repo_work/day1/python_app/*.py"):
    prepend_to_file(file, header_py)

for file in glob.glob("/home/ubuntu/repo_work/day1/r_app/*.R"):
    prepend_to_file(file, header_r)

prepend_to_file("/home/ubuntu/repo_work/day1/app.py", header_py)

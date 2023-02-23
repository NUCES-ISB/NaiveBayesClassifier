import os
import sys

if not os.path.exists('example.txt'):
    print('Error: example.txt not found!')
    sys.exit(1)
else:
    print('Success: example.txt found.')

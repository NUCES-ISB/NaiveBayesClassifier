import os
import sys

if not os.path.exists('itaimoor/example.txt'):
    print('Success: example.txt found.')
else:

    print('Error: example.txt not found!')
    sys.exit(1)

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'contacts' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts 2D_STRING_ARRAY queries as parameter.
#

def contacts(queries):
    # Write your code here
    contacts = []
    output = []
    
    for query in queries:
        if "a" == query[0]:
            contacts.append(query[4:])
        else:
            search = query[5:]
            n = len(search)
            count = 0
            for contact in contacts:
                if contact[:n] == search:
                    count+=1
            output.append(count)
    print(contacts)
    return output
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    queries_rows = int(input().strip())

    queries = []

    for _ in range(queries_rows):
        queries.append(input().rstrip().split())

    result = contacts(queries)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()

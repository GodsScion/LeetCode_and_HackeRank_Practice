#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'countGroups' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING_ARRAY related as parameter.
#

from collections import defaultdict
def countGroups(related):
    groups = defaultdict(set)
    for i, rels in enumerate(related):
        for user, rel in enumerate(rels):
            if rel == '1':
                groups[i].add(user)
    
    visited = set()
    
    def dfs(user):
        if user in visited:
            return
        visited.add(user)
        
        children = set()
        
        for relative in groups[user]:
            dfs(relative)
            children.update(groups[relative])
            if relative in groups:
                groups.pop(relative)
        
        groups[user].update(children)
    
    for i in range(len(related)):
        dfs(i)
    
    return len(groups)


        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    related_count = int(input().strip())

    related = []

    for _ in range(related_count):
        related_item = input()
        related.append(related_item)

    result = countGroups(related)

    fptr.write(str(result) + '\n')

    fptr.close()

print(countGroups(["1100","1110","0110","0001"]))

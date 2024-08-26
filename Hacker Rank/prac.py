############################### DATA STRUCTURES ###############################


#---------------------- TREES ----------------------#

#<< Tree: Height of a Binary Tree (https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem)

# class Node:
#     def __init__(self, info): 
#         self.info = info  
#         self.left = None  
#         self.right = None 
#         self.level = None 

#     def __str__(self):
#         return str(self.info) 

# class BinarySearchTree:
#     def __init__(self): 
#         self.root = None

#     def create(self, val):  
#         if self.root == None:
#             self.root = Node(val)
#         else:
#             current = self.root
         
#             while True:
#                 if val < current.info:
#                     if current.left:
#                         current = current.left
#                     else:
#                         current.left = Node(val)
#                         break
#                 elif val > current.info:
#                     if current.right:
#                         current = current.right
#                     else:
#                         current.right = Node(val)
#                         break
#                 else:
#                     break

# Enter your code here. Read input from STDIN. Print output to STDOUT
'''
class Node:
      def __init__(self,info): 
          self.info = info  
          self.left = None  
          self.right = None 
           

       // this is a node of the tree , which contains info as data, left , right
'''
def heightHelper(root):
    if not root:
        return 0
    return 1 + max(heightHelper(root.left), heightHelper(root.right))

def height(root):
    if not root:
        return 0
    return max(heightHelper(root.left), heightHelper(root.right))

# tree = BinarySearchTree()
# t = int(input())

# arr = list(map(int, input().split()))

# for i in range(t):
#     tree.create(arr[i])

# print(height(tree.root))

#>>



#<< Tree: Level Order Traversal (https://www.hackerrank.com/challenges/tree-level-order-traversal/problem)

# class Node:
#     def __init__(self, info): 
#         self.info = info  
#         self.left = None  
#         self.right = None 
#         self.level = None 

#     def __str__(self):
#         return str(self.info) 

# class BinarySearchTree:
#     def __init__(self): 
#         self.root = None

#     def create(self, val):  
#         if self.root == None:
#             self.root = Node(val)
#         else:
#             current = self.root
         
#             while True:
#                 if val < current.info:
#                     if current.left:
#                         current = current.left
#                     else:
#                         current.left = Node(val)
#                         break
#                 elif val > current.info:
#                     if current.right:
#                         current = current.right
#                     else:
#                         current.right = Node(val)
#                         break
#                 else:
#                     break

"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""
def levelOrder(root):
    # Write your code here
    output = []
    unvisited = [root]
    while unvisited:
        root = unvisited.pop(0)
        output.append(str(root.info))
        if root.left:
            unvisited.append(root.left)
        if root.right:
            unvisited.append(root.right)
    
    print(" ".join(output))
    

# tree = BinarySearchTree()
# t = int(input())

# arr = list(map(int, input().split()))

# for i in range(t):
#     tree.create(arr[i])

# levelOrder(tree.root)

#>>





#---------------------- STACKS ----------------------#

#<< Balanced Brackets (https://www.hackerrank.com/challenges/balanced-brackets/problem)

# #!/bin/python3

# import math
# import os
# import random
# import re
# import sys

# #
# # Complete the 'isBalanced' function below.
# #
# # The function is expected to return a STRING.
# # The function accepts STRING s as parameter.
# #

def isBalanced(s):
    # Write your code here
    bras = []
    braOpen = set(['(', '[', '{'])
    braClose = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    
    for bra in s:
        if bra in braOpen:
            bras.append(bra)
        elif bra in braClose:
            if not bras or bras[-1] != braClose[bra]:
                return "NO"
            else:
                bras.pop(-1)
    
    return "NO" if bras else "YES"


# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     t = int(input().strip())

#     for t_itr in range(t):
#         s = input()

#         result = isBalanced(s)

#         fptr.write(result + '\n')

#     fptr.close()

#>>
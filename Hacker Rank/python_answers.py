############################### DATA STRUCTURES ###############################


#---------------------- TREE ----------------------#

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






#<< Swap Nodes [Algo] (https://www.hackerrank.com/challenges/swap-nodes-algo/problem)

# #!/bin/python3

# # import math
# import os
# # import random
# # import re
# # import sys

# #
# # Complete the 'swapNodes' function below.
# #
# # The function is expected to return a 2D_INTEGER_ARRAY.
# # The function accepts following parameters:
# #  1. 2D_INTEGER_ARRAY indexes
# #  2. INTEGER_ARRAY queries
# #

from collections import deque
class Node:
    def __init__(self, val: int = -1, height: int = -1):
        self.val: int = val
        self.height: int = height
        self.left: Node | None = None
        self.right: Node | None = None
        
class Tree:
    def __init__(self, arr: list = []):
        self.root = Node(1, 1)
        queue: deque = deque([(self.root, 2)])
        i: int = 0
        while queue and i < len(arr):
            (root, childHeight) = queue.popleft()
            [left, right] = arr[i]
            if left > -1:
                root.left = Node(left, childHeight)
                queue.append((root.left, childHeight+1))
            if right > -1:
                root.right = Node(right, childHeight)
                queue.append((root.right, childHeight+1))
            i += 1
        print()
    
    def __inOrderTraversal(self, root, output) -> list:
        if not root:
            return output
        output = self.__inOrderTraversal(root.left, output)
        output.append(root.val)
        output = self.__inOrderTraversal(root.right, output)
        return output
        
    def getInOrderListRecursive(self) -> list:
        return self.__inOrderTraversal(self.root, [])
        
    def getInOrderListIterative(self) -> list:
        stack, result = [], []
        current = self.root
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.val)
            current = current.right
        return result

    def __preOrderTraversal(self, root, output) -> list:
        if not root:
            return output
        output.append(root.val)
        output = self.__preOrderTraversal(root.left, output)
        output = self.__preOrderTraversal(root.right, output)
        return output
    
    def getPreOrderListRecursive(self) -> list:
        return self.__preOrderTraversal(self.root, [])
    
    def getLevelOrderPairs(self) -> list:
        output: list = []
        queue: deque = deque([self.root])
        while queue:
            root = queue.popleft()
            left, right = -1, -1
            if root.left:
                left = root.left.val
                queue.append(root.left)
            if root.right:
                right = root.right.val
                queue.append(root.right)
            output.append([left, right])
        return output
    
    def swap(self, height: int = 1) -> None:
        queue: deque = deque([self.root])
        while queue:
            root: Node = queue.popleft()
            if root.height % height == 0:
                root.left, root.right = root.right, root.left
            if root.left:
                queue.append(root.left)
            if root.right:
                queue.append(root.right)

def swapNodes(indexes, queries):
    # Write your code here
    tree = Tree(indexes)
    output = []
    for query in queries:
        tree.swap(query)
        output.append(tree.getInOrderListIterative())
    return output

# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     n = int(input().strip())

#     indexes = []

#     for _ in range(n):
#         indexes.append(list(map(int, input().rstrip().split())))

#     queries_count = int(input().strip())

#     queries = []

#     for _ in range(queries_count):
#         queries_item = int(input().strip())
#         queries.append(queries_item)

#     result = swapNodes(indexes, queries)

#     fptr.write('\n'.join([' '.join(map(str, x)) for x in result]))
#     fptr.write('\n')

#     fptr.close()

#>>





#---------------------- STACK ----------------------#

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





#---------------------- TRIE ----------------------#

#<< Contacts (https://www.hackerrank.com/challenges/contacts/problem)

# #!/bin/python3

# # import math
# import os
# # import random
# # import re
# # import sys

# #
# # Complete the 'contacts' function below.
# #
# # The function is expected to return an INTEGER_ARRAY.
# # The function accepts 2D_STRING_ARRAY queries as parameter.
# #

class Node:
    def __init__(self, children: dict = None, end: bool = False, count: int = 0):
        self.children = children if children else dict()
        self.endOfWord = end
        self.countOfWords = count
        
class Contacts:
    def __init__(self, node: Node = None):
        self.root = node if node else Node()
    
    def find(self, word):
        root: Node = self.root
        for ch in word:
            if ch not in root.children:
                return 0
            root = root.children[ch]
        return root.countOfWords
    
    def add(self, word):
        root: Node = self.root
        for ch in word:
            if ch not in root.children:
                root.children[ch] = Node()
            root.countOfWords += 1
            root = root.children[ch]
        root.endOfWord = True
        root.countOfWords += 1
        return root.countOfWords
    
def contacts(queries):
    # Write your code here
    contacts = Contacts()
    output = []
    
    for query in queries:
        if "add" == query[0]:
            contacts.add(query[1])
        else:
            output.append(contacts.find(query[1]))
    return output

# Brute Force Solution (Not recommended because of time complexity)
# def contacts(queries):
#     # Write your code here
#     contacts = []
#     output = []
    
#     for query in queries:
#         if "a" == query[0]:
#             contacts.append(query[4:])
#         else:
#             search = query[5:]
#             n = len(search)
#             count = 0
#             for contact in contacts:
#                 if contact[:n] == search:
#                     count+=1
#             output.append(count)
#     print(contacts)
#     return output
    
    
# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     queries_rows = int(input().strip())

#     queries = []

#     for _ in range(queries_rows):
#         queries.append(input().rstrip().split())

#     result = contacts(queries)

#     fptr.write('\n'.join(map(str, result)))
#     fptr.write('\n')

#     fptr.close()



#>>




#---------------------- HEAP ----------------------#

#<< Find the Running Median (https://www.hackerrank.com/challenges/find-the-running-median/problem)

# #!/bin/python3

# import math
# import os
# import random
# import re
# import sys

# #
# # Complete the 'runningMedian' function below.
# #
# # The function is expected to return a DOUBLE_ARRAY.
# # The function accepts INTEGER_ARRAY a as parameter.
# #


import heapq
def runningMedian(a):
    # Write your code here
    isEven = True
    left, right, output = [], [], []
    heapq.heapify(left)
    heapq.heapify(right)
    
    for n in a:
        if right and n > right[0]:
            if isEven:
                res = heapq.heappushpop(right, n)
                heapq.heappush(left, -1*res)
            else:
                heapq.heappush(right, n)
        else:
            if isEven:
                heapq.heappush(left, -1*n)
            else:
                res = -1*heapq.heappushpop(left, -1*n)
                heapq.heappush(right, res)
        isEven = not isEven
        median = (right[0]-left[0])/2 if isEven else -1*left[0]
        output.append(median)
    return output
        

# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')

#     a_count = int(input().strip())

#     a = []

#     for _ in range(a_count):
#         a_item = int(input().strip())
#         a.append(a_item)

#     result = runningMedian(a)

#     fptr.write('\n'.join(map(str, result)))
#     fptr.write('\n')

#     fptr.close()

#>>




#---------------------- GRAPHS ----------------------#

#<< BFS: Shortest Reach in a Graph (https://www.hackerrank.com/challenges/ctci-bfs-shortest-reach/problem)

# Dijkstra's Shortest Path Algorithm
from heapq import heapify, heappop, heappush
class Graph:
    def __init__(self, n: int):
        self.neighbors = [[] for i in range(n)]
        
    def connect(self, source: int, destination: int, weight: int = 6) -> None:
        self.neighbors[source].append((destination, weight))
        self.neighbors[destination].append((source, weight))
        
    def find_all_distances(self, source: int) -> None:
        output = ""
        shortest = dict()
        minDistance = [(0, source)]
        heapify(minDistance)
        
        while minDistance:
            w1, n1 = heappop(minDistance)
            if n1 in shortest:
                continue
            for n2, w2 in self.neighbors[n1]:
                if n2 not in shortest:
                    heappush(minDistance, (w1 + w2, n2))
            shortest[n1] = w1
        
        for node in range(len(self.neighbors)):
            if node not in shortest:
                shortest[node] = -1
            if node != source:
                output += f"{shortest[node]} "
        print(output)


# t = int(input())
# for i in range(t):
#     n, m = [int(value) for value in input().split()]
#     graph = Graph(n)
#     for i in range(m):
#         x, y = [int(x) for x in input().split()]
#         graph.connect(x-1, y-1)
#     s = int(input())
#     graph.find_all_distances(s-1)

#>>
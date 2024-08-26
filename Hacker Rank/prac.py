############################### DATA STRUCTURES ###############################


#---------------------- TREES ----------------------#

#<< Tree: Height of a Binary Tree

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


#<< Tree: Level Order Traversal

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
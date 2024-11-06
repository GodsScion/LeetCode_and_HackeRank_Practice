// ##### LINKED LIST ##### //
class ListNode {
    val: number
    next: ListNode | null
    constructor(val?: number, next?: ListNode | null) {
        this.val = (val===undefined ? 0 : val)
        this.next = (next===undefined ? null : next)
    }
}
// 141
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     val: number
 *     next: ListNode | null
 *     constructor(val?: number, next?: ListNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.next = (next===undefined ? null : next)
 *     }
 * }
 */
function hasCycle(head): boolean {
    let runner = head;
    let chaser = head;
    while (runner && runner.next) {
        runner = runner.next.next;
        chaser = chaser.next;
        if ( runner === chaser ) {
            return true;
        }
    }
    return false;
};
// This code works too, flawlessly! But the type checking is giving an annoying false error
// function hasCycle(head: ListNode | null): boolean {
//     let runner: ListNode | null = head;
//     let chaser: ListNode | null = head;
//     while (runner && runner.next) {
//         runner = runner.next.next;
//         chaser = chaser.next;  // chaser will never be null in this line 
//         if ( runner === chaser ) {
//             return true;
//         }
//     }
//     return false;
// };



// ##### TREES ##### //
class TreeNode {
    val: number
    left: TreeNode | null
    right: TreeNode | null
    constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
        this.val = (val===undefined ? 0 : val)
        this.left = (left===undefined ? null : left)
        this.right = (right===undefined ? null : right)
    }
}

// 226. Invert Binary Tree (https://leetcode.com/problems/invert-binary-tree/description/)
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */
function invertTree(root: TreeNode | null): TreeNode | null {
    if (!root) return null;
    const temp: TreeNode | null = root.left;
    root.left = invertTree(root.right);
    root.right = invertTree(temp);
    return root;
};

// 104. Maximum Depth of Binary Tree (https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */
function maxDepth(root: TreeNode | null): number {
    if (root == null) return 0;
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
};


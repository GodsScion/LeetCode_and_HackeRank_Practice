// ##### LINKED LIST ##### //
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
function hasCycle(head: ListNode | null): boolean {
    let runner: ListNode | null = head;
    let chaser: ListNode | null = head;
    while (runner && runner.next) {
        runner = runner.next.next;
        chaser = chaser.next;
        if ( runner === chaser ) {
            return true;
        }
    }
    return false;
};



// ##### TREES ##### //

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

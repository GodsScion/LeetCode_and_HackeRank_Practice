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

// const x = 2; let y = 4; function update(arg) { return Math.random() + y * arg; } y = 2; y=3; const result = update(x); console.log(result);;

var longestConsecutive = function(nums) {
    let numSet = new Set(nums);
    let maxSeqLen = 0;

    while (maxSeqLen < numSet.size) {
        let num = [...numSet][0];
        let longest = num + 1;

        while (numSet.has(longest)) {
            numSet.delete(longest);
            longest++;
        }

        num = num - 1;
        while (numSet.has(num)) {
            numSet.delete(num);
            num--;
        }

        maxSeqLen = Math.max(maxSeqLen, longest - num - 1);
    }

    return maxSeqLen;
};
    
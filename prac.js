// const x = 2; let y = 4; function update(arg) { return Math.random() + y * arg; } y = 2; y=3; const result = update(x); console.log(result);;


var longestConsecutive = function(nums) {
    let maxSeqLen = 0
    while (maxSeqLen < nums.length) {
        let num = nums.pop()
        let longest = num
        while (nums.includes(num-1)) {
            num--
            nums.splice(nums.indexOf(num),1)
        }
        while (nums.includes(longest+1)) {
            longest++
            nums.splice(nums.indexOf(longest),1)
        }
        maxSeqLen = Math.max(longest - num + 1, maxSeqLen)
    }
    return maxSeqLen;
};


var questions = [
    [100,4,200,1,3,2]
]

for (let q of questions) {
    console.log(longestConsecutive(q));
}
// const x = 2; let y = 4; function update(arg) { return Math.random() + y * arg; } y = 2; y=3; const result = update(x); console.log(result);;

var longestConsecutive = function() {
    let nums = []
    let maxSeqLen = 0
    while (maxSeqLen < nums.length) {
        let num = nums.pop()
        while (nums.includes(num-1)) {
            num--
            nums.splice(nums.indexOf(num),1)
        }
        let longest = nums
        while (nums.includes(longest+1)) {
            longest++
            nums.splice(nums.indexOf(longest),1)
        }
        maxSeqLen = Math.max(longest - nums + 1, maxSeqLen)
    }
    return maxSeqLen;
};


var questions = [
    [100,4,200,1,3,2]
]

for (let q of questions) {
    console.log(longestConsecutive(q));
}
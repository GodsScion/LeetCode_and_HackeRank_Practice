// const x = 2; let y = 4; function update(arg) { return Math.random() + y * arg; } y = 2; y=3; const result = update(x); console.log(result);;

// Problem 424 
var characterReplacement = function(s,k) {
    let window = {}, i = 0, maxLen = 1;
    for (let j=0; j<s.length; j++) {
        if (s[j] in window) { 
            window[s[j]] += 1;
            maxLen = Math.max(maxLen, window[s[j]])
        } else {
            window[s[j]] = 1;
        }
        if (j-i+1 > maxLen+k) {
            window[s[i]] -= 1;
            i++;
        }
    }
    return s.length - i
};

testcases = [
    ["ABABA",2],
    ["ABAB",1],
    ["ABAA",0],
    ["AAAAA",0]
]

for (let test of testcases) { console.log(characterReplacement(test[0],test[1])); }

// var maxProductDifference = function(nums) {
//     nums.sort((a,b) => a-b);
//     return nums[nums.length-1]*nums[nums.length-2] - nums[1]*nums[0]
// };


// var longestConsecutive = function(nums) {
//     if (!nums.length) return 0
//     nums.sort((a,b) => a-b)
//     let prev = nums[0]
//     let seq = 1
//     let maxSeq = 1
//     for (let num of nums) {
//         if ( prev + 1 === num ) {
//             seq++
//             maxSeq = Math.max(maxSeq, seq)
//         } else if (prev !== num) {
//             seq = 1
//         }
//         prev = num
//     }
//     return maxSeq
// };


// var questions = [
//     [100,4,200,1,3,2],[],[1,9,3,7,5],[1,2,3,4,5,6,7]
// ]

// for (let q of questions) {
//     console.log(maxProductDifference(q));
// }

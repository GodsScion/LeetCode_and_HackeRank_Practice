// Check for if given input is a valid integer
function isInt(input) {
    return Number.isInteger(input);
}

// JS collections

// Set
let set = new Set();

// Map
let map = new Map();


// Array
let arr = [1, 2, 3];

// Objects
let obj = { a: 1, b: 2 };

// Priority Queue, there is no built-in priority queue in JS



function main() {
    let tests = ["45.7", "45", NaN, null, undefined, {}, [], '4', 20.5, -5, 20];

    for (let test of tests) {
        console.log(`"${test}" of type "${typeof test}" is int: ${isInt(test)}`);
    }
    
}

main();
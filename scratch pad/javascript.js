// Check for if given input is a valid integer
function isInt(input) {
    return Number.isInteger(input);
}


function main() {
    let tests = ["45.7", "45", NaN, null, undefined, {}, [], '4', 20.5, -5, 20];

    for (let test of tests) {
        console.log(`"${test}" of type "${typeof test}" is int: ${isInt(test)}`);
    }
    
}

main();
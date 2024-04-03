int maxProductDifference(int* nums, int numsSize){
    int max1 = 0, max2 = 0, min1 = 100000, min2 = 100000;
    for(int i=0; i < numsSize; i++){
        if (nums[i] > max1) {max2=max1; max1=nums[i];} else if (nums[i] > max2) {max2=nums[i];}
        if (nums[i] < min1) {min2=min1; min1=nums[i];} else if(nums[i] < min2) {min2=nums[i];}
    }
    return max2*max1 - min1*min2;
}
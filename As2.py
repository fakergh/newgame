Bubble sort-

import time

import random

#timer to keep track of performance

start = time.perf_counter()

# a function to implement bubble sort in parallel

def Parallel_bubble_sort(lst):

# variable to keep track of swaps to end the while loop

Sorted = 0

# variable to get length of list

n = len(lst)

#loop to traverse all list elements in phases

while Sorted == 0:

# set to 1 initially to assume list is sorted

# and no swaps occurred

Sorted = 1

# traverse all list elements in pair

# start at index 0 for odd phase

# start at index 1 for even phase

for i in range(0, n-1, 2):

# check if current element greater than next element

if lst[i] > lst[i+1]:

# if so, swap the elements

lst[i], lst[i+1] = lst[i+1], lst[i]

# set to 0 to imply a swap occurred

Sorted = 0

for i in range(1, n-1, 2):

if lst[i] > lst[i+1]:

lst[i], lst[i+1] = lst[i+1], lst[i]

Sorted = 0
# print final sorted list

print(lst)

# an example list to test above program

lst = [(random.randint(0,100)) for i in range(100)]

Parallel_Bubble_Sort(lst)

finish = time.perf_counter()

print(f'Finished in {round(finish-start,2)} second(s)')

Merge sort-

def merge(arr, l, m, r):

n1 = m - l + 1

n2 = r - m

# create temp arrays

L = [0] * (n1)

R = [0] * (n2)

# Copy data to temp arrays L[] and R[]

for i in range(0, n1):

L[i] = arr[l + i]

for j in range(0, n2):

R[j] = arr[m + 1 + j]

# Merge the temp arrays back into arr[l..r]

i = 0 # Initial index of first subarray

j = 0 # Initial index of second subarray

k = l # Initial index of merged subarray

while i < n1 and j < n2:

if L[i] <= R[j]:

arr[k] = L[i]

i += 1

else:

arr[k] = R[j]

j += 1

k += 1

# Copy the remaining elements of L[], if there

# are any

while i < n1:

arr[k] = L[i]

i += 1

k += 1

# Copy the remaining elements of R[], if there

# are any

while j < n2:

arr[k] = R[j]
j += 1

k += 1

# l is for left index and r is right index of the

# sub-array of arr to be sorted

def mergeSort(arr, l, r):

if l < r:

# Same as (l+r)//2, but avoids overflow for

# large l and h

m = l+(r-l)//2

# Sort first and second halves

mergeSort(arr, l, m)

mergeSort(arr, m+1, r)

merge(arr, l, m, r)

# Driver code to test above

arr = [12, 11, 13, 5, 6, 7]

n = len(arr)

print("Given array is")

for i in range(n):

print("%d" % arr[i],end=" ")

mergeSort(arr, 0, n-1)

print("\n\nSorted array is")

for i in range(n):

print("%d" % arr[i],end=" ")

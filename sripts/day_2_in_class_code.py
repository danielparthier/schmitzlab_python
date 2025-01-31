
def func1():
    # local variable
    s = "I love Python"
    print("Inside Function:", s)

# Driver code
func1()
print(s) #nothing 


# Global scope
def func2():
    # not defining s in this function
    print("Inside Function:", s)

s = "great"
print("Outside Function", s)
func2() # error? no because s is a global variable

# mixing them up

def func3():
    s = "Me too."
    print(s)

# Global scope
s = "I love Geeksforgeeks"
func3()
print(s)

# you can't modify global variables in functions
def fail():
    s.append(3)
    print("Inside Function", s)


# Global scope
s = [1,2,5]
fail()

# solution?



# FOR LOOPS

list_to_sum = [2,3,4,5,7,45]
# write a for loop that sums the numbers in the list

num_sum = 0
for num in list_to_sum:
    num_sum = num_sum + num # num_sum += num # alternative


# SOLUTIONS
A = [3, 4, 5, 9, 12, 87, -65, 300, 450, -32]
# in class - a for loop that adds 3 to each element of a list

# clean solution
B = []
for val in A:
    val += 3
    B.append(val)
print(B)

# one line solution
B = []
for val in A:
    B.append(val + 3)
print(B)

# calcualte the average but if an element is negative --> make this element 0
sum_num = 0
for val in A:
    if val < 0:
        val = 0
    sum_num += val
print(sum_num / len(A))

# find the maximum value
biggest = A[0]
for val in A:
    if val > biggest:
        biggest = val

# position of max value - better but not perfect
biggest = A[0]
count = -1
for val in A:
    count += 1
    if val > biggest:
        biggest = val
        indx_max = count

print('the max value is ', biggest)
print('its position is ', indx_max)
print(A[indx_max])

# fixing errors
## syntax error
def extract_max(a)
    return max(a)

def solve_problem():
    b = ((1+1)*3))+(4/2)
    c = b * 2


# indentation error
def extract_max(a):
    if len(a) == 0:
        print("Empty list")
    else:
    return max(a)


## name error
a = [1,2,3]
def extract_max(a):
    return maxa(a)

def extract_double_max(a):
    import numpy as np
    c = np.array(a) * 2
    return max(c)

# type error
extract_double_max(2)

print("We can connect strings with numbers like" + 2)

# index error
a = [1,2,3]
print(a[3])

for i in range(10):
    print(a[i])
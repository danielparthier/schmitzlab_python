
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


RMPs = [1,2,10,4,5,6,7,8,9,10]
if RMPs[0] > 5:
    RMPs[1]
elif RMPs[1] > 5:
    RMPs[2]
elif RMPs[2] > 5:
    RMPs[3]
elif RMPs[3] > 5:
    RMPs[4]
elif RMPs[4] > 5:
    RMPs[5]
else:
    RMPs[2]

if RMPs[0] > -10:
    if RMPs[1] > 1:
        RMPs[1] * RMPs[2]
    else:
        pass
filename = "blabla.txt"
if filename.endswith(".txt"):
    print("process text")
elif filename.endswith(".csv"):
    print("process csv")
elif filename.endswith(".npy"):
    print("process numpy")

if 8.0 in RMPs:
    print("8 is in the list")
else:
    print("8 :(")

if isinstance(8.0, int):
    print("8 is an integer")
else:
    print("nope")

a = 8.0
if isinstance(a, int) and a in RMPs:
    print("8 is in the list and it is an integer")
else:
    print("nope")

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

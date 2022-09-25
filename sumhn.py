gp = 0
noOfIter = list(range(0,100))

for item in noOfIter:
    for i in range(0,item):
        gp += pow(-0.5,i)
    sum = 0.5 + 1.25*(gp)
    print("No. of Iterations:",item,"|","Sum of h(n):",sum)
    gp = 0

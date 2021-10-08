
# simple coloring algorithm still in progress
allowedbitfield=[0,
        1,
        1 << 1,#10
        1 << 2,#100
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
        1 << 8,
    ]


allallowed=sum(allowedbitfield)


def solve(data):
    placed=solveboard(data)
    return placed==81


def solveboard(data):
    placed=0
    allowedval=[[allallowed]*9]*9
    
    for i in range(9):
        for j in range(9):
            if data[i][j]>0:
                allowedval[i][j]=0
                change(data,allowedval,i,j)
                placed+=1

    return solveboard(data,allowedval,placed)


def solveboard(data,allowedval,placed):
    pass

def brute(data,allowedval,placed):

    for i in range(9):
        for j in range(9):
            if(data[i][j]==0):
                for k in range(1,10):

                    if((allowedval[i][j] & allowedbitfield[k])>0):
                        copy_data=data
                        copy_allowed=allowedval

                        copy_data[i][j]=k
                        copy_allowed=0
                        change(copy_data,copy_allowed,i,j)


                        placed_num=solveboard(copy_data,copy_allowed,placed=1)

                        if(placed_num==81):
                            return copy_data
                return 
    return 




def mfmf(data,allowedval):

    movecount=0

    for i in range(1,10):
        allowedfield

def  countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

def nakedpairs(allowedval):

    for i in range(9):
        for j in range(9):

            val=allowedval[i][j]

            # check 2 candidates
            if(countSetBits(val)==2):

                for k in range(j+1,9):

                    if allowedval[i][k]==val:

                        remove=~val
                        for m in range(9):
                            if(m is not j or m is not k):
                                allowedval[i][m]&=remove

    for j in range(9):
        for i in range(9):

            val=allowedval[i][j]

            # check 2 candidates
            if(countSetBits(val)==2):

                for k in range(i+1,9):

                    if allowedval[k][j]==val:

                        remove=~val
                        for m in range(9):
                            if(m is not i or m is not k):
                                allowedval[m][j]&=remove




def remrowcol(data,allowedval):
    movecount=0

    for val in range(1,10):
        allowed=allowedval[val]

        for i in range(9):
            
            for j in range(9):


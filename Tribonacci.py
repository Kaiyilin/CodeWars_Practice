#Tribonacci

"""
Clever method by others
def tribonacci(signature, n):
  res = signature[:n]
  for i in range(n - 3): res.append(sum(res[-3:]))
  return res
"""

def tribonacci(signature, n):
    #your code here
    pointer = 0 
    tibarray = signature.copy()
    if n == 0: 
        return []
    elif n > 0 and n <3:
        lesser_array = []
        for i in range(n):
            lesser_array.append(signature[i])
        return lesser_array
    else:
        while len(tibarray) < n:
            next_element = sum([tibarray[pointer], tibarray[pointer+1],tibarray[pointer+2]])
            tibarray.append(next_element)
            pointer += 1
        return tibarray
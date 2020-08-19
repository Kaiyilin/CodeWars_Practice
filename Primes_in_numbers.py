# primes in numbers

"""
Given a positive number n > 1 find the prime factor decomposition of n. The result will be a string with the following form :

 "(p1**n1)(p2**n2)...(pk**nk)"

 Example: n = 86240 should return "(2**5)(5)(7**2)(11)"
"""

def primeFactors(n):
    init_num = 2
    power = 0
    num_list = list([])
    num_list_2 = list([])
    while n/init_num != 1:
        if n%init_num == 0:
            num_list.append(init_num)
        print(num_list)
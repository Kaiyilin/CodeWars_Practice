# Codewars practice
"""
Format a string of names like 'Bart, Lisa & Maggie'.

Given: an array containing hashes of names

Return: a string formatted as a list of names separated by commas except for the last two names, which should be separated by an ampersand.

namelist([ {'name': 'Bart'}, {'name': 'Lisa'}, {'name': 'Maggie'} ])
# returns 'Bart, Lisa & Maggie'

namelist([ {'name': 'Bart'}, {'name': 'Lisa'} ])
# returns 'Bart & Lisa'

namelist([ {'name': 'Bart'} ])
# returns 'Bart'

namelist([])
# returns ''
"""

# testing the usage of list, dictionary and string manipulation

def namelist(names):
    str = ''
    if len(names) != 0:
        array = []
        for i in range(len(names)-1):
            array.append(names[i].get('name'))
        #print(array)
        str = ', '.join(array)
        str += ' & ' + names[-1].get('name') if str != '' else names[-1].get('name') # using shorthand if else and += Addition Assignment

        return str
    
    else:
        return str
    
    
# Reference https://python-reference.readthedocs.io/en/latest/docs/operators/addition_assignment.html and https://www.w3schools.com/python/ref_string_join.asp

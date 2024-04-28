# print(set(A) | set(B)) Returns all the KEYS.

def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    return {x: 1 - A[x] for x in A}

def fuzzy_difference(A, B):
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in set(A) | set(B)}

def cartesian_product(A, B):
    return {(x, y): min(A[x], B[y]) for x in A for y in B}

def max_min_composition(R, S):
    # Ensure that the second element of R and the first element of S match
    result = {}
    for (x, y), r_val in R.items():
        for (y_prime, z), s_val in S.items():
            if y == y_prime:
                if (x, z) not in result:
                    result[(x, z)] = r_val if r_val < s_val else s_val
                else:
                    result[(x, z)] = max(result[(x, z)], min(r_val, s_val))
    return result

# Define fuzzy sets
A = {'a': 0.5, 'b': 0.8}
B = {'b': 0.6, 'c': 0.9}

# Operations on fuzzy sets
print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference A - B:", fuzzy_difference(A, B))

# # Define fuzzy relations
R = {('x1', 'y1'): 0.2, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.8}
S = {('y1', 'z1'): 0.3, ('y2', 'z1'): 0.6, ('y1', 'z2'): 0.4}

R2 = {('x1', 'y1'): 0.7, ('x1', 'y2'): 0.5, ('x2', 'y1'): 0.8, ('x2', 'y2'): 0.4 }
S2 = {('y1', 'z1'): 0.9, ('y1', 'z2'): 0.6, ('y2', 'z1'): 0.1, ('y2', 'z2'): 0.7 }
# # Cartesian product and max-min composition
print("Cartesian Product of A and B:", cartesian_product(A, B))
# print("Max-Min Composition of R and S:", max_min_composition(R, S))
print("Max-Min Composition of R and S:", max_min_composition(R2, S2))
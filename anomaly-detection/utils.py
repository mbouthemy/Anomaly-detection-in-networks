# utils.py


def compute_product(list_of_elements, begin):
    """Compute the product of a list"""
    product = 1
    for j in range(begin, len(list_of_elements)):
        product *= list_of_elements[j]
    return product


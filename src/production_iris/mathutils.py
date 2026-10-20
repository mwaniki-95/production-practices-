"""The sum of the numbers in a list."""
import numbers


def sum_numbers(numbers: list[float]) -> float:
    return sum(numbers)

print(sum_numbers([1, 2, 3, 4, 5]))  # Output: 15

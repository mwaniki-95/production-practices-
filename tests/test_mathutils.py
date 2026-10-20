from production_iris.mathutils import sum_numbers

def test_sum_numbers():
    numbers = [1, 2, 3, 4, 5]
    result = sum_numbers(numbers)
    assert result == 15
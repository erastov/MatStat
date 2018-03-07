import unittest

def get_formatted_name(first, last, middle=''):
    """Строит отформатированное полное имя."""
    if middle:
        full_name = first + ' ' + middle + ' ' + last
    else:
        full_name = first + ' ' + last
    return full_name.title()

class NamesTestCase(unittest.TestCase):
    """Тесты"""

    def test_first_last_name(self):
        """Имена вида 'Janis Joplin' работают правильно?"""
        formatted_name = get_formatted_name('janis', 'joplin')
        self.assertEqual(formatted_name, 'Janis Joplin')

    def test_first_last_middle_name(self):
        """Работают ли такие имена, как 'Wolfgang Amadeus Mozart'?"""
        formatted_name = get_formatted_name('wolfgang', 'mozart', 'amadeus')
        self.assertEqual(formatted_name, 'Wolfgang Amadeus Mozart')

unittest.main()

# def main():
#     print("Enter 'q' at any time to quit.")
#     while True:
#         first = input("\nPlease give me a first name: ")
#         if first == 'q':
#             break
#         last = input("Please give me a last name: ")
#         if last == 'q':
#             break
#
#         formatted_name = get_formatted_name(first, last)
#         print("\tNeatly formatted name: " + formatted_name + '.')
#
# if __name__ == '__main__':
#     main()

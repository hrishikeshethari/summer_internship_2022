import unittest


source_file_extensions = [
    'asm', 
    'c', 
    'class', 
    'cpp', 
    'java', 
    'js', 
    'jsp', 
    'pl', 
    'py', 
    'R', 
    'r',
    'aj',
    'RexsterExtension',
    'awk',
    'MPL',
    'java~',
    'mhtml',
    'jspf',
    'dml',
    'pydml']


def is_source_code_file(file_path):
    extension = file_path.split('.')[-1].lower()

    if extension in source_file_extensions:
        return True
    else:
        return False

# write some test cases for is_source_code_file by giving some file paths
# and check if the function returns the correct value

class TestIsNotSourceFile(unittest.TestCase):
    
    def test_is_not_source_file(self):

        test_cases = [
            ('src/main/java/com/example/HelloWorld.java', True),
            ('src/main/java/com/example/HelloWorld.py', True),
            ('src/main/java/com/example/HelloWorld.r', True),
            ('src/main/java/com/example/HelloWorld.R', True),
            ('src/main/java/com/example/HelloWorld.pl', True),
            ('src/main/java/com/example/HelloWorld.jsp', True),
            ('src/main/java/com/example/HelloWorld.js', True),
            ('src/main/java/com/example/HelloWorld.c', True),
            ('src/main/java/com/example/HelloWorld.cpp', True),
            ('src/main/java/com/example/HelloWorld.class', True),
            ('src/main/java/com/example/HelloWorld.asm', True),
            ('src/main/java/com/example/HelloWorld.csv', False),
            ('src/main/java/com/example/HelloWorld.txt', False),
            ('src/main/java/com/example/HelloWorld.md', False)
        ]

        print("\n")
        for tup in test_cases:
            file_path, expected_result = tup
            
            with self.subTest(file_path=file_path):
                self.assertEqual(is_source_code_file(file_path), expected_result)
                print(f'\t{file_path} -> {expected_result}')
                
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
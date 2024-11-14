import random
import string
import os

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_test_data(file_path, num_lines, line_length=8):
    with open(file_path, 'w') as file:
        for _ in range(num_lines):
            random_string = generate_random_string(line_length)
            file.write(random_string + '\n')  # A separate random string for each row
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Generated test data: {num_lines} lines, {line_length} characters per line.")
    print(f"File size: {file_size:.2f} MB.")

if __name__ == "__main__":
    # The file size and the length of each row can be set through command-line arguments
    import sys
    num_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 1024 * 1024
    line_length = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    generate_test_data('test.txt', num_lines, line_length)

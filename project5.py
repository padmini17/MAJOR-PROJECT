from termcolor import cprint
import os
from rich import print
from rich.console import Console


def decompress_lzw(compressed_data):
    dictionary = {i: chr(i) for i in range(256)}
    result = ""
    current_code = 256
    current_sequence = chr(compressed_data[0])
    result += current_sequence

    for code in compressed_data[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == current_code:
            entry = current_sequence + current_sequence[0]
        else:
            raise ValueError("Bad compressed sequence")

        result += entry
        dictionary[current_code] = current_sequence + entry[0]
        current_code += 1
        current_sequence = entry

    return result

def decompress_file_lzw(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        compressed_data_lzw = [int(code) for code in file.read().split()]

    decompressed_text = decompress_lzw(compressed_data_lzw)

    with open(output_filename, 'w') as file:
        file.write(decompressed_text)

    return decompressed_text

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n\n\n")

    # Print Major Project heading
    cprint("MAJOR PROJECT".center(os.get_terminal_size().columns), 'red', attrs=['bold'])
    print("\n\n")

    # Print Team heading
    cprint("TEAM 1".center(os.get_terminal_size().columns), 'yellow', attrs=['bold'])
    print("\n\n")

    # Print Topic and project details
    cprint("PROJECT : ENHANCED IMAGE STEGANOGRAPHY WITH DUAL AUTHENTICATION".center(os.get_terminal_size().columns), 'cyan')
    cprint("AND CAMELLIA CIPHER ENCRYPTION".center(os.get_terminal_size().columns), 'cyan')

    print("\n\n\n")

    cprint(" STAGE 5: TEXT DECOMPRESSION USING LZW ALGORITHM ".center(os.get_terminal_size().columns), 'green')
    print("\n\n")

    # Example usage:
    input_filename = "decoded_message.txt"
    output_filename = "received_message.txt"

    decompressed_text_lzw = decompress_file_lzw(input_filename, output_filename)

    # Printing information about the compressed and decompressed texts
    with open(input_filename, 'r') as file:
        compressed_data = [int(code) for code in file.read().split()]

    cprint("\nDetails about Message that is decompressed:", 'red')
    cprint(f"\nThe Compressed Text that needs to be decompressed is in the {input_filename}.", 'yellow')
    cprint("\nCompressed Data is ", 'cyan')
    print(compressed_data)

    cprint("\nDecompressed Text using LZW Algorithm: ", 'cyan')
    print(decompressed_text_lzw)
    cprint(f"\nThe Decompressed Text is saved in the {output_filename}.", 'yellow')



    

from termcolor import cprint
import os
import subprocess
from rich import print

def compress_lzw(data):
    dictionary = {chr(i): i for i in range(256)}
    result = []
    current_code = 256
    current_sequence = ""

    for char in data:
        if current_sequence + char in dictionary:
            current_sequence += char
        else:
            result.append(dictionary[current_sequence])
            dictionary[current_sequence + char] = current_code
            current_code += 1
            current_sequence = char

    if current_sequence in dictionary:
        result.append(dictionary[current_sequence])

    return result


def compress_file_lzw(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        original_text = file.read()

    compressed_data_lzw = compress_lzw(original_text)

    with open(output_filename, 'w') as file:
        for code in compressed_data_lzw:
            file.write(str(code) + ' ')

    return compressed_data_lzw

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

    cprint(" STAGE 3: TEXT COMPRESSION USING LZW ALGORITHM ".center(os.get_terminal_size().columns), 'green')
    print("\n\n")

    # Example usage:
    input_filename = "input_message.txt"
    output_filename = "compressed_message.txt"

    compressed_data_lzw = compress_file_lzw(input_filename, output_filename)

    # Printing information about the original, compressed, and decompressed texts
    with open(input_filename, 'r') as file:
        original_text = file.read()

    original_size = int(len(original_text))
    cprint("\nDetails about Message that is to be transmitted:", 'red')
    cprint(f"\nThe Original Text that the sender wants to send is in the {input_filename}.", 'yellow')
    cprint("\nOriginal Text is ", 'cyan')
    print(original_text)
    original_characters = len(original_text)
    print(f"\n[green]Number of Characters in Original Text:[/green] {original_characters}")

    cprint("\nCompressed Data using LZW Algorithm: ", 'cyan')
    try:
        with open(output_filename, 'r') as file:
            message = file.read()
        print("[green]\nMessage loaded from compressed file successfully.[/green]")
    except Exception as e:
        print(f"[red]Error loading message from file: {e}[/red]")
        sys.exit(0)
    plaintext_bits = ''.join(format(ord(char), '08b') for char in message)
    print("[yellow]\nCompressed Text: \n[/yellow]")
    print(message)
    compressed_size_bytes=len(message)
    compressed_characters = len(message.split())  # Assuming each code is separated by a space
    print(f"\n[yellow]Number of Characters in Compressed Text:[/yellow] {compressed_characters}")
    compression_percentage = ((original_characters - compressed_characters) / original_characters) * 100
    print(f"\n[green]Compression Percentage:[/green] {compression_percentage:.2f}%")
    print(f"\n[green]Formula: (1 - compressed_characters / original_characters) * 100)[/green]")
    cprint(f"\nThe Compressed Text that the sender will send is in the {output_filename}.", 'yellow')
    input("")
    subprocess.run(["python", "project4.py"])

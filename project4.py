#importing the required libraries
from PIL import Image                   #Pillow package
from os import path                     #os package
from Crypto.Cipher import AES          #Pycryptodome package
from Crypto.Hash import SHA256     #Pycryptodome package
from Crypto import Random               #Pycryptodome package
import base64                               #Base64
from termcolor import cprint        #Termcolor package
from pyfiglet import figlet_format    #Pyfiglet package
from rich import print                       #Rich package
from rich.console import Console             #Console 
import os                                   #To interact directly with os
import getpass                           #For the entry of the password
import sys                                  #To interact with python runtime environment
from Crypto.Util.Padding import pad, unpad      #For padding and unpaddingthe text
import binascii                                     #Convert the text to hexadecimal and then to binary
import time                                           #Time package
from PIL import Image, ImageDraw
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7
import secrets
import io
import subprocess


DEBUG = False                       #Used to display the debug information
console = Console()                  #To interact with the console
headertext = "HeaderText"    #HeaderText to add before encoding the image


#To convert into RGB mode from Grayscale, CMYK, RGBA modes of images.
def convertToRGBAndSave(img, output_filename):
    try:
        rgba_image = img
        rgba_image.load()
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image.split()[3])
        background.save(output_filename)
        print("[green]Converted image to RGB and saved as [bold]%s[/bold][/green]" % output_filename)
        return output_filename  
    except Exception as e:
        print("[red]Couldn't convert image to RGB or save it - %s[/red]" % e)
        return None  

#To know the Pixel Count of the image
def getPixelCount(img,display=True):
    width, height = Image.open(img).size
    if display:
        print("Width of the image : [green]%d[/green]  "%width)
        print("Height of the image : [green]%d[/green]" %height)
    return width * height

def is_png_image(image_path):
    try:
        img = Image.open(image_path)
        if img.format=="PNG":
            print(f"[green]\nThe image is in PNG Format[/green]\n")
        return img.format == "PNG"
    except Exception as e:
        return False

#To display the image
def show_image(image_path):
    try:
        print(f"\nEnter [red]'OPEN'[/red] to display the image: ")
        text=input()
        if text.lower()=="open" or text=="Open" or text=="OPEN":
            img = Image.open(image_path)
            img.show()
        else:
            return
    except FileNotFoundError:
        print(f"[red]Image file not found at path: {image_path}[/red]")
    except Exception as e:
        print(f"[red]An error occurred: {str(e)}[/red]")


def encrypt(key, source, encode=True, print_output=True):
    start_time = time.time()
    key_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
    key_hash.update(key)
    key = key_hash.finalize()

    IV = secrets.token_bytes(algorithms.Camellia.block_size // 8)
    encryptor = Cipher(algorithms.Camellia(key), modes.CFB(IV), backend=default_backend()).encryptor()
    padder = PKCS7(algorithms.Camellia.block_size).padder()

    # Padding the source data
    padded_data = padder.update(source) + padder.finalize()

    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    if print_output:
        print("\n[red]\nDetails about the Message, Key, and IV: [/red]")
        print(f"\nThe Plaintext after padding is: [green]{padded_data}[/green]")
        print(f"\nLength of the Plaintext after padding in Bytes:[blue]{len(padded_data)}[/blue] ")
        print(f"Length of the Plaintext after padding in Bits:[blue]{len(padded_data) * 8}[/blue] ")
        print(f"\nKey is:[green]{key}[/green]")
        print(f"Length of Key in bytes:[blue]{len(key)}[/blue]")
        print(f"Length of Key in bits: [blue]{len(key) * 8}[/blue]")
        print(f"\nIV: [green]{IV}[/green]")
        print(f"Length of IV in bytes: [blue]{len(IV)}[/blue]")
        print(f"Length of IV in bits: [blue]{len(IV) * 8}[/blue]")
        print("\n[red]Details about encrypting the padded Plaintext using Camellia encryption :[/red]")
        print(f"\nThe padded bits are encrypted, and the ciphertext is: [green]{ciphertext} [/green]")
        print(f"\nLength of CipherText in bytes: [blue]{len(ciphertext)}[/blue]")  # Print the encrypted text
        print(f"Length of CipherText in bits: [blue]{len(ciphertext) * 8}[/blue]")
        print("\n[yellow]Cipher Text bits:[/yellow]")
        hex_string = binascii.hexlify(ciphertext)  # Converting to binary bits
        binary_string = bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)
        print("\n[")
        for i in range(0, len(binary_string), 8):
            row = binary_string[i:i + 8]
            print("    " + ", ".join(map(str, row)) + ",")
        print("]")
        print(f"\n[yellow]The above CipherText is appended to the IV.[/yellow]")
        data = IV + ciphertext
        print(f"\n The resultant Cipher Text after it is appended to IV is : ")
        print(f"[green]{IV}[/green] + [red]{ciphertext}[/red]")
        print(f"\n[yellow]The Resultant Cipher Text is:[yellow]")
        print(f"\n[yellow]{data}[/yellow]")

    if encode:
        print("\nThe obtained ciphertext is encoded using BASE-64 encoding: \n")
        final_ciphertext = base64.b64encode(data).decode()  # Base-64 Encoding
        print(f"[green]{base64.b64encode(data).decode()}[/green]")
        ascii_values = [ord(char) for char in final_ciphertext]
        binary_values = [format(value, '08b') for value in ascii_values]
        bit_string = ''.join(binary_values)
        print("[yellow]\nThe BASE-64 encoded ciphertext bits: [/yellow] ")
        print("\n[")
        for i in range(0, len(bit_string), 8):
            row = bit_string[i:i + 8]
            print("[green]    " + ", ".join(map(str, row)) + ",")
        print("]")
        print(f"[yellow]\nLength of the BASE-64 encoded text in bytes[/yellow] : {len(bit_string)//8}")
        print(f"[yellow]\nLength of the BASE-64 encoded text in bits[/yellow] : {len(bit_string)}")
        encryption_time = time.time() - start_time  # Calculating the encryption time
        print(f"\n[red]Encryption Time: {encryption_time:.5f} seconds[/red]")
        return base64.b64encode(data).decode()
    else:
        return data

    
def decrypt(key, source, decode=True, print_output=True):
    start_time = time.time()
    key_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
    key_hash.update(key)
    key = key_hash.finalize()

    # Decode the base64-encoded data
    source = base64.b64decode(source)

    IV = source[:algorithms.Camellia.block_size // 8]
    source = source[algorithms.Camellia.block_size // 8:]
    decryptor = Cipher(algorithms.Camellia(key), modes.CFB(IV), backend=default_backend()).decryptor()
    padded_data = decryptor.update(source) + decryptor.finalize()

    # Use PKCS7 unpadding here
    unpadder = PKCS7(algorithms.Camellia.block_size).unpadder()
    decrypted_data = unpadder.update(padded_data) + unpadder.finalize()

    if decode:
        print("\nThe obtained decrypted plaintext is decoded using BASE-64 decoding: \n")
        decoded_text = decrypted_data.decode('utf-8')
        print(f"[green]{decoded_text}[/green]")
        ascii_values = [ord(char) for char in decoded_text]
        binary_values = [format(value, '08b') for value in ascii_values]
        bit_string = ''.join(binary_values)
        print("[yellow]\nThe BASE-64 decoded plaintext bits: [/yellow] ")
        print("\n[")
        for i in range(0, len(bit_string), 8):
            row = bit_string[i:i + 8]
            print("[green]    " + ", ".join(map(str, row)) + ",")
        print("]")
        print(f"[yellow]\nLength of the BASE-64 decoded text in bits[/yellow] : {len(bit_string)}")
        decryption_time = time.time() - start_time  # Calculating the decryption time
        print(f"\n[red]Decryption Time: {decryption_time:.5f} seconds[/red]")
        return decoded_text
    else:
        return decrypted_data


#To convert the headertext to binary bits
def text_header_bits(header_text):
    header_bits = []
    for char in header_text:
        # Converting each character to its ASCII value and then to binary
        binary_value = format(ord(char), '08b')
        # Appending each bit as an integer to the header bits list
        header_bits.extend([int(bit) for bit in binary_value])
    print("\n")
    print(f"[red]Details about the header text and header bits: [/red]" )
    print(f"\nThis header text is combined with the base64 encoded ciphertext and is encoded")
    print(f"\nHeader Text is : [green]{header_text}[/green]")
    print(f"\nLength of the Header Text in Bytes is : {len(header_text)}")
    print("\nLength of the header bits in Bits is:", len(header_bits))
    print("\n[yellow]Header bits:[/yellow]")
    print("\n[")
    for i in range(0, len(header_bits), 8):  
        row = header_bits[i:i + 8]  
        print("[green]    " + ", ".join(map(str, row)) + ",")
    print("]")
    return None


#Code for AES-256 Decryption function

#Function used for LSB EMBEDDING STEGANOGRAPHY FOR ENCODING THE IMAGE
def encodeImage(image, message, filename):
    with console.status("[green]Encoding image......") as status:
        try:
            width, height = image.size
            pix = image.getdata()
            current_pixel = 0
            tmp = 0
            x = 0
            y = 0
            original_pixels = list(image.getdata())
            for ch in message:
                binary_value = format(ord(ch), '08b')
                p1 = pix[current_pixel]
                p2 = pix[current_pixel + 1]
                p3 = pix[current_pixel + 2]
                three_pixels = [val for val in p1 + p2 + p3]
                for i in range(0, 8):
                    current_bit = binary_value[i]
                    if current_bit == '0':
                        if three_pixels[i] % 2 != 0:
                            three_pixels[i] = three_pixels[i] - 1 if three_pixels[i] == 255 else three_pixels[i] + 1

                    elif current_bit == '1':
                        if three_pixels[i] % 2 == 0:
                            three_pixels[i] = three_pixels[i] - 1 if three_pixels[i] == 255 else three_pixels[i] + 1

                current_pixel += 3
                tmp += 1

                if tmp == len(message):
                    if three_pixels[-1] % 2 == 0:
                        three_pixels[-1] = three_pixels[-1] - 1 if three_pixels[-1] == 255 else three_pixels[-1] + 1
                else:
                    if three_pixels[-1] % 2 != 0:
                        three_pixels[-1] = three_pixels[-1] - 1 if three_pixels[-1] == 255 else three_pixels[-1] + 1

                three_pixels = tuple(three_pixels)

                st = 0
                end = 3

                for i in range(0, 3):
                    image.putpixel((x, y), three_pixels[st:end])
                    st += 3
                    end += 3

                    if x == width - 1:
                        x = 0
                        y += 1
                    else:
                        x += 1

            print("\n\n")
            print(f"[red]Details while Encoding the message to the image: [/red]")
            print("[yellow]Original File: [u]%s[/u][/yellow]" % filename)

            encoded_filename = filename.split('.')[0] + "-encoded.png"
            image.save(encoded_filename)

            print("[green]Image encoded and saved as [u][bold]%s[/green][/u][/bold]" % encoded_filename)

            # Compare original and encoded images
            encoded_image = Image.open(encoded_filename)
            encoded_pixels = list(encoded_image.getdata())

            num_pixels_changed = sum([1 for orig, enc in zip(original_pixels, encoded_pixels) if orig != enc])
            total_pixels = len(original_pixels)
            percent_change = (num_pixels_changed / total_pixels) * 100

            print(f"\n[red]Comparison between Original and Encoded Image:[/red]")
            print(f"\n[yellow]Total Pixels in the Image:[/yellow] {total_pixels}")
            print(f"[yellow]\nNumber of Pixels Changed:[/yellow] {num_pixels_changed}")
            print(f"[yellow]\nPercentage Change:[/yellow] {percent_change:.10f}%")
            max_line_length=1200
            if num_pixels_changed > 0 and num_pixels_changed < 2000:
                draw = ImageDraw.Draw(encoded_image)
                line_color = (255, 0, 0)  # Red color
                line_thickness = 20
                total_line_length = num_pixels_changed % max_line_length
                # Calculate the remaining space on the current line
                remaining_space = max_line_length - total_line_length
                # Determine the length of the line for the current change
                line_length = min(num_pixels_changed, remaining_space)
                # Start the line from the left corner and extend it horizontally
                draw.line((0, 0, line_length, 0), fill=line_color, width=line_thickness)
                encoded_image_with_line_filename = filename.split('.')[0] + "-encoded-with-line.png"
                encoded_image.save(encoded_image_with_line_filename)
                encoded_image.show()
            else:
                draw = ImageDraw.Draw(encoded_image)
                line_color = (255, 0, 0)  # Red color
                line_thickness = 30
                total_line_length = num_pixels_changed % max_line_length
                # Calculate the remaining space on the current line
                remaining_space = max_line_length - total_line_length
                # Determine the length of the line for the current change
                line_length = min(num_pixels_changed, remaining_space)
                # Start the line from the left corner and extend it horizontally
                draw.line((0, 0, line_length, 0), fill=line_color, width=line_thickness)
                encoded_image_with_line_filename = filename.split('.')[0] + "-encoded-with-line.png"
                encoded_image.save(encoded_image_with_line_filename)
                encoded_image.show()
                

            encoded_filename = filename.split('.')[0] + "-encoded.png"
            image.save(encoded_filename)
            print("[green]Image encoded and saved as [u][bold]%s[/green][/u][/bold]" % encoded_filename)
        except Exception as e:
            print("[red]An error occurred - [/red]%s" % e)
            sys.exit(0)

        
#Function for LSB EMBEDDING STEGANOGRAPHY DECODING THE IMAGE
def decodeImage(image):
    with console.status("[green]Decoding image.....") as status:
        try:
            pix = image.getdata()
            current_pixel = 0
            decoded = ""
            while True:
                binary_value = ""
                p1 = pix[current_pixel]
                p2 = pix[current_pixel + 1]
                p3 = pix[current_pixel + 2]
                three_pixels = [val for val in p1 + p2 + p3]

                for i in range(0, 8):
                    if three_pixels[i] % 2 == 0:
                        binary_value += "0"
                    elif three_pixels[i] % 2 != 0:
                        binary_value += "1"

                binary_value.strip()
                ascii_value = int(binary_value, 2)
                decoded += chr(ascii_value)
                current_pixel += 3

                if three_pixels[-1] % 2 != 0:
                    break

            return decoded
        except Exception as e:
            print("[red]An error occurred - [/red]%s" % e)
            sys.exit(0)



#Main Function
def main():
    print("\n\n")
    print("[red]Choose one from below:\n [/red]")
    op = int(input("1. Sender: Encrypt and Encode the Text\n2. Receiver: Decode and Decrypt the Text\n\n>>"))

    if op == 1:
        print("\n[red]Details about the Image: [/red]\n")
        print("[yellow]Enter the image path with the extension: [/yellow]")
        img=input(">>")
        original_image_path = img
        print("\n[red]Original Image:[/red]")
        print("[yellow]\nThe Image u have selected to Encode the Message is:[/yellow]")
        show_image(original_image_path)
        if not path.exists(img) or not is_png_image(img) or not img.endswith(".png"):
            print("\n[red]Image not found or not in PNG format. Please provide a valid PNG image.[/red]")
            return
        total_pixels=getPixelCount(original_image_path)
        print(f"\n[yellow]Total number of pixels in the image:[/yellow] [green] {total_pixels}[/green]")
        print(f"\n[yellow]Maximum message that can be encoded into the Image is {total_pixels //3} bytes[/yellow]")

        print("[red]\nDetails about the message:[/red]")
        print("\n[yellow]Enter the Input Text File Name: [/yellow]")
        file_path = input(">>")
        try:
            with open(file_path, 'r') as file:
                message = file.read()
            print("[green]\nMessage loaded from file successfully.[/green]")
        except Exception as e:
            print(f"[red]Error loading message from file: {e}[/red]")
            sys.exit(0)

        plaintext_bits = ''.join(format(ord(char), '08b') for char in message)
        print("\n[red]Details about the plaintext: [/red]")
        print("[yellow]\nPlaintext: \n[/yellow]")
        print(message)
        print("\n[yellow]Plaintext Bits:[/yellow]\n")
        print("[")
        for i in range(0, len(plaintext_bits), 8):  
            row = plaintext_bits[i:i + 8]  
            print("    " + ", ".join(map(str, row)) + ",")
        print("]\n")
        print(f"Length of Plaintext in Bytes: [blue]{len(message)}[/blue]")
        print(f"Length of Plaintext in Bits: [blue]{len(plaintext_bits)}[/blue]")
        if (len(message) * 3 > getPixelCount(img,display=False)):
            raise Exception("Given message is too long to be encoded in the image.")

        password = ""
        while True:
            print("\n[red]Details about the password: [/red]")
            print("\n[yellow]Enter your password (at least 12 characters): [/yellow]")
            password = getpass.getpass("Password: ")
            if len(password) < 12:
                print("[red]Password must be at least 12 characters long. Try again.[/red]")
            else:
                print("[yellow]\nRe-confirm the password: [/yellow]")
                confirm_password = getpass.getpass("Password: ")
                if password != confirm_password:
                    print("[red]Passwords do not match. Try again.[/red]")
                else:
                    break

        cipher = ""
        if password != "":
            cipher = encrypt(key=password.encode(), source=message.encode(),print_output=True)
            cipher=headertext+cipher

        else:
            cipher = message

        if DEBUG:
            print("[yellow]Encrypted : [/yellow]", cipher)

        image = Image.open(img)
        print("\n")
        print("[yellow]Image Mode: [/yellow]%s" % image.mode)
        if image.mode != 'RGB':
            output_filename = "converted_" + os.path.splitext(img)[0] + ".png"
            converted_image_path = convertToRGBAndSave(image, output_filename=output_filename)
            if converted_image_path:
                image = Image.open(converted_image_path)
            else:
                print("[red]Conversion to RGB and saving is failed")
        text_header_bits(headertext)

        print(f"\nThe header text that is used to combine with the ciphertext before encoding the total message : \n [green]{headertext}[/green] ")
        print(f"\nThe text that is encoded into the image is: \n[green]{cipher}[/green] ")
        print(f"[yellow]\n The encoded text bits are: \n[/yellow]")
        total_cipher_bits = ''.join(format(ord(char), '08b') for char in cipher) 
        print("\n[")
        for i in range(0, len(total_cipher_bits), 8):
            row = total_cipher_bits[i:i + 8]
            print("[green]    " + ", ".join(map(str, row)) + ",")
        print("\n]\n")

        print(f"\nThe length of the total encoded text in Bytes is : {len(cipher)}")
        cipher_in_bits=len(cipher)*8
        print(f"\nThe length of the total encoded text in Bits is : {cipher_in_bits}")
        pixels_used= cipher_in_bits *3 // 8
        print(f"\nNumber of status bits that are used here are : {(cipher_in_bits * 3 //8) // 3}")

        newimg = image.copy()
        encodeImage(image=newimg, message=cipher, filename=image.filename)
        
        print(f"[green]The Encoded Image is:[/green] ")
        show_image(image.filename)

    elif op == 2:
        print("[yellow]\nEnter the image path with the extension: [/yellow]")
        img = input(">>")
        if not path.exists(img) or not is_png_image(img) or not img.endswith(".png"):
            raise Exception(f"[red]Image not found![/red]")
        print(f"[red]Details about the password: [/red]")
        print(f"\n[yellow]Enter password that is entered at encode option:[/yellow]\n")
        password = getpass.getpass("Password: ")

        print(f"\n[green]The Password is Correct ! ! ![/green]")
        print(f"\n[yellow]The image that needs to be decoded is: [/yellow]")

        image = Image.open(img)
        show_image(image.filename)

        print(f"\n[red]Details about decoding of the image: [/red]")

        cipher = decodeImage(image)

        print(f"[yellow]\nThe Decoded Text from the image is:[/yellow]")
        print(f"[green]{cipher}[/green]")
        print(f"[yellow]\nThe Length of the Encoded Text in the image in Bytes: [/yellow][green]{len(cipher)} [/green]")
        print(f"\n[green]Decoding is successfully completed. \n[/green]")
        header = cipher[:len(headertext)]
        print(f"\n[green]Validating the header ! ! ! [/green]")
        print(f"[yellow]\nThe header decoded from the image is:[/yellow] [green]{header}[/green]")

        if header.strip() != headertext:
            print("[red]The header is not correct. Hence the data is Invalid data![/red]")
            sys.exit(0)
        else:
            print(f"[green]\n Header is validated[/green]......[yellow]You can proceed to Decrypt the data.[/yellow]")

        if DEBUG:
            print("[yellow]The finally Decoded text that the Sender Sended is : \n%s[/yellow]" % cipher)

        decrypted = ""

        if password != "":
            cipher = cipher[len(headertext):]
            try:
                print(f"\n[yellow]The Decoded Text (Not including HeaderText) from the Image is :[/yellow] \n[green]%s[/green]" % cipher)
                decrypted = decrypt(key=password.encode(), source=cipher.encode(), decode=True)
                print(f"[yellow]\nThe Decoded Text should be Decrypted....[/yellow]")
                print(f"[yellow]\nAfter Decryption, the Decrypted Text: [/yellow] [green]{decrypted}[/green]")
                print("[green]\nDecryption successful......! ! !\n\n[/green]")
            except Exception as e:
                print("[red]\nDecryption failed! Please check the password and try again.[/red]")
                print(f"[red]Error details: {e}[/red]")
                sys.exit(0)
        else:
            decrypted = cipher
        print(f"[yellow]The Final Text that the receiver received is:[/yellow] [green][bold]\n{decrypted}[/bold][/green]")
        output_file_path = "decoded_message.txt"
        with io.open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(decrypted)
        print(f"\n[red]Decrypted message saved to {output_file_path} successfully.[/red]")
        input("")
        subprocess.run(["python", "project5.py"])


    
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n\n\n")
    cprint("MAJOR PROJECT".center(os.get_terminal_size().columns), 'red', attrs=['bold'])
    print("\n\n")
    # Print Team heading
    cprint("TEAM 1".center(os.get_terminal_size().columns), 'yellow', attrs=['bold'])
    print("\n\n\n")
    cprint("PROJECT : ENHANCED IMAGE STEGANOGRAPHY WITH DUAL AUTHENTICATION".center(os.get_terminal_size().columns), 'cyan')
    cprint("AND CAMELLIA CIPHER ENCRYPTION".center(os.get_terminal_size().columns), 'cyan')
    print("\n\n")
    cprint(" STAGE 4: LSB IMAGE STEGANOGRAPHY WITH CAMELLIA CIPHER ENCRYPTION ".center(os.get_terminal_size().columns), 'green')
    main()


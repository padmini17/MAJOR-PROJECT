from Crypto.Hash import HMAC
from Crypto.Hash import SHA256
from Crypto.Util.number import getPrime
from PIL import Image                   
from os import path
from termcolor import cprint        
from pyfiglet import figlet_format    
from rich import print                    
from rich.console import Console
import os                                   
from base64 import b64encode, b64decode  
import sys
import time
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import subprocess



DEBUG = False                       #Used to display the debug information
console = Console()    


def sender_partial_dh():
    p = getPrime(16)  # 0 - 65535
    g = getPrime(8)   # 0 - 255
    sender_private_key = getPrime(8)  # Sender's private key as a pseudo-random prime number

    # Compute partial public value
    x = (g ** sender_private_key) % p

    cprint("\nPublic Keys:",attrs=['underline'])
    print(f"\nPublic Parameters (p, g) are : ({p}, {g})")
    cprint("\n\nSender's Private Key:",attrs=['underline'])
    print(f"\nSender's Private Key (sender_private_key) is : {sender_private_key}")
    cprint("\n\nSender's Partial Public Value Calculation:",attrs=['underline'])
    print(f"\nx = (g ^ sender_private_key mod p)")
    print(f"\nx = ({g} ^{sender_private_key} mod {p}) = {x}\n")
    print(f"Sender's Partial Public Value (x) is : {x}")

    return p, g, sender_private_key, x

def receiver_partial_dh():
    # Receiver's partial Diffie-Hellman parameters
    receiver_private_key = getPrime(8)  # Receiver's private key as a pseudo-random prime number
    cprint("******************************  RECEIVER  ******************************".center(os.get_terminal_size().columns), 'red', attrs=['bold'])

    # Securely exchange parameters p and g
    cprint("\nThe public parameters are shared by the sender only to the receiver.",'cyan')
    p = int(input("\nEnter the Public Key (p) shared by the sender: "))
    g = int(input("\nEnter the Public Key (g) shared by the sender: "))

    # Compute partial public value
    y = (g ** receiver_private_key) % p  

    # Print the values for the receiver

    cprint("\n\nReceiver Private Key:",attrs=['underline'])
    print(f"\nReceiver's Private Key (receiver_private_key) is: {receiver_private_key}")
    cprint("\n\nReceiver's Partial Public Value Calculation:",attrs=['underline'])
    print(f"\ny = (g ^ receiver_private_key mod p)")
    print(f"\ny = ({g} ^ {receiver_private_key} mod {p}) = {y}\n")
    print(f"\nThe Computed Partial Public Value from the receiver's end is (y) : {y}")
    user_input = int(input("\nEnter the Partial Public Key Value generated (y) that is computed from the receiver side: "))
    print("\n\n")

    return receiver_private_key, y, user_input

def complete_dh(p, g, private_key, partial_public):
    # Complete Diffie-Hellman computation
    shared_secret = (partial_public ** private_key) % p
    return shared_secret


def main():
    print("\n")
    # Print the values for the sender
    cprint("(1) DIFFIE HELLMAN KEY EXCHANGE AUTHENTICATION\n\n".center(os.get_terminal_size().columns), 'yellow', attrs=['bold'])
    cprint("******************************  SENDER  ******************************".center(os.get_terminal_size().columns), 'red', attrs=['bold'])
        # Sender's partial DH
    p, g, sender_private_key, x = sender_partial_dh()
    
    print("\n")

    # Receiver's partial DH
    receiver_private_key, expected_y, received_y = receiver_partial_dh()

    # Sender gets missing part from receiver
    partial_public_sender = received_y
    cprint("The Sender and Receiver will interchange the Partial Public Key Values (x) and (y) respectively.\n",'cyan')
    print(f"Then, the Secret Key at Sender's & Receiver's side is computed.\n")

    # Receiver gets missing part from sender
    partial_public_receiver = x

    # Complete DH computation

    shared_secret_sender = complete_dh(p, g, sender_private_key, partial_public_sender)
    cprint("\nShared Secret Key Calculation at Sender Side:",attrs=['underline'])
    print(f"\nShared Secret: (received_y) ** (sender_private_key) % (p) = (shared_secret_sender)\n")
    print(f"\nShared Secret: ({received_y} ** {sender_private_key}) % {p} = {shared_secret_sender}\n")
    shared_secret_receiver = complete_dh(p, g, receiver_private_key, partial_public_receiver)
    cprint("\nShared Secret Key Calculation at Receiver Side:",attrs=['underline'])
    print(f"\nShared Secret: (x) ** (receiver_private_key) % (p) = (shared_secret_receiver)\n")
    print(f"\nShared Secret: ({x} ** {receiver_private_key}) % {p} = {shared_secret_receiver}\n")

    if shared_secret_sender == shared_secret_receiver:
        cprint("\n\nDiffie-Hellman Key Exchange: Authentication successful!", 'green')
        cprint("\n\nSender Authentication: Successful!", 'green')
        cprint("\nReceiver Authentication: Successful!", 'green')
        cprint("\nDual Authentication: Successful!", 'green')
        input("")
        subprocess.run(["python","project3.py"])
    else:
        cprint("\nSender Authentication: Failed!", 'red')
        cprint("\nReceiver Authentication: Failed!", 'red')
        cprint("Dual Authentication: Failed!", 'red')
        cprint("\nOh! Beware ! ! ! Sender is sending the information to the wrong receiver.",'red')
        sys.exit(0)
  
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n\n\n")
    # Print Major Project heading
    cprint("MAJOR PROJECT".center(os.get_terminal_size().columns), 'red', attrs=['bold'])
    print("\n\n")
    # Print Team heading
    cprint("TEAM 1".center(os.get_terminal_size().columns), 'yellow', attrs=['bold'])
    print("\n\n\n")
    cprint("PROJECT : ENHANCED IMAGE STEGANOGRAPHY WITH DUAL AUTHENTICATION".center(os.get_terminal_size().columns), 'cyan')
    cprint("AND CAMELLIA CIPHER ENCRYPTION".center(os.get_terminal_size().columns), 'cyan')
    print("\n\n")
    cprint(" STAGE 2: DUAL AUTHENTICATION USING DIFFIE HELLMAN".center(os.get_terminal_size().columns), 'green')
    print("\n")
    main()

import hashlib
import base64


def print_format(digest, hex_value, base64_hex):
    print('Digest (%s)                      >>> %s' % (len(digest), digest))
    print('Hexadecimal (%s)                 >>> %s' % (len(hex_value), hex_value))
    print('Decoded Base64 of hex value (%s) >>> %s' % (len(base64_hex), base64_hex))


def md5_usage():
    print("MD5 Usage")
    hash_function = hashlib.md5()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


def sha1_usage():
    print("\nSHA1 Usage")
    hash_function = hashlib.sha1()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


def sha224_usage():
    print("\nSHA224 Usage")
    hash_function = hashlib.sha224()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


def sha256_usage():
    print("\nSHA256 Usage")
    hash_function = hashlib.sha256()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


def sha384_usage():
    print("\nSHA384 Usage")
    hash_function = hashlib.sha384()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


def sha512_usage():
    print("\nSHA512 Usage")
    hash_function = hashlib.sha512()
    hash_function.update(b'Hello World')
    hex_value = hash_function.hexdigest()
    print_format(hash_function.digest(), hex_value, base64.b64decode(hex_value))


if __name__ == '__main__':
    md5_usage()
    sha1_usage()
    sha224_usage()
    sha256_usage()
    sha384_usage()
    sha512_usage()

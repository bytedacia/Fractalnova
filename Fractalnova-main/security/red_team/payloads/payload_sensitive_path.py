# RED TEAM PAYLOAD - I/O percorsi sensibili (solo per test detection)

def load_config():
    with open("/etc/passwd") as f:
        return f.read()
    with open("/root/.ssh/id_rsa") as k:
        return k.read()

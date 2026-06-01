# RED TEAM PAYLOAD - eval/exec (solo per test detection)

def helper():
    eval("os.system('id')")
    exec(open("/etc/passwd").read())

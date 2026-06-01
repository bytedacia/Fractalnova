#!/usr/bin/env python3
"""
Script di avvio per il sistema di sicurezza ultra-avanzato
Controlla TUTTO il codice 24/7 inclusi i file di sicurezza stessi
"""

import os
import sys
import json
import time
import signal
from datetime import datetime
from pathlib import Path

def print_banner():
    """Stampa banner del sistema"""
    print("""
🛡️ SISTEMA DI SICUREZZA ULTRA-AVANZATO
=====================================
Protezione 24/7 per codice e file di sicurezza
Auto-monitoraggio e auto-protezione attivi
""")

def check_requirements():
    """Verifica requisiti del sistema"""
    print("🔍 Verifica requisiti...")
    
    # Verifica Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ richiesto")
        return False
    
    # Verifica dipendenze
    required_modules = [
        'cryptography', 'hashlib', 'json', 'threading', 
        'subprocess', 'ast', 're', 'os', 'time'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Moduli mancanti: {', '.join(missing)}")
        print("💡 Esegui: pip install -r requirements.txt")
        return False
    
    print("✅ Tutti i requisiti soddisfatti")
    return True

def initialize_security_system(project_root: str):
    """Inizializza il sistema di sicurezza"""
    print("🔧 Inizializzazione sistema di sicurezza...")
    
    try:
        # Importa componenti
        from security.ultra_guard import UltraSecurityGuard
        from security.self_monitor import SecuritySelfMonitor
        from security.advanced_verifier import AdvancedSecurityVerifier
        from security.backup_manager import SecureBackupManager
        from security.auto_repair import AutoRepairSystem
        
        # Crea directory di sicurezza se non esiste
        security_dir = os.path.join(project_root, "security")
        os.makedirs(security_dir, exist_ok=True)
        
        # Inizializza guard
        guard = UltraSecurityGuard(project_root)
        
        print("✅ Sistema di sicurezza inizializzato")
        return guard
        
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        return None

def create_initial_baseline(guard):
    """Crea baseline iniziale se non esiste"""
    print("📋 Creazione baseline di sicurezza...")
    
    try:
        if guard.self_monitor:
            baseline_exists = os.path.exists(guard.self_monitor.baseline_file)
            
            if not baseline_exists:
                guard.self_monitor.create_baseline()
                print("✅ Baseline iniziale creata")
            else:
                print("✅ Baseline esistente trovata")
        else:
            print("⚠️ Self-monitor non disponibile")
            
    except Exception as e:
        print(f"❌ Errore creazione baseline: {e}")

def run_initial_scan(guard):
    """Esegue scansione iniziale"""
    print("🔍 Esecuzione scansione iniziale...")
    
    try:
        # Scansione di emergenza
        results = guard.run_emergency_scan()
        
        threats_found = results.get('threats_found', 0)
        if threats_found > 0:
            print(f"⚠️ {threats_found} threat rilevati nella scansione iniziale")
            return False
        else:
            print("✅ Sistema pulito - nessun threat rilevato")
            return True
            
    except Exception as e:
        print(f"❌ Errore scansione iniziale: {e}")
        return False

def start_continuous_protection(guard):
    """Avvia protezione continua"""
    print("🚀 Avvio protezione continua 24/7...")
    print("   Premi Ctrl+C per fermare")
    
    try:
        # Avvia ultra guard
        guard.start_ultra_guard()
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Fermata richiesta dall'utente")
        return True
    except Exception as e:
        print(f"❌ Errore durante protezione: {e}")
        return False

def show_status(guard):
    """Mostra stato del sistema"""
    print("📊 STATO SISTEMA DI SICUREZZA")
    print("=" * 40)
    
    try:
        status = guard.get_status()
        
        print(f"Guard Attivo: {'✅' if status['guard_active'] else '❌'}")
        print(f"Modalità Emergenza: {'🚨' if status['emergency_mode'] else '✅'}")
        print(f"Scansioni Eseguite: {status['stats']['scans_performed']}")
        print(f"Threat Bloccati: {status['stats']['threats_blocked']}")
        print(f"Risposte Emergenza: {status['stats']['emergency_responses']}")
        print(f"Ultima Scansione: {status['stats']['last_scan'] or 'Mai'}")
        
        print("\nComponenti:")
        for component, loaded in status['components_loaded'].items():
            print(f"  {component}: {'✅' if loaded else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore lettura stato: {e}")
        return False

def main():
    """Funzione principale"""
    print_banner()
    
    # Ottieni directory progetto
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    project_root = os.path.abspath(project_root)
    
    print(f"📁 Directory progetto: {project_root}")
    
    # Verifica requisiti
    if not check_requirements():
        sys.exit(1)
    
    # Inizializza sistema
    guard = initialize_security_system(project_root)
    if not guard:
        sys.exit(1)
    
    # Gestisci comandi
    if len(sys.argv) > 2:
        command = sys.argv[2]
        
        if command == "status":
            show_status(guard)
            return
        
        elif command == "baseline":
            create_initial_baseline(guard)
            return
        
        elif command == "scan":
            if run_initial_scan(guard):
                print("✅ Sistema sicuro")
            else:
                print("⚠️ Threat rilevati - controlla i report")
            return
        
        elif command == "start":
            create_initial_baseline(guard)
            if run_initial_scan(guard):
                start_continuous_protection(guard)
            else:
                print("❌ Impossibile avviare - threat rilevati")
            return
        
        else:
            print("❌ Comando non riconosciuto")
            print("Comandi disponibili: status, baseline, scan, start")
            return
    
    # Modalità interattiva
    print("\n🔧 MODALITÀ INTERATTIVA")
    print("Comandi disponibili:")
    print("  status      - Mostra stato sistema")
    print("  baseline    - Crea baseline iniziale")
    print("  scan        - Esegue scansione una volta")
    print("  backup      - Crea backup completo")
    print("  repair      - Scansiona e ripara file infetti")
    print("  emergency   - Riparazione di emergenza")
    print("  start       - Avvia protezione continua")
    print("  exit        - Esci")
    
    while True:
        try:
            cmd = input("\n🛡️ Ultra Security> ").strip().lower()
            
            if cmd == "exit":
                print("👋 Arrivederci!")
                break
            elif cmd == "status":
                show_status(guard)
            elif cmd == "baseline":
                create_initial_baseline(guard)
            elif cmd == "scan":
                if run_initial_scan(guard):
                    print("✅ Sistema sicuro")
                else:
                    print("⚠️ Threat rilevati")
            elif cmd == "backup":
                print("🔒 Creazione backup completo...")
                backup_results = guard.backup_manager.create_full_backup()
                print(f"✅ Backup completato: {backup_results['files_backed_up']} file protetti")
            elif cmd == "repair":
                print("🔧 Scansione e riparazione file infetti...")
                repair_results = guard.auto_repair.scan_and_repair_all()
                print(f"✅ Riparazione completata: {repair_results['files_repaired']} file riparati")
            elif cmd == "emergency":
                print("🚨 Riparazione di emergenza...")
                emergency_results = guard.auto_repair.emergency_repair()
                if emergency_results['system_clean']:
                    print("✅ Sistema completamente riparato")
                else:
                    print("⚠️ Alcuni file potrebbero essere ancora infetti")
            elif cmd == "start":
                create_initial_baseline(guard)
                if run_initial_scan(guard):
                    start_continuous_protection(guard)
                else:
                    print("❌ Impossibile avviare - threat rilevati")
            else:
                print("❌ Comando non riconosciuto")
                
        except KeyboardInterrupt:
            print("\n👋 Arrivederci!")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()

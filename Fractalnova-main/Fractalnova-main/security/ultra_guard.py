import os
import json
import time
import threading
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import signal

class UltraSecurityGuard:
    """
    Sistema di sicurezza ultra-avanzato che controlla TUTTO 24/7
    inclusi i file di sicurezza stessi con auto-protezione
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.security_dir = os.path.join(project_root, "security")
        self.guard_active = False
        self.emergency_mode = False
        
        # File di configurazione
        self.config_file = os.path.join(self.security_dir, "ultra_guard_config.json")
        self.status_file = os.path.join(self.security_dir, "ultra_guard_status.json")
        self.emergency_log = os.path.join(self.security_dir, "emergency_log.json")
        
        # Carica configurazione
        self.config = self._load_config()
        
        # Inizializza componenti
        self.advanced_verifier = None
        self.self_monitor = None
        self.encryptor = None
        
        # Thread di monitoraggio
        self.monitor_thread = None
        self.verifier_thread = None
        
        # Statistiche
        self.stats = {
            'start_time': None,
            'scans_performed': 0,
            'threats_blocked': 0,
            'emergency_responses': 0,
            'files_protected': 0,
            'last_scan': None
        }
    
    def _load_config(self) -> Dict:
        """Carica configurazione del sistema"""
        default_config = {
            'scan_interval_seconds': 60,  # Scansione ogni minuto
            'verifier_interval_seconds': 300,  # Verificatore ogni 5 minuti
            'emergency_response_enabled': True,
            'auto_encrypt_on_threat': True,
            'max_consecutive_failures': 3,
            'threat_severity_threshold': 'high',
            'monitor_security_files': True,
            'monitor_self_modification': True,
            'emergency_notifications': True,
            'backup_critical_files': True,
            'log_all_activities': True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge con default
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception:
                pass
        
        # Salva configurazione di default
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _initialize_components(self):
        """Inizializza componenti di sicurezza"""
        try:
            # Importa componenti
            from security.advanced_verifier import AdvancedSecurityVerifier
            from security.self_monitor import SecuritySelfMonitor
            from security.encryptor import FileEncryptor
            from security.backup_manager import SecureBackupManager
            from security.auto_repair import AutoRepairSystem
            
            self.advanced_verifier = AdvancedSecurityVerifier(self.project_root)
            self.self_monitor = SecuritySelfMonitor(self.project_root)
            self.encryptor = FileEncryptor()
            self.backup_manager = SecureBackupManager(self.project_root)
            self.auto_repair = AutoRepairSystem(self.project_root)
            
            print("✅ Componenti di sicurezza inizializzati")
            return True
        except Exception as e:
            print(f"❌ Errore inizializzazione componenti: {e}")
            return False
    
    def _save_status(self):
        """Salva stato del sistema"""
        status = {
            'guard_active': self.guard_active,
            'emergency_mode': self.emergency_mode,
            'stats': self.stats,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2)
    
    def _log_activity(self, activity: str, details: Dict = None):
        """Registra attività nel log"""
        if not self.config.get('log_all_activities', True):
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'details': details or {},
            'guard_active': self.guard_active,
            'emergency_mode': self.emergency_mode
        }
        
        log_file = os.path.join(self.security_dir, "ultra_guard_activity.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{log_entry['timestamp']} - {activity}\n")
    
    def _trigger_emergency_response(self, threat_info: Dict):
        """Attiva risposta di emergenza"""
        if not self.config.get('emergency_response_enabled', True):
            return
        
        print("🚨 ATTIVAZIONE RISPOSTA DI EMERGENZA")
        print(f"   Motivo: {threat_info.get('reason', 'Threat sconosciuto')}")
        
        self.emergency_mode = True
        self.stats['emergency_responses'] += 1
        
        # Log emergenza
        emergency_entry = {
            'timestamp': datetime.now().isoformat(),
            'threat_info': threat_info,
            'response_actions': []
        }
        
        # 1. Cripta file critici
        if self.config.get('auto_encrypt_on_threat', True):
            try:
                critical_files = [
                    'inference/generate.py',
                    'inference/orchestrator.py',
                    'app.py',
                    'requirements.txt'
                ]
                
                for file in critical_files:
                    full_path = os.path.join(self.project_root, file)
                    if os.path.exists(full_path):
                        self.encryptor.encrypt_file(full_path)
                        emergency_entry['response_actions'].append(f"Encrypted: {file}")
                        print(f"  🔒 Criptato: {file}")
            except Exception as e:
                print(f"  ❌ Errore criptazione: {e}")
        
        # 2. Backup file critici
        if self.config.get('backup_critical_files', True):
            try:
                backup_dir = os.path.join(self.security_dir, "emergency_backup")
                os.makedirs(backup_dir, exist_ok=True)
                
                # Copia file critici
                import shutil
                for file in ['inference/generate.py', 'inference/orchestrator.py', 'app.py']:
                    src = os.path.join(self.project_root, file)
                    if os.path.exists(src):
                        dst = os.path.join(backup_dir, os.path.basename(file))
                        shutil.copy2(src, dst)
                        emergency_entry['response_actions'].append(f"Backed up: {file}")
            except Exception as e:
                print(f"  ❌ Errore backup: {e}")
        
        # 3. Auto-riparazione di emergenza
        if self.auto_repair:
            try:
                print("🔧 Avvio riparazione di emergenza...")
                repair_results = self.auto_repair.emergency_repair()
                emergency_entry['response_actions'].append(f"Emergency repair: {repair_results['repair_results']['files_repaired']} files repaired")
                print(f"✅ Riparazione emergenza completata: {repair_results['repair_results']['files_repaired']} file riparati")
            except Exception as e:
                print(f"❌ Errore riparazione emergenza: {e}")
                emergency_entry['response_actions'].append(f"Emergency repair failed: {e}")
        
        # 4. Notifica emergenza
        if self.config.get('emergency_notifications', True):
            self._send_emergency_notification(threat_info)
            emergency_entry['response_actions'].append("Emergency notification sent")
        
        # Salva log emergenza
        with open(self.emergency_log, 'w', encoding='utf-8') as f:
            json.dump(emergency_entry, f, indent=2)
        
        self._log_activity("emergency_response_triggered", threat_info)
    
    def _send_emergency_notification(self, threat_info: Dict):
        """Invia notifica di emergenza"""
        print("📧 Invio notifica di emergenza...")
        # Implementa notifica via email, webhook, etc.
    
    def _monitor_loop(self):
        """Loop principale di monitoraggio"""
        print("🛡️ Avvio loop di monitoraggio ultra-avanzato...")
        
        while self.guard_active:
            try:
                # 1. Verifica integrità file
                if self.self_monitor:
                    integrity_result = self.self_monitor.check_integrity()
                    
                    if integrity_result['status'] == 'changes_detected':
                        changes = integrity_result['changes']
                        
                        # Allerta per modifiche ai file di sicurezza
                        if changes.get('security_modified'):
                            self._trigger_emergency_response({
                                'type': 'security_file_modified',
                                'reason': f"File di sicurezza modificati: {changes['security_modified']}",
                                'files': changes['security_modified']
                            })
                        
                        # Allerta per auto-modifica
                        if changes.get('self_modification_found'):
                            self._trigger_emergency_response({
                                'type': 'self_modification_detected',
                                'reason': f"Auto-modifica rilevata in {len(changes['self_modification_found'])} file",
                                'files': changes['self_modification_found']
                            })
                
                # 2. Scansione avanzata (ogni 5 minuti)
                if self.advanced_verifier and self.stats['scans_performed'] % 5 == 0:
                    print("🔍 Esecuzione scansione avanzata...")
                    scan_results = self.advanced_verifier.scan_project_comprehensive()
                    
                    if scan_results['overall_severity'] in ['critical', 'high']:
                        self._trigger_emergency_response({
                            'type': 'advanced_threat_detected',
                            'reason': f"Threat {scan_results['overall_severity']} rilevati",
                            'threats_count': scan_results['threats_found'],
                            'critical_threats': scan_results['critical_threats']
                        })
                
                # 3. Auto-riparazione (ogni 10 minuti)
                if self.auto_repair and self.stats['scans_performed'] % 10 == 0:
                    print("🔧 Esecuzione auto-riparazione...")
                    repair_results = self.auto_repair.scan_and_repair_all()
                    
                    if repair_results['files_infected'] > 0:
                        print(f"🚨 {repair_results['files_infected']} file infetti rilevati e riparati")
                        self._log_activity("auto_repair_performed", repair_results)
                    else:
                        print("✅ Sistema pulito - nessuna riparazione necessaria")
                
                # 4. Backup automatico (ogni 15 minuti)
                if self.backup_manager and self.stats['scans_performed'] % 15 == 0:
                    print("🔒 Esecuzione backup automatico...")
                    backup_results = self.backup_manager.create_full_backup()
                    print(f"✅ Backup completato: {backup_results['files_backed_up']} file protetti")
                
                # Aggiorna statistiche
                self.stats['scans_performed'] += 1
                self.stats['last_scan'] = datetime.now().isoformat()
                self._save_status()
                self._log_activity("monitoring_scan_completed")
                
                # Pausa tra scansioni
                time.sleep(self.config['scan_interval_seconds'])
                
            except Exception as e:
                print(f"❌ Errore nel loop di monitoraggio: {e}")
                self._log_activity("monitoring_error", {'error': str(e)})
                
                # Controlla fallimenti consecutivi
                if hasattr(self, '_consecutive_errors'):
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 1
                
                if self._consecutive_errors >= self.config['max_consecutive_failures']:
                    self._trigger_emergency_response({
                        'type': 'monitoring_failure',
                        'reason': f"Fallimenti consecutivi: {self._consecutive_errors}",
                        'error': str(e)
                    })
                    break
                
                time.sleep(10)  # Pausa breve prima di riprovare
    
    def start_ultra_guard(self):
        """Avvia il sistema di sicurezza ultra-avanzato"""
        print("🛡️ AVVIO SISTEMA DI SICUREZZA ULTRA-AVANZATO")
        print("=" * 60)
        
        # Inizializza componenti
        if not self._initialize_components():
            print("❌ Impossibile inizializzare componenti di sicurezza")
            return False
        
        # Crea baseline se non esiste
        if self.self_monitor and not os.path.exists(self.self_monitor.baseline_file):
            print("📋 Creazione baseline di sicurezza...")
            self.self_monitor.create_baseline()
        
        # Avvia monitoraggio
        self.guard_active = True
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Avvia thread di monitoraggio
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✅ Sistema di sicurezza attivo")
        print(f"   Intervallo scansione: {self.config['scan_interval_seconds']}s")
        print(f"   Monitoraggio file di sicurezza: {self.config['monitor_security_files']}")
        print(f"   Risposta di emergenza: {self.config['emergency_response_enabled']}")
        print("   Premi Ctrl+C per fermare")
        
        self._log_activity("ultra_guard_started")
        
        try:
            # Loop principale
            while self.guard_active:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Fermata richiesta dall'utente")
        except Exception as e:
            print(f"\n❌ Errore critico: {e}")
        finally:
            self.stop_ultra_guard()
        
        return True
    
    def stop_ultra_guard(self):
        """Ferma il sistema di sicurezza"""
        print("🛑 Fermata sistema di sicurezza...")
        
        self.guard_active = False
        
        # Aspetta che i thread finiscano
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self._save_status()
        self._log_activity("ultra_guard_stopped")
        
        print("✅ Sistema di sicurezza fermato")
    
    def get_status(self) -> Dict:
        """Ottiene stato del sistema"""
        status = {
            'guard_active': self.guard_active,
            'emergency_mode': self.emergency_mode,
            'stats': self.stats,
            'config': self.config,
            'components_loaded': {
                'advanced_verifier': self.advanced_verifier is not None,
                'self_monitor': self.self_monitor is not None,
                'encryptor': self.encryptor is not None
            }
        }
        
        return status
    
    def run_emergency_scan(self) -> Dict:
        """Esegue scansione di emergenza immediata"""
        print("🚨 SCANSIONE DI EMERGENZA IMMEDIATA")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'integrity_check': None,
            'advanced_scan': None,
            'threats_found': 0,
            'emergency_actions': []
        }
        
        # 1. Verifica integrità
        if self.self_monitor:
            results['integrity_check'] = self.self_monitor.check_integrity()
            if results['integrity_check']['status'] == 'changes_detected':
                results['threats_found'] += 1
        
        # 2. Scansione avanzata
        if self.advanced_verifier:
            results['advanced_scan'] = self.advanced_verifier.scan_project_comprehensive()
            results['threats_found'] += results['advanced_scan']['threats_found']
        
        # 3. Azioni di emergenza se necessario
        if results['threats_found'] > 0:
            self._trigger_emergency_response({
                'type': 'emergency_scan_threats',
                'reason': f"Threat rilevati durante scansione di emergenza: {results['threats_found']}",
                'threats_count': results['threats_found']
            })
            results['emergency_actions'].append("Emergency response triggered")
        
        return results


def signal_handler(signum, frame):
    """Gestore segnali per fermata pulita"""
    print(f"\n🛑 Segnale {signum} ricevuto - fermata sistema...")
    sys.exit(0)


if __name__ == "__main__":
    import sys
    
    # Registra gestori segnali
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    guard = UltraSecurityGuard(project_root)
    
    if len(sys.argv) > 2:
        command = sys.argv[2]
        
        if command == "start":
            guard.start_ultra_guard()
        elif command == "status":
            status = guard.get_status()
            print(json.dumps(status, indent=2))
        elif command == "emergency_scan":
            results = guard.run_emergency_scan()
            print(json.dumps(results, indent=2))
        else:
            print("Comandi disponibili: start, status, emergency_scan")
    else:
        print("🛡️ SISTEMA DI SICUREZZA ULTRA-AVANZATO")
        print("=" * 50)
        print("Uso: python security/ultra_guard.py <project_root> <command>")
        print("Comandi:")
        print("  start          - Avvia sistema di sicurezza 24/7")
        print("  status         - Mostra stato sistema")
        print("  emergency_scan - Esegue scansione di emergenza")

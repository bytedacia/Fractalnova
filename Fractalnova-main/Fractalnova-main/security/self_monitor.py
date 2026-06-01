import os
import json
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Set
import subprocess
import sys

class SecuritySelfMonitor:
    """
    Sistema di auto-monitoraggio che controlla TUTTI i file 24/7
    inclusi i file di sicurezza stessi per prevenire auto-modifiche
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.security_dir = os.path.join(project_root, "security")
        self.monitor_active = False
        self.baseline_file = os.path.join(self.security_dir, "self_monitor_baseline.json")
        self.log_file = os.path.join(self.security_dir, "self_monitor.log")
        self.alert_file = os.path.join(self.security_dir, "security_alerts.json")
        
        # File critici da monitorare (inclusi file di sicurezza)
        self.critical_files = self._get_all_critical_files()
        
        # Pattern di auto-modifica da rilevare
        self.self_modification_patterns = [
            r'open\s*\(__file__', r'open\s*\(os\.path\.abspath\s*\(__file__\)',
            r'write\s*\(.*__file__', r'exec\s*\(open\s*\(__file__',
            r'compile\s*\(open\s*\(__file__', r'__import__\s*\(.*__file__',
            r'self\.', r'this\.', r'__self__', r'__this__'
        ]
        
        # Soglie di allerta
        self.alert_thresholds = {
            'file_changes_per_hour': 5,
            'security_file_changes': 1,  # Qualsiasi modifica ai file di sicurezza
            'suspicious_patterns': 1,
            'consecutive_failures': 3
        }
        
        self.stats = {
            'scan_count': 0,
            'threats_detected': 0,
            'files_modified': 0,
            'security_files_modified': 0,
            'last_scan': None,
            'consecutive_failures': 0
        }
    
    def _get_all_critical_files(self) -> List[str]:
        """Ottiene TUTTI i file critici da monitorare"""
        critical = []
        
        # File di sicurezza (auto-monitoraggio)
        security_files = [
            'security/advanced_verifier.py',
            'security/self_monitor.py', 
            'security/guard.py',
            'security/verifier.py',
            'security/watchdog.py',
            'security/encryptor.py',
            'security/ci_runner.py'
        ]
        
        for file in security_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        # File di applicazione
        app_files = [
            'app.py',
            'inference/generate.py',
            'inference/orchestrator.py',
            '.github/workflows/security.yml',
            'requirements.txt',
            'README.md'
        ]
        
        for file in app_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        # File di configurazione
        config_files = [
            '.env', '.gitignore', 'pyproject.toml', 'setup.py'
        ]
        
        for file in config_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        return critical
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calcola hash SHA256 di un file"""
        try:
            hasher = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def _scan_file_for_self_modification(self, filepath: str) -> List[Dict]:
        """Scansiona file per pattern di auto-modifica"""
        findings = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import re
            for pattern in self.self_modification_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    findings.append({
                        'pattern': pattern,
                        'line': content[:match.start()].count('\n') + 1,
                        'context': content[max(0, match.start()-50):match.end()+50],
                        'severity': 'critical' if 'security' in filepath else 'high'
                    })
        except Exception as e:
            findings.append({
                'pattern': 'scan_error',
                'error': str(e),
                'severity': 'medium'
            })
        
        return findings
    
    def create_baseline(self) -> Dict:
        """Crea baseline di tutti i file critici"""
        print("🔒 Creazione baseline di sicurezza...")
        
        baseline = {
            'created_at': datetime.now().isoformat(),
            'project_root': self.project_root,
            'files': {}
        }
        
        for filepath in self.critical_files:
            if os.path.exists(filepath):
                file_hash = self._calculate_file_hash(filepath)
                if file_hash:
                    baseline['files'][filepath] = {
                        'hash': file_hash,
                        'size': os.path.getsize(filepath),
                        'modified': os.path.getmtime(filepath),
                        'is_security_file': 'security' in filepath
                    }
                    print(f"  ✅ {os.path.relpath(filepath, self.project_root)}")
                else:
                    print(f"  ❌ Errore hash: {os.path.relpath(filepath, self.project_root)}")
        
        # Salva baseline
        with open(self.baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"✅ Baseline creata: {len(baseline['files'])} file monitorati")
        return baseline
    
    def check_integrity(self) -> Dict:
        """Verifica integrità di tutti i file critici"""
        if not os.path.exists(self.baseline_file):
            return {'status': 'no_baseline', 'message': 'Baseline non trovata'}
        
        # Carica baseline
        try:
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
        except Exception as e:
            return {'status': 'error', 'message': f'Errore caricamento baseline: {e}'}
        
        changes = {
            'modified': [],
            'added': [],
            'deleted': [],
            'security_modified': [],
            'self_modification_found': []
        }
        
        # Verifica file esistenti
        for filepath, baseline_info in baseline['files'].items():
            if not os.path.exists(filepath):
                changes['deleted'].append(filepath)
                if baseline_info.get('is_security_file'):
                    changes['security_modified'].append(filepath)
            else:
                current_hash = self._calculate_file_hash(filepath)
                if current_hash != baseline_info['hash']:
                    changes['modified'].append(filepath)
                    if baseline_info.get('is_security_file'):
                        changes['security_modified'].append(filepath)
                    
                    # Controlla auto-modifica
                    self_mod_findings = self._scan_file_for_self_modification(filepath)
                    if self_mod_findings:
                        changes['self_modification_found'].append({
                            'file': filepath,
                            'findings': self_mod_findings
                        })
        
        # Verifica file nuovi
        for filepath in self.critical_files:
            if filepath not in baseline['files'] and os.path.exists(filepath):
                changes['added'].append(filepath)
                if 'security' in filepath:
                    changes['security_modified'].append(filepath)
        
        # Aggiorna statistiche
        self.stats['scan_count'] += 1
        self.stats['last_scan'] = datetime.now().isoformat()
        self.stats['files_modified'] = len(changes['modified']) + len(changes['added'])
        self.stats['security_files_modified'] = len(changes['security_modified'])
        
        if changes['modified'] or changes['added'] or changes['deleted']:
            self.stats['threats_detected'] += 1
        else:
            self.stats['consecutive_failures'] = 0
        
        return {
            'status': 'ok' if not any(changes.values()) else 'changes_detected',
            'changes': changes,
            'stats': self.stats
        }
    
    def _log_alert(self, alert_type: str, message: str, details: Dict = None):
        """Registra allerta nel log"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'details': details or {}
        }
        
        # Aggiungi al file di log
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{alert['timestamp']} - {alert_type}: {message}\n")
        
        # Aggiungi al file di allerte
        alerts = []
        if os.path.exists(self.alert_file):
            try:
                with open(self.alert_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)
            except:
                pass
        
        alerts.append(alert)
        
        # Mantieni solo le ultime 100 allerte
        if len(alerts) > 100:
            alerts = alerts[-100:]
        
        with open(self.alert_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2)
    
    def _trigger_emergency_response(self, reason: str):
        """Attiva risposta di emergenza"""
        print(f"🚨 EMERGENZA SICUREZZA: {reason}")
        
        # Log allerta critica
        self._log_alert('emergency', f'Risposta di emergenza attivata: {reason}')
        
        # Cripta file critici
        try:
            from security.encryptor import FileEncryptor
            encryptor = FileEncryptor()
            
            critical_to_encrypt = [
                'inference/generate.py',
                'inference/orchestrator.py',
                'app.py'
            ]
            
            for file in critical_to_encrypt:
                full_path = os.path.join(self.project_root, file)
                if os.path.exists(full_path):
                    encryptor.encrypt_file(full_path)
                    print(f"  🔒 Criptato: {file}")
        except Exception as e:
            print(f"  ❌ Errore criptazione: {e}")
        
        # Notifica (se configurato)
        self._send_emergency_notification(reason)
    
    def _send_emergency_notification(self, reason: str):
        """Invia notifica di emergenza"""
        # Implementa notifica via email, webhook, etc.
        print(f"📧 Notifica emergenza: {reason}")
    
    def start_continuous_monitoring(self, interval_seconds: int = 300):
        """Avvia monitoraggio continuo 24/7"""
        print("🛡️ Avvio monitoraggio di sicurezza continuo...")
        print(f"   Intervallo: {interval_seconds} secondi")
        print("   Premi Ctrl+C per fermare")
        
        self.monitor_active = True
        
        try:
            while self.monitor_active:
                # Verifica integrità
                result = self.check_integrity()
                
                if result['status'] == 'changes_detected':
                    changes = result['changes']
                    
                    # Allerta per modifiche ai file di sicurezza
                    if changes['security_modified']:
                        self._log_alert(
                            'security_breach',
                            f'File di sicurezza modificati: {changes["security_modified"]}',
                            changes
                        )
                        
                        # Attiva risposta di emergenza
                        self._trigger_emergency_response(
                            f"File di sicurezza modificati: {changes['security_modified']}"
                        )
                    
                    # Allerta per auto-modifica
                    if changes['self_modification_found']:
                        self._log_alert(
                            'self_modification',
                            f'Pattern di auto-modifica rilevati: {len(changes["self_modification_found"])} file',
                            changes['self_modification_found']
                        )
                        
                        self._trigger_emergency_response(
                            f"Auto-modifica rilevata in {len(changes['self_modification_found'])} file"
                        )
                    
                    # Allerta per modifiche generali
                    if changes['modified'] or changes['added'] or changes['deleted']:
                        self._log_alert(
                            'file_changes',
                            f'Modifiche rilevate: {len(changes["modified"])} mod, {len(changes["added"])} add, {len(changes["deleted"])} del',
                            changes
                        )
                
                # Pausa tra scansioni
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoraggio fermato dall'utente")
        except Exception as e:
            print(f"\n❌ Errore nel monitoraggio: {e}")
            self.stats['consecutive_failures'] += 1
            
            if self.stats['consecutive_failures'] >= self.alert_thresholds['consecutive_failures']:
                self._trigger_emergency_response(f"Fallimenti consecutivi: {self.stats['consecutive_failures']}")
        finally:
            self.monitor_active = False
    
    def stop_monitoring(self):
        """Ferma il monitoraggio"""
        self.monitor_active = False
        print("🛑 Monitoraggio fermato")
    
    def get_status(self) -> Dict:
        """Ottiene stato del sistema di monitoraggio"""
        return {
            'monitor_active': self.monitor_active,
            'stats': self.stats,
            'critical_files_count': len(self.critical_files),
            'baseline_exists': os.path.exists(self.baseline_file),
            'last_scan': self.stats.get('last_scan')
        }


if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    monitor = SecuritySelfMonitor(project_root)
    
    if len(sys.argv) > 2:
        command = sys.argv[2]
        
        if command == "create_baseline":
            monitor.create_baseline()
        elif command == "check":
            result = monitor.check_integrity()
            print(json.dumps(result, indent=2))
        elif command == "start":
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            monitor.start_continuous_monitoring(interval)
        elif command == "status":
            status = monitor.get_status()
            print(json.dumps(status, indent=2))
        else:
            print("Comandi disponibili: create_baseline, check, start, status")
    else:
        print("🛡️ SISTEMA DI AUTO-MONITORAGGIO SICUREZZA")
        print("=" * 50)
        print("Uso: python security/self_monitor.py <project_root> <command>")
        print("Comandi:")
        print("  create_baseline - Crea baseline iniziale")
        print("  check          - Verifica integrità una volta")
        print("  start [sec]    - Avvia monitoraggio continuo")
        print("  status         - Mostra stato sistema")

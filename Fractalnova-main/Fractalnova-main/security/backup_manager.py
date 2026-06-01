import os
import json
import shutil
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import schedule

class SecureBackupManager:
    """
    Sistema di backup automatico con criptazione per auto-riparazione
    Mantiene copie criptate di tutti i file critici per ripristino automatico
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.backup_dir = os.path.join(project_root, "security", "secure_backups")
        self.metadata_file = os.path.join(self.backup_dir, "backup_metadata.json")
        self.encryption_key = os.getenv('SECURITY_ENC_KEY', self._generate_key())
        
        # File critici da proteggere
        self.critical_files = self._get_critical_files()
        
        # Configurazione backup
        self.config = {
            'backup_interval_hours': 1,  # Backup ogni ora
            'max_backups_per_file': 10,  # Massimo 10 backup per file
            'auto_cleanup_days': 30,    # Pulisci backup più vecchi di 30 giorni
            'encrypt_backups': True,     # Cripta sempre i backup
            'verify_integrity': True,   # Verifica integrità backup
            'auto_repair': True         # Auto-riparazione attiva
        }
        
        # Statistiche
        self.stats = {
            'backups_created': 0,
            'files_protected': 0,
            'repairs_performed': 0,
            'last_backup': None,
            'last_repair': None
        }
        
        # Crea directory backup
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Carica metadati esistenti
        self.metadata = self._load_metadata()
    
    def _generate_key(self) -> str:
        """Genera chiave di criptazione se non esiste"""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        return key.decode()
    
    def _get_critical_files(self) -> List[str]:
        """Ottiene lista di file critici da proteggere"""
        critical = []
        
        # File di applicazione
        app_files = [
            'app.py',
            'inference/generate.py',
            'inference/orchestrator.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file in app_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        # File di sicurezza (auto-protezione)
        security_files = [
            'security/advanced_verifier.py',
            'security/self_monitor.py',
            'security/ultra_guard.py',
            'security/encryptor.py',
            'security/guard.py',
            'security/verifier.py',
            'security/watchdog.py'
        ]
        
        for file in security_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        # File di configurazione
        config_files = [
            '.env', '.gitignore', 'pyproject.toml'
        ]
        
        for file in config_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        return critical
    
    def _load_metadata(self) -> Dict:
        """Carica metadati dei backup"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'files': {},
            'backups': [],
            'created_at': datetime.now().isoformat(),
            'last_cleanup': None
        }
    
    def _save_metadata(self):
        """Salva metadati dei backup"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
    
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
    
    def _encrypt_file(self, filepath: str) -> str:
        """Cripta un file e restituisce il percorso del file criptato"""
        try:
            from cryptography.fernet import Fernet
            
            # Leggi file originale
            with open(filepath, 'rb') as f:
                original_data = f.read()
            
            # Cripta
            fernet = Fernet(self.encryption_key.encode())
            encrypted_data = fernet.encrypt(original_data)
            
            # Salva file criptato
            encrypted_path = filepath + '.enc'
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            return encrypted_path
            
        except Exception as e:
            print(f"❌ Errore criptazione {filepath}: {e}")
            return None
    
    def _decrypt_file(self, encrypted_path: str, output_path: str) -> bool:
        """Decripta un file"""
        try:
            from cryptography.fernet import Fernet
            
            # Leggi file criptato
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decripta
            fernet = Fernet(self.encryption_key.encode())
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Salva file decriptato
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            return True
            
        except Exception as e:
            print(f"❌ Errore decriptazione {encrypted_path}: {e}")
            return False
    
    def create_secure_backup(self, filepath: str) -> Optional[str]:
        """Crea backup criptato di un file"""
        try:
            if not os.path.exists(filepath):
                return None
            
            # Calcola hash del file originale
            file_hash = self._calculate_file_hash(filepath)
            if not file_hash:
                return None
            
            # Crea directory per il file
            relative_path = os.path.relpath(filepath, self.project_root)
            file_backup_dir = os.path.join(self.backup_dir, relative_path.replace(os.sep, '_'))
            os.makedirs(file_backup_dir, exist_ok=True)
            
            # Nome backup con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}_{file_hash[:8]}.enc"
            backup_path = os.path.join(file_backup_dir, backup_name)
            
            # Cripta e salva backup
            encrypted_path = self._encrypt_file(filepath)
            if encrypted_path and os.path.exists(encrypted_path):
                shutil.move(encrypted_path, backup_path)
                
                # Aggiorna metadati
                if relative_path not in self.metadata['files']:
                    self.metadata['files'][relative_path] = {
                        'original_path': filepath,
                        'backups': [],
                        'last_backup': None,
                        'file_hash': file_hash
                    }
                
                backup_info = {
                    'backup_path': backup_path,
                    'timestamp': timestamp,
                    'file_hash': file_hash,
                    'size': os.path.getsize(backup_path),
                    'created_at': datetime.now().isoformat()
                }
                
                self.metadata['files'][relative_path]['backups'].append(backup_info)
                self.metadata['files'][relative_path]['last_backup'] = datetime.now().isoformat()
                self.metadata['files'][relative_path]['file_hash'] = file_hash
                
                # Limita numero di backup per file
                if len(self.metadata['files'][relative_path]['backups']) > self.config['max_backups_per_file']:
                    oldest_backup = self.metadata['files'][relative_path]['backups'].pop(0)
                    if os.path.exists(oldest_backup['backup_path']):
                        os.remove(oldest_backup['backup_path'])
                
                self.metadata['backups'].append(backup_info)
                self._save_metadata()
                
                self.stats['backups_created'] += 1
                print(f"✅ Backup creato: {relative_path}")
                return backup_path
            
        except Exception as e:
            print(f"❌ Errore backup {filepath}: {e}")
        
        return None
    
    def create_full_backup(self) -> Dict:
        """Crea backup completo di tutti i file critici"""
        print("🔒 Creazione backup completo sicuro...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'files_backed_up': 0,
            'files_failed': 0,
            'backup_paths': [],
            'errors': []
        }
        
        for filepath in self.critical_files:
            if os.path.exists(filepath):
                backup_path = self.create_secure_backup(filepath)
                if backup_path:
                    results['files_backed_up'] += 1
                    results['backup_paths'].append(backup_path)
                else:
                    results['files_failed'] += 1
                    results['errors'].append(f"Failed to backup: {filepath}")
            else:
                results['files_failed'] += 1
                results['errors'].append(f"File not found: {filepath}")
        
        self.stats['files_protected'] = results['files_backed_up']
        self.stats['last_backup'] = datetime.now().isoformat()
        
        print(f"✅ Backup completato: {results['files_backed_up']} file protetti")
        return results
    
    def detect_file_infection(self, filepath: str) -> Tuple[bool, List[str]]:
        """Rileva se un file è infetto"""
        infections = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Pattern di infezione
            infection_patterns = [
                r'eval\s*\(', r'exec\s*\(', r'__import__\s*\(',
                r'os\.system\s*\(', r'subprocess\.[a-zA-Z]+\s*\(',
                r'pickle\.loads?\s*\(', r'marshal\.loads?\s*\(',
                r'base64\.b64decode\s*\(', r'zlib\.decompress\s*\(',
                r'open\s*\(__file__', r'exec\s*\(open\s*\(__file__',
                r'#.*bypass', r'#.*hack', r'#.*backdoor', r'#.*trojan',
                r'#.*malware', r'#.*virus', r'#.*exploit'
            ]
            
            import re
            for pattern in infection_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    infections.append(f"Pattern '{pattern}' trovato alla riga {line_num}")
            
            # Controlla hash contro backup puliti
            current_hash = self._calculate_file_hash(filepath)
            relative_path = os.path.relpath(filepath, self.project_root)
            
            if relative_path in self.metadata['files']:
                clean_hash = self.metadata['files'][relative_path].get('file_hash')
                if clean_hash and current_hash != clean_hash:
                    infections.append(f"Hash modificato: {current_hash[:8]} != {clean_hash[:8]}")
            
            return len(infections) > 0, infections
            
        except Exception as e:
            infections.append(f"Errore scansione: {e}")
            return True, infections
    
    def repair_infected_file(self, filepath: str) -> bool:
        """Ripara un file infetto ripristinando l'ultimo backup pulito"""
        try:
            relative_path = os.path.relpath(filepath, self.project_root)
            
            if relative_path not in self.metadata['files']:
                print(f"❌ Nessun backup trovato per {relative_path}")
                return False
            
            file_metadata = self.metadata['files'][relative_path]
            backups = file_metadata.get('backups', [])
            
            if not backups:
                print(f"❌ Nessun backup disponibile per {relative_path}")
                return False
            
            # Usa l'ultimo backup
            latest_backup = backups[-1]
            backup_path = latest_backup['backup_path']
            
            if not os.path.exists(backup_path):
                print(f"❌ Backup non trovato: {backup_path}")
                return False
            
            # Crea backup del file infetto
            infected_backup = filepath + f".infected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(filepath, infected_backup)
            print(f"📁 File infetto salvato come: {infected_backup}")
            
            # Decripta e ripristina file pulito
            temp_restore = filepath + ".temp_restore"
            if self._decrypt_file(backup_path, temp_restore):
                # Sostituisci file infetto con versione pulita
                shutil.move(temp_restore, filepath)
                
                # Verifica integrità
                restored_hash = self._calculate_file_hash(filepath)
                if restored_hash == latest_backup['file_hash']:
                    print(f"✅ File riparato: {relative_path}")
                    self.stats['repairs_performed'] += 1
                    self.stats['last_repair'] = datetime.now().isoformat()
                    return True
                else:
                    print(f"❌ Verifica integrità fallita per {relative_path}")
                    # Ripristina file infetto
                    shutil.move(infected_backup, filepath)
                    return False
            else:
                print(f"❌ Errore decriptazione backup per {relative_path}")
                return False
                
        except Exception as e:
            print(f"❌ Errore riparazione {filepath}: {e}")
            return False
    
    def scan_and_repair_all(self) -> Dict:
        """Scansiona tutti i file critici e ripara quelli infetti"""
        print("🔍 Scansione e riparazione automatica...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'files_scanned': 0,
            'files_infected': 0,
            'files_repaired': 0,
            'repair_failures': 0,
            'infections_found': [],
            'repairs_performed': []
        }
        
        for filepath in self.critical_files:
            if os.path.exists(filepath):
                results['files_scanned'] += 1
                print(f"  🔍 Scansione: {os.path.relpath(filepath, self.project_root)}")
                
                is_infected, infections = self.detect_file_infection(filepath)
                
                if is_infected:
                    results['files_infected'] += 1
                    results['infections_found'].append({
                        'file': filepath,
                        'infections': infections
                    })
                    
                    print(f"    🚨 INFETTO: {len(infections)} pattern rilevati")
                    for infection in infections:
                        print(f"      - {infection}")
                    
                    # Tenta riparazione
                    if self.repair_infected_file(filepath):
                        results['files_repaired'] += 1
                        results['repairs_performed'].append(filepath)
                        print(f"    ✅ RIPARATO: {os.path.relpath(filepath, self.project_root)}")
                    else:
                        results['repair_failures'] += 1
                        print(f"    ❌ RIPARAZIONE FALLITA: {os.path.relpath(filepath, self.project_root)}")
                else:
                    print(f"    ✅ PULITO: {os.path.relpath(filepath, self.project_root)}")
        
        print(f"\n📊 RISULTATI SCANSIONE:")
        print(f"   File scansionati: {results['files_scanned']}")
        print(f"   File infetti: {results['files_infected']}")
        print(f"   File riparati: {results['files_repaired']}")
        print(f"   Riparazioni fallite: {results['repair_failures']}")
        
        return results
    
    def cleanup_old_backups(self):
        """Pulisce backup vecchi"""
        if not self.config['auto_cleanup_days']:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.config['auto_cleanup_days'])
        cleaned_count = 0
        
        for relative_path, file_metadata in self.metadata['files'].items():
            backups = file_metadata.get('backups', [])
            backups_to_remove = []
            
            for backup in backups:
                backup_date = datetime.fromisoformat(backup['created_at'])
                if backup_date < cutoff_date:
                    backups_to_remove.append(backup)
            
            for backup in backups_to_remove:
                if os.path.exists(backup['backup_path']):
                    os.remove(backup['backup_path'])
                    cleaned_count += 1
                
                backups.remove(backup)
        
        self.metadata['last_cleanup'] = datetime.now().isoformat()
        self._save_metadata()
        
        if cleaned_count > 0:
            print(f"🧹 Pulizia completata: {cleaned_count} backup vecchi rimossi")
    
    def start_automatic_backup(self):
        """Avvia backup automatico"""
        print("🔄 Avvio backup automatico...")
        
        def backup_job():
            try:
                self.create_full_backup()
                self.cleanup_old_backups()
            except Exception as e:
                print(f"❌ Errore backup automatico: {e}")
        
        # Programma backup
        schedule.every(self.config['backup_interval_hours']).hours.do(backup_job)
        
        print(f"✅ Backup automatico attivo (ogni {self.config['backup_interval_hours']} ore)")
        
        # Esegui backup iniziale
        backup_job()
        
        # Loop principale
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Controlla ogni minuto
        except KeyboardInterrupt:
            print("\n🛑 Backup automatico fermato")
    
    def get_status(self) -> Dict:
        """Ottiene stato del sistema di backup"""
        return {
            'backup_dir': self.backup_dir,
            'critical_files_count': len(self.critical_files),
            'files_protected': len(self.metadata['files']),
            'total_backups': len(self.metadata['backups']),
            'stats': self.stats,
            'config': self.config
        }


if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    backup_manager = SecureBackupManager(project_root)
    
    if len(sys.argv) > 2:
        command = sys.argv[2]
        
        if command == "backup":
            backup_manager.create_full_backup()
        elif command == "scan":
            backup_manager.scan_and_repair_all()
        elif command == "start":
            backup_manager.start_automatic_backup()
        elif command == "status":
            status = backup_manager.get_status()
            print(json.dumps(status, indent=2))
        else:
            print("Comandi disponibili: backup, scan, start, status")
    else:
        print("🔒 SISTEMA DI BACKUP SICURO")
        print("=" * 40)
        print("Uso: python security/backup_manager.py <project_root> <command>")
        print("Comandi:")
        print("  backup - Crea backup completo")
        print("  scan   - Scansiona e ripara file infetti")
        print("  start  - Avvia backup automatico")
        print("  status - Mostra stato sistema")

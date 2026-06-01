import os
import json
import shutil
import hashlib
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

class AutoRepairSystem:
    """
    Sistema di auto-riparazione che rileva e corregge automaticamente
    file infetti ripristinando versioni pulite dai backup criptati
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.security_dir = os.path.join(project_root, "security")
        self.repair_log = os.path.join(self.security_dir, "auto_repair_log.json")
        self.quarantine_dir = os.path.join(self.security_dir, "quarantine")
        
        # Crea directory necessarie
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        # Carica sistema di backup
        from security.backup_manager import SecureBackupManager
        self.backup_manager = SecureBackupManager(project_root)
        
        # Carica sistema di verifica
        from security.advanced_verifier import AdvancedSecurityVerifier
        self.verifier = AdvancedSecurityVerifier(project_root)
        
        # Configurazione auto-riparazione
        self.config = {
            'auto_repair_enabled': True,
            'quarantine_infected': True,
            'restore_from_backup': True,
            'verify_after_repair': True,
            'max_repair_attempts': 3,
            'repair_interval_seconds': 300,  # 5 minuti
            'emergency_mode': False
        }
        
        # Statistiche
        self.stats = {
            'repairs_attempted': 0,
            'repairs_successful': 0,
            'repairs_failed': 0,
            'files_quarantined': 0,
            'files_restored': 0,
            'last_repair': None,
            'emergency_repairs': 0
        }
        
        # Carica log riparazioni
        self.repair_log_data = self._load_repair_log()
    
    def _load_repair_log(self) -> List[Dict]:
        """Carica log delle riparazioni"""
        if os.path.exists(self.repair_log):
            try:
                with open(self.repair_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def _save_repair_log(self):
        """Salva log delle riparazioni"""
        with open(self.repair_log, 'w', encoding='utf-8') as f:
            json.dump(self.repair_log_data, f, indent=2)
    
    def _quarantine_file(self, filepath: str, reason: str) -> str:
        """Sposta file infetto in quarantena"""
        try:
            # Crea nome file quarantena
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(filepath)
            quarantine_name = f"{timestamp}_{filename}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_name)
            
            # Sposta file in quarantena
            shutil.move(filepath, quarantine_path)
            
            # Registra in quarantena
            quarantine_record = {
                'original_path': filepath,
                'quarantine_path': quarantine_path,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'file_hash': self._calculate_file_hash(quarantine_path)
            }
            
            self.stats['files_quarantined'] += 1
            print(f"🔒 File in quarantena: {os.path.relpath(filepath, self.project_root)}")
            return quarantine_path
            
        except Exception as e:
            print(f"❌ Errore quarantena {filepath}: {e}")
            return None
    
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
    
    def _restore_from_backup(self, filepath: str) -> bool:
        """Ripristina file da backup pulito"""
        try:
            relative_path = os.path.relpath(filepath, self.project_root)
            
            # Verifica se esiste backup
            if relative_path not in self.backup_manager.metadata['files']:
                print(f"❌ Nessun backup disponibile per {relative_path}")
                return False
            
            file_metadata = self.backup_manager.metadata['files'][relative_path]
            backups = file_metadata.get('backups', [])
            
            if not backups:
                print(f"❌ Nessun backup trovato per {relative_path}")
                return False
            
            # Usa l'ultimo backup pulito
            latest_backup = backups[-1]
            backup_path = latest_backup['backup_path']
            
            if not os.path.exists(backup_path):
                print(f"❌ Backup non trovato: {backup_path}")
                return False
            
            # Decripta e ripristina
            temp_restore = filepath + ".temp_restore"
            if self.backup_manager._decrypt_file(backup_path, temp_restore):
                # Sostituisci file infetto
                shutil.move(temp_restore, filepath)
                
                # Verifica integrità
                restored_hash = self._calculate_file_hash(filepath)
                if restored_hash == latest_backup['file_hash']:
                    print(f"✅ File ripristinato: {relative_path}")
                    self.stats['files_restored'] += 1
                    return True
                else:
                    print(f"❌ Verifica integrità fallita per {relative_path}")
                    return False
            else:
                print(f"❌ Errore decriptazione backup per {relative_path}")
                return False
                
        except Exception as e:
            print(f"❌ Errore ripristino {filepath}: {e}")
            return False
    
    def _verify_file_clean(self, filepath: str) -> bool:
        """Verifica che un file sia pulito dopo riparazione"""
        try:
            # Scansione con verifier avanzato
            file_results = self.verifier.scan_file_advanced(filepath)
            
            if file_results['threats']:
                print(f"⚠️ File ancora infetto dopo riparazione: {filepath}")
                for threat in file_results['threats']:
                    print(f"  - {threat['category']}: {threat['pattern']}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Errore verifica {filepath}: {e}")
            return False
    
    def repair_infected_file(self, filepath: str, infection_reason: str) -> bool:
        """Ripara un singolo file infetto"""
        print(f"🔧 Riparazione file: {os.path.relpath(filepath, self.project_root)}")
        print(f"   Motivo: {infection_reason}")
        
        self.stats['repairs_attempted'] += 1
        
        try:
            # 1. Quarantena file infetto
            if self.config['quarantine_infected']:
                quarantine_path = self._quarantine_file(filepath, infection_reason)
                if not quarantine_path:
                    print(f"❌ Impossibile mettere in quarantena: {filepath}")
                    return False
            
            # 2. Ripristina da backup pulito
            if self.config['restore_from_backup']:
                if not self._restore_from_backup(filepath):
                    print(f"❌ Impossibile ripristinare da backup: {filepath}")
                    return False
            
            # 3. Verifica file riparato
            if self.config['verify_after_repair']:
                if not self._verify_file_clean(filepath):
                    print(f"❌ File ancora infetto dopo riparazione: {filepath}")
                    return False
            
            # 4. Registra riparazione
            repair_record = {
                'file': filepath,
                'reason': infection_reason,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'quarantine_path': quarantine_path if self.config['quarantine_infected'] else None
            }
            
            self.repair_log_data.append(repair_record)
            self._save_repair_log()
            
            self.stats['repairs_successful'] += 1
            self.stats['last_repair'] = datetime.now().isoformat()
            
            print(f"✅ File riparato con successo: {os.path.relpath(filepath, self.project_root)}")
            return True
            
        except Exception as e:
            print(f"❌ Errore riparazione {filepath}: {e}")
            
            repair_record = {
                'file': filepath,
                'reason': infection_reason,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            
            self.repair_log_data.append(repair_record)
            self._save_repair_log()
            
            self.stats['repairs_failed'] += 1
            return False
    
    def scan_and_repair_all(self) -> Dict:
        """Scansiona e ripara tutti i file critici"""
        print("🔍 Scansione e riparazione automatica completa...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'files_scanned': 0,
            'files_infected': 0,
            'files_repaired': 0,
            'repair_failures': 0,
            'infections_found': [],
            'repairs_performed': []
        }
        
        # Ottieni tutti i file critici
        critical_files = self.backup_manager.critical_files
        
        for filepath in critical_files:
            if os.path.exists(filepath):
                results['files_scanned'] += 1
                print(f"\n🔍 Scansione: {os.path.relpath(filepath, self.project_root)}")
                
                # Rileva infezioni
                is_infected, infections = self.backup_manager.detect_file_infection(filepath)
                
                if is_infected:
                    results['files_infected'] += 1
                    infection_reason = f"{len(infections)} pattern rilevati: {', '.join(infections[:3])}"
                    results['infections_found'].append({
                        'file': filepath,
                        'infections': infections,
                        'reason': infection_reason
                    })
                    
                    print(f"  🚨 INFETTO: {infection_reason}")
                    
                    # Tenta riparazione
                    if self.repair_infected_file(filepath, infection_reason):
                        results['files_repaired'] += 1
                        results['repairs_performed'].append(filepath)
                        print(f"  ✅ RIPARATO: {os.path.relpath(filepath, self.project_root)}")
                    else:
                        results['repair_failures'] += 1
                        print(f"  ❌ RIPARAZIONE FALLITA: {os.path.relpath(filepath, self.project_root)}")
                else:
                    print(f"  ✅ PULITO: {os.path.relpath(filepath, self.project_root)}")
        
        print(f"\n📊 RISULTATI RIPARAZIONE:")
        print(f"   File scansionati: {results['files_scanned']}")
        print(f"   File infetti: {results['files_infected']}")
        print(f"   File riparati: {results['files_repaired']}")
        print(f"   Riparazioni fallite: {results['repair_failures']}")
        
        return results
    
    def emergency_repair(self) -> Dict:
        """Riparazione di emergenza per threat critici"""
        print("🚨 RIPARAZIONE DI EMERGENZA")
        print("=" * 40)
        
        self.config['emergency_mode'] = True
        self.stats['emergency_repairs'] += 1
        
        # 1. Crea backup di emergenza
        print("🔒 Creazione backup di emergenza...")
        backup_results = self.backup_manager.create_full_backup()
        
        # 2. Scansione e riparazione completa
        print("🔍 Scansione di emergenza...")
        repair_results = self.scan_and_repair_all()
        
        # 3. Verifica finale
        print("✅ Verifica finale sistema...")
        final_scan = self.backup_manager.scan_and_repair_all()
        
        emergency_results = {
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': True,
            'backup_results': backup_results,
            'repair_results': repair_results,
            'final_scan': final_scan,
            'system_clean': final_scan['files_infected'] == 0
        }
        
        if emergency_results['system_clean']:
            print("✅ SISTEMA RIPARATO: Tutti i file sono puliti")
        else:
            print("⚠️ ATTENZIONE: Alcuni file potrebbero essere ancora infetti")
        
        return emergency_results
    
    def start_continuous_repair(self):
        """Avvia riparazione continua"""
        print("🔄 Avvio riparazione continua...")
        print(f"   Intervallo: {self.config['repair_interval_seconds']} secondi")
        print("   Premi Ctrl+C per fermare")
        
        try:
            while True:
                # Esegui scansione e riparazione
                results = self.scan_and_repair_all()
                
                if results['files_infected'] > 0:
                    print(f"🚨 {results['files_infected']} file infetti rilevati e riparati")
                else:
                    print("✅ Sistema pulito")
                
                # Pausa tra scansioni
                time.sleep(self.config['repair_interval_seconds'])
                
        except KeyboardInterrupt:
            print("\n🛑 Riparazione continua fermata")
        except Exception as e:
            print(f"\n❌ Errore riparazione continua: {e}")
    
    def get_quarantine_status(self) -> Dict:
        """Ottiene stato della quarantena"""
        quarantine_files = []
        
        if os.path.exists(self.quarantine_dir):
            for filename in os.listdir(self.quarantine_dir):
                filepath = os.path.join(self.quarantine_dir, filename)
                if os.path.isfile(filepath):
                    quarantine_files.append({
                        'filename': filename,
                        'path': filepath,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    })
        
        return {
            'quarantine_dir': self.quarantine_dir,
            'files_in_quarantine': len(quarantine_files),
            'quarantine_files': quarantine_files,
            'total_size': sum(f['size'] for f in quarantine_files)
        }
    
    def get_status(self) -> Dict:
        """Ottiene stato del sistema di auto-riparazione"""
        return {
            'config': self.config,
            'stats': self.stats,
            'quarantine_status': self.get_quarantine_status(),
            'repair_log_entries': len(self.repair_log_data),
            'last_repair': self.stats.get('last_repair')
        }


if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    repair_system = AutoRepairSystem(project_root)
    
    if len(sys.argv) > 2:
        command = sys.argv[2]
        
        if command == "scan":
            repair_system.scan_and_repair_all()
        elif command == "emergency":
            repair_system.emergency_repair()
        elif command == "start":
            repair_system.start_continuous_repair()
        elif command == "status":
            status = repair_system.get_status()
            print(json.dumps(status, indent=2))
        elif command == "quarantine":
            quarantine = repair_system.get_quarantine_status()
            print(json.dumps(quarantine, indent=2))
        else:
            print("Comandi disponibili: scan, emergency, start, status, quarantine")
    else:
        print("🔧 SISTEMA DI AUTO-RIPARAZIONE")
        print("=" * 40)
        print("Uso: python security/auto_repair.py <project_root> <command>")
        print("Comandi:")
        print("  scan      - Scansiona e ripara file infetti")
        print("  emergency - Riparazione di emergenza")
        print("  start     - Avvia riparazione continua")
        print("  status    - Mostra stato sistema")
        print("  quarantine - Mostra file in quarantena")

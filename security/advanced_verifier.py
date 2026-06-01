import os
import re
import ast
import json
import hashlib
import subprocess
from typing import List, Dict, Set, Tuple
from datetime import datetime
import warnings

class AdvancedSecurityVerifier:
    """
    Sistema di sicurezza ultra-avanzato che controlla TUTTO il codice 24/7
    inclusi i file di sicurezza stessi per prevenire auto-modifiche malevoli
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.security_dir = os.path.join(project_root, "security")
        self.critical_files = self._get_critical_files()
        
        # Pattern di rilevamento ultra-aggressivi per codice malevolo
        self.malicious_patterns = {
            # Pattern di esecuzione pericolosa
            'execution': [
                r'eval\s*\(', r'exec\s*\(', r'compile\s*\(', r'__import__\s*\(',
                r'getattr\s*\(.*,\s*[\'"]__.*__[\'"]', r'setattr\s*\(.*,\s*[\'"]__.*__[\'"]',
                r'globals\s*\(', r'locals\s*\(', r'vars\s*\('
            ],
            
            # Pattern di accesso al sistema
            'system_access': [
                r'os\.system\s*\(', r'subprocess\.[a-zA-Z]+\s*\(', r'popen\s*\(',
                r'shell\s*=\s*True', r'check_output\s*\(', r'call\s*\(',
                r'run\s*\(', r'Popen\s*\(', r'communicate\s*\('
            ],
            
            # Pattern di manipolazione file pericolosa
            'file_manipulation': [
                r'open\s*\([^)]*[\'"]w[\'"]', r'open\s*\([^)]*[\'"]a[\'"]',
                r'remove\s*\(', r'unlink\s*\(', r'rmtree\s*\(', r'rmdir\s*\(',
                r'shutil\.rmtree\s*\(', r'os\.remove\s*\(', r'os\.unlink\s*\(',
                r'delete\s*\(', r'erase\s*\(', r'wipe\s*\('
            ],
            
            # Pattern di rete sospetti
            'network_suspicious': [
                r'requests\.(get|post|put|delete)\s*\([^)]*verify\s*=\s*False',
                r'urllib\.request\.urlopen\s*\(', r'urllib2\.urlopen\s*\(',
                r'socket\.socket\s*\(', r'httplib\s*\(', r'ftplib\s*\(',
                r'smtplib\s*\(', r'poplib\s*\(', r'imap\s*\('
            ],
            
            # Pattern di serializzazione pericolosa
            'serialization': [
                r'pickle\.loads?\s*\(', r'marshal\.loads?\s*\(', r'shelve\s*\(',
                r'cPickle\s*\(', r'dill\s*\(', r'cloudpickle\s*\('
            ],
            
            # Pattern di crittografia sospetta
            'crypto_suspicious': [
                r'base64\.b64decode\s*\(', r'binascii\.unhexlify\s*\(',
                r'zlib\.decompress\s*\(', r'gzip\.decompress\s*\(',
                r'hashlib\.new\s*\(', r'hmac\.new\s*\('
            ],
            
            # Pattern di modifica runtime
            'runtime_modification': [
                r'sys\.modules\s*\[', r'__builtins__\s*\[', r'__globals__\s*\[',
                r'frame\.f_globals\s*\[', r'frame\.f_locals\s*\[',
                r'inspect\.currentframe\s*\(', r'inspect\.getframeinfo\s*\('
            ],
            
            # Pattern di auto-modifica (pericolosissimo!)
            'self_modification': [
                r'open\s*\(__file__', r'open\s*\(os\.path\.abspath\s*\(__file__\)',
                r'write\s*\(.*__file__', r'exec\s*\(open\s*\(__file__',
                r'compile\s*\(open\s*\(__file__', r'__import__\s*\(.*__file__'
            ],
            
            # Pattern di sicurezza bypass
            'security_bypass': [
                r'#.*bypass', r'#.*hack', r'#.*backdoor', r'#.*trojan',
                r'#.*malware', r'#.*virus', r'#.*exploit',
                r'disable.*security', r'bypass.*check', r'skip.*validation'
            ],
            
            # Pattern di obfuscazione
            'obfuscation': [
                r'chr\s*\([0-9]+\)', r'ord\s*\([^)]+\)', r'lambda.*lambda',
                r'\[.*for.*in.*if.*\]', r'getattr\s*\(.*,\s*[\'"]\w+[\'"]\s*\)',
                r'hasattr\s*\(.*,\s*[\'"]\w+[\'"]\s*\)'
            ]
        }
        
        # Pattern specifici per file di sicurezza
        self.security_specific_patterns = [
            r'#.*disable.*security', r'#.*remove.*check', r'#.*skip.*scan',
            r'return.*False.*#.*bypass', r'if.*False.*#.*hack',
            r'encrypt.*=.*False', r'verify.*=.*False', r'scan.*=.*False'
        ]
    
    def _get_critical_files(self) -> List[str]:
        """Ottiene lista di file critici da monitorare"""
        critical = []
        
        # File di sicurezza (auto-monitoraggio)
        for root, dirs, files in os.walk(self.security_dir):
            for file in files:
                if file.endswith('.py'):
                    critical.append(os.path.join(root, file))
        
        # File di applicazione
        app_files = [
            'app.py', 'inference/generate.py', 'inference/orchestrator.py',
            '.github/workflows/security.yml', 'requirements.txt'
        ]
        
        for file in app_files:
            full_path = os.path.join(self.project_root, file)
            if os.path.exists(full_path):
                critical.append(full_path)
        
        return critical
    
    def scan_file_advanced(self, filepath: str) -> Dict:
        """Scansione avanzata di un singolo file"""
        findings = {
            'file': filepath,
            'timestamp': datetime.now().isoformat(),
            'threats': [],
            'severity': 'low',
            'recommendations': []
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Scansione pattern malevoli
            for category, patterns in self.malicious_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        threat = {
                            'category': category,
                            'pattern': pattern,
                            'line': content[:match.start()].count('\n') + 1,
                            'context': self._get_context(content, match.start(), match.end()),
                            'severity': self._get_severity(category, pattern)
                        }
                        findings['threats'].append(threat)
            
            # Scansione specifica per file di sicurezza
            if 'security' in filepath:
                for pattern in self.security_specific_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        threat = {
                            'category': 'security_self_modification',
                            'pattern': pattern,
                            'line': content[:match.start()].count('\n') + 1,
                            'context': self._get_context(content, match.start(), match.end()),
                            'severity': 'critical'
                        }
                        findings['threats'].append(threat)
            
            # Analisi AST per pattern complessi
            try:
                tree = ast.parse(content)
                ast_findings = self._analyze_ast(tree)
                findings['threats'].extend(ast_findings)
            except SyntaxError:
                findings['threats'].append({
                    'category': 'syntax_error',
                    'pattern': 'Invalid Python syntax',
                    'severity': 'high'
                })
            
            # Calcolo severità complessiva
            if findings['threats']:
                severities = [t['severity'] for t in findings['threats']]
                if 'critical' in severities:
                    findings['severity'] = 'critical'
                elif 'high' in severities:
                    findings['severity'] = 'high'
                elif 'medium' in severities:
                    findings['severity'] = 'medium'
                else:
                    findings['severity'] = 'low'
            
            # Generazione raccomandazioni
            findings['recommendations'] = self._generate_recommendations(findings['threats'])
            
        except Exception as e:
            findings['threats'].append({
                'category': 'scan_error',
                'pattern': f'Error scanning file: {str(e)}',
                'severity': 'medium'
            })
        
        return findings
    
    def _get_context(self, content: str, start: int, end: int, context_lines: int = 3) -> str:
        """Ottiene contesto intorno al pattern trovato"""
        lines = content.split('\n')
        start_line = content[:start].count('\n')
        
        context_start = max(0, start_line - context_lines)
        context_end = min(len(lines), start_line + context_lines + 1)
        
        context_lines_list = []
        for i in range(context_start, context_end):
            prefix = ">>> " if i == start_line else "    "
            context_lines_list.append(f"{prefix}{i+1:4d}: {lines[i]}")
        
        return '\n'.join(context_lines_list)
    
    def _get_severity(self, category: str, pattern: str) -> str:
        """Determina severità del threat"""
        critical_categories = ['self_modification', 'security_bypass', 'execution']
        high_categories = ['system_access', 'file_manipulation', 'runtime_modification']
        
        if category in critical_categories:
            return 'critical'
        elif category in high_categories:
            return 'high'
        elif category in ['network_suspicious', 'serialization']:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_ast(self, tree: ast.AST) -> List[Dict]:
        """Analizza AST per pattern complessi"""
        findings = []
        
        class ThreatVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Rileva chiamate pericolose
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                        findings.append({
                            'category': 'ast_execution',
                            'pattern': f'AST: {node.func.id}() call',
                            'line': node.lineno,
                            'severity': 'critical'
                        })
                
                # Rileva accessi a attributi pericolosi
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'call', 'run']:
                        findings.append({
                            'category': 'ast_system',
                            'pattern': f'AST: {node.func.attr}() call',
                            'line': node.lineno,
                            'severity': 'high'
                        })
                
                self.generic_visit(node)
        
        visitor = ThreatVisitor()
        visitor.visit(tree)
        return findings
    
    def _generate_recommendations(self, threats: List[Dict]) -> List[str]:
        """Genera raccomandazioni basate sui threat trovati"""
        recommendations = []
        
        categories = set(t['category'] for t in threats)
        
        if 'execution' in categories:
            recommendations.append("Rimuovi o sostituisci eval/exec con alternative sicure")
        
        if 'system_access' in categories:
            recommendations.append("Verifica e sanitizza tutti gli input per chiamate di sistema")
        
        if 'self_modification' in categories:
            recommendations.append("CRITICO: File di sicurezza modificato - verifica integrità")
        
        if 'security_bypass' in categories:
            recommendations.append("Rimuovi commenti o codice che bypassa controlli di sicurezza")
        
        return recommendations
    
    def scan_project_comprehensive(self) -> Dict:
        """Scansione completa del progetto"""
        print("🔍 Avvio scansione di sicurezza ultra-avanzata...")
        
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'project_root': self.project_root,
            'files_scanned': 0,
            'threats_found': 0,
            'critical_threats': 0,
            'files_with_threats': [],
            'overall_severity': 'low',
            'recommendations': []
        }
        
        # Scansione di tutti i file Python
        for root, dirs, files in os.walk(self.project_root):
            # Skip directories non rilevanti
            if any(skip in root for skip in ['.git', '__pycache__', 'models', '.venv']):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    results['files_scanned'] += 1
                    
                    print(f"  📄 Scansione: {os.path.relpath(filepath, self.project_root)}")
                    
                    file_results = self.scan_file_advanced(filepath)
                    
                    if file_results['threats']:
                        results['files_with_threats'].append(file_results)
                        results['threats_found'] += len(file_results['threats'])
                        
                        if file_results['severity'] == 'critical':
                            results['critical_threats'] += 1
        
        # Calcolo severità complessiva
        if results['critical_threats'] > 0:
            results['overall_severity'] = 'critical'
        elif results['threats_found'] > 10:
            results['overall_severity'] = 'high'
        elif results['threats_found'] > 5:
            results['overall_severity'] = 'medium'
        
        # Raccomandazioni generali
        if results['overall_severity'] == 'critical':
            results['recommendations'].append("🚨 AZIONE IMMEDIATA RICHIESTA: Threat critici rilevati!")
            results['recommendations'].append("🔒 Cripta immediatamente i file critici")
            results['recommendations'].append("🛡️ Verifica integrità dei file di sicurezza")
        
        print(f"✅ Scansione completata: {results['threats_found']} threat trovati")
        return results
    
    def save_report(self, results: Dict, report_path: str = None) -> str:
        """Salva report di sicurezza"""
        if not report_path:
            report_path = os.path.join(self.security_dir, "advanced_security_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return report_path


if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    verifier = AdvancedSecurityVerifier(project_root)
    
    print("🛡️ SISTEMA DI SICUREZZA ULTRA-AVANZATO")
    print("=" * 50)
    
    results = verifier.scan_project_comprehensive()
    report_path = verifier.save_report(results)
    
    print(f"\n📊 RISULTATI SCANSIONE:")
    print(f"   File scansionati: {results['files_scanned']}")
    print(f"   Threat trovati: {results['threats_found']}")
    print(f"   Threat critici: {results['critical_threats']}")
    print(f"   Severità: {results['overall_severity'].upper()}")
    print(f"   Report salvato: {report_path}")
    
    if results['overall_severity'] == 'critical':
        print("\n🚨 ATTENZIONE: Threat critici rilevati!")
        sys.exit(1)
    else:
        print("\n✅ Sistema sicuro")
        sys.exit(0)

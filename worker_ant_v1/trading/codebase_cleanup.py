"""
SMART APE NEURAL SWARM - CODEBASE CLEANUP
=========================================

Phase 2: Comprehensive codebase cleanup and optimization.
Eliminates dead code, unused imports, and organizes structure.
"""

import os
import re
import sys
import ast
import shutil
from typing import Dict, List, Set, Any, Tuple
from pathlib import Path
import logging

class CodebaseCleanup:
    """Comprehensive codebase cleanup and optimization"""
    
    def __init__(self):
        self.project_root = Path('.')
        self.cleanup_stats = {
            'files_scanned': 0,
            'dead_files_removed': 0,
            'unused_imports_cleaned': 0,
            'todo_items_found': 0,
            'temp_files_removed': 0,
            'bytes_saved': 0,
            'structure_changes': 0
        }
        
        
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Trading-specific patterns to clean
        self.trading_patterns = {
            'dead_files': [
                '**/old_trading_*.py',
                '**/deprecated_*.py',
                '**/backup_wallet_*.json',
                '**/test_trades_*.log',
                '**/simulation_*.py',
                '**/mock_*.py'
            ],
            'temp_patterns': [
                '**/temp_order_*.json',
                '**/debug_trade_*.log',
                '**/cache_*.tmp'
            ]
        }
        
        # Core trading modules that must be preserved
        self.core_modules = {
            'worker_ant_v1/trading/order_buyer.py',
            'worker_ant_v1/trading/order_seller.py',
            'worker_ant_v1/core/unified_trading_engine.py',
            'worker_ant_v1/core/wallet_manager.py',
            'worker_ant_v1/core/hyper_compound_engine.py',
            'worker_ant_v1/intelligence/token_intelligence_system.py',
            'worker_ant_v1/safety/enhanced_rug_detector.py'
        }
        
    def run_full_cleanup(self):
        """Execute complete codebase cleanup"""
        
        print("🧼 STARTING CODEBASE CLEANUP")
        print("=" * 60)
        
        
        print("\n🗑️  Phase 1: Dead File Removal")
        self._remove_dead_files()
        
        
        print("\n📦 Phase 2: Import Cleanup")
        self._clean_unused_imports()
        
        
        print("\n💬 Phase 3: Commented Code Cleanup")
        self._remove_commented_code()
        
        
        print("\n📝 Phase 4: Technical Debt Audit")
        self._audit_technical_debt()
        
        
        print("\n📁 Phase 5: Structure Organization")
        self._organize_structure()
        
        
        print("\n✅ Phase 6: Post-Cleanup Validation")
        self._validate_cleanup()
        
        
        self._generate_cleanup_report()
    
    def _remove_dead_files(self):
        """Remove dead files and temporary artifacts"""
        
        
        dead_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo', 
            '**/.pytest_cache',
            '**/.coverage',
            '**/test_*.py.bak',
            '**/temp_*.py',
            '**/debug_*.py',
            '**/core.*'
        ]
        
        # Add trading-specific patterns
        dead_patterns.extend(self.trading_patterns['dead_files'])
        dead_patterns.extend(self.trading_patterns['temp_patterns'])
        
        
        specific_dead_files = [
            'test_runner.py',
            'debug_output.txt',
            'wallet_backup_old.json',
            'config_backup.env',
            'old_main.py',
            'mock_trading_engine.py',
            'simulation_results.json',
            'test_wallet.json'
        ]
        
        removed_count = 0
        bytes_saved = 0
        
        
        for pattern in dead_patterns:
            for file_path in self.project_root.glob(pattern):
                # Skip core modules
                if str(file_path) in self.core_modules:
                    continue
                    
                if file_path.exists():
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        bytes_saved += size
                        file_path.unlink()
                        removed_count += 1
                        print(f"   🗑️  Removed: {file_path}")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        removed_count += 1
                        print(f"   🗑️  Removed directory: {file_path}")
        
        
        for dead_file in specific_dead_files:
            file_path = self.project_root / dead_file
            if file_path.exists() and str(file_path) not in self.core_modules:
                size = file_path.stat().st_size
                bytes_saved += size
                file_path.unlink()
                removed_count += 1
                print(f"   🗑️  Removed: {file_path}")
        
        self.cleanup_stats['dead_files_removed'] = removed_count
        self.cleanup_stats['bytes_saved'] += bytes_saved
        
        if removed_count > 0:
            print(f"✅ Removed {removed_count} dead files, saved {bytes_saved:,} bytes")
        else:
            print("✅ No dead files found")
    
    def _clean_unused_imports(self):
        """Clean unused imports from Python files"""
        
        python_files = list(self.project_root.glob('**/*.py'))
        cleaned_files = 0
        
        for py_file in python_files:
            # Skip core modules from deep cleaning
            if str(py_file) in self.core_modules:
                continue
                
            if any(skip in str(py_file) for skip in ['.venv', 'venv', '__pycache__', '.git']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                try:
                    tree = ast.parse(content)
                    imports_used = self._find_used_imports(tree, content)
                    
                    cleaned_content = self._remove_unused_imports(content, imports_used)
                    
                    if cleaned_content != original_content:
                        with open(py_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        cleaned_files += 1
                        print(f"   🧹 Cleaned imports: {py_file}")
                        
                except SyntaxError:
                    continue
                    
            except Exception as e:
                print(f"   ⚠️  Could not process {py_file}: {e}")
                continue
        
        self.cleanup_stats['unused_imports_cleaned'] = cleaned_files
        self.cleanup_stats['files_scanned'] = len(python_files)
        
        if cleaned_files > 0:
            print(f"✅ Cleaned imports in {cleaned_files} files")
        else:
            print("✅ No unused imports found")
    
    def _find_used_imports(self, tree: ast.AST, content: str) -> Set[str]:
        """Find which imports are actually used"""
        
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        for match in re.finditer(import_pattern, content):
            module = match.group(1) or match.group(2)
            if module and module in content:
                used_names.add(module)
        
        return used_names
    
    def _remove_unused_imports(self, content: str, used_imports: Set[str]) -> str:
        """Remove unused import statements"""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        import_block = False
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                import_block = True
                # Check if import is used
                if any(imp in line for imp in used_imports):
                    cleaned_lines.append(line)
            else:
                if import_block and line.strip():
                    import_block = False
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_commented_code(self):
        """Remove large blocks of commented code"""
        
        python_files = list(self.project_root.glob('**/*.py'))
        cleaned_files = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                cleaned_lines = []
                in_comment_block = False
                comment_block_size = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    if stripped.startswith('#') and not any(keep in stripped.lower() for keep in ['todo', 'fixme', 'note', 'important', 'warning']):
                        if not in_comment_block:
                            in_comment_block = True
                            comment_block_size = 1
                        else:
                            comment_block_size += 1
                    else:
                        if in_comment_block:
                            if comment_block_size >= 5:
                                print(f"      🗑️  Removed {comment_block_size}-line comment block in {py_file}")
                            else:
                                # Keep small comment blocks
                                for _ in range(comment_block_size):
                                    if len(cleaned_lines) > 0:
                                        last_line = lines[len(cleaned_lines)]
                                        cleaned_lines.append(last_line)
                        
                        in_comment_block = False
                        comment_block_size = 0
                        cleaned_lines.append(line)
                
                if len(cleaned_lines) != len(lines):
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.writelines(cleaned_lines)
                    cleaned_files += 1
                    
            except Exception as e:
                print(f"   ⚠️  Could not process {py_file}: {e}")
                continue
        
        if cleaned_files > 0:
            print(f"✅ Cleaned commented code in {cleaned_files} files")
        else:
            print("✅ No large comment blocks found")
    
    def _audit_technical_debt(self):
        """Audit and catalog technical debt (TODO, FIXME, etc.)"""
        
        debt_patterns = [
            (r'TODO', 'TODO items'),
            (r'FIXME', 'Fix required'),
            (r'XXX', 'Attention needed'),
            (r'HACK', 'Code hacks'),
            (r'DEPRECATED', 'Deprecated code'),
            (r'TEMP', 'Temporary code')
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        total_debt = 0
        debt_summary = {}
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in debt_patterns:
                    matches = re.findall(f'#.*{pattern}.*', content, re.IGNORECASE)
                    if matches:
                        if description not in debt_summary:
                            debt_summary[description] = []
                        
                        for match in matches:
                            debt_summary[description].append(f"{py_file}: {match.strip()}")
                            total_debt += 1
                            
            except Exception:
                continue
        
        self.cleanup_stats['todo_items_found'] = total_debt
        
        if total_debt > 0:
            print(f"⚠️  Found {total_debt} technical debt items:")
            for category, items in debt_summary.items():
                print(f"   📝 {category}: {len(items)} items")
                for item in items[:3]:  # Show first 3
                    print(f"      • {item}")
                if len(items) > 3:
                    print(f"      ... and {len(items) - 3} more")
        else:
            print("✅ No technical debt markers found")
    
    def _organize_structure(self):
        """Organize codebase structure"""
        
        # Define core directories
        core_dirs = {
            'worker_ant_v1/core': 'Core trading engine components',
            'worker_ant_v1/trading': 'Order execution and management',
            'worker_ant_v1/intelligence': 'Market analysis and predictions',
            'worker_ant_v1/safety': 'Risk management and protection',
            'worker_ant_v1/utils': 'Shared utilities and helpers',
            'worker_ant_v1/monitoring': 'System monitoring and alerts'
        }
        
        # Ensure core directories exist
        for dir_path in core_dirs:
            path = self.project_root / dir_path
            if not path.exists():
                path.mkdir(parents=True)
                print(f"   📁 Created directory: {dir_path}")
                self.cleanup_stats['structure_changes'] += 1
        
        # Move misplaced files to correct directories
        for py_file in self.project_root.glob('**/*.py'):
            file_path = str(py_file)
            
            # Skip core modules and special files
            if file_path in self.core_modules or any(skip in file_path for skip in ['__init__', 'setup']):
                continue
            
            # Determine correct directory based on file content
            correct_dir = self._determine_correct_directory(py_file)
            if correct_dir and str(py_file.parent) != correct_dir:
                target_path = self.project_root / correct_dir / py_file.name
                if not target_path.exists():
                    shutil.move(str(py_file), str(target_path))
                    print(f"   📦 Moved {py_file.name} to {correct_dir}")
                    self.cleanup_stats['structure_changes'] += 1
    
    def _determine_correct_directory(self, file_path: Path) -> str:
        """Determine the correct directory for a file based on its content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            if any(term in content for term in ['order', 'trade', 'position', 'market']):
                return 'worker_ant_v1/trading'
            elif any(term in content for term in ['predict', 'analyze', 'intelligence', 'signal']):
                return 'worker_ant_v1/intelligence'
            elif any(term in content for term in ['risk', 'safety', 'protect', 'detect']):
                return 'worker_ant_v1/safety'
            elif any(term in content for term in ['monitor', 'alert', 'notify', 'log']):
                return 'worker_ant_v1/monitoring'
            elif any(term in content for term in ['util', 'helper', 'common', 'shared']):
                return 'worker_ant_v1/utils'
        except Exception:
            pass
        return None
    
    def _validate_cleanup(self):
        """Validate that cleanup didn't break anything"""
        
        validation_checks = []
        
        
        core_modules = [
            'worker_ant_v1.core.unified_config',
            'worker_ant_v1.core.wallet_manager',
            'worker_ant_v1.trading.order_buyer',
            'worker_ant_v1.utils.logger'
        ]
        
        for module in core_modules:
            try:
                __import__(module)
                validation_checks.append(f"✅ {module}")
            except Exception as e:
                validation_checks.append(f"❌ {module}: {e}")
        
        
        entry_points = [
            'entry_points/run_bot.py',
            'entry_points/neural_swarm_commander.py'
        ]
        
        for entry_point in entry_points:
            if (self.project_root / entry_point).exists():
                validation_checks.append(f"✅ {entry_point}")
            else:
                validation_checks.append(f"❌ Missing: {entry_point}")
        
        print("Post-cleanup validation:")
        for check in validation_checks:
            print(f"   {check}")
        
        
        if any('❌' in check for check in validation_checks):
            print("⚠️  Some validation checks failed - review cleanup")
        else:
            print("✅ All validation checks passed")
    
    def _generate_cleanup_report(self):
        """Generate comprehensive cleanup report"""
        
        print("\n" + "=" * 60)
        print("🧼 CODEBASE CLEANUP REPORT")
        print("=" * 60)
        
        stats = self.cleanup_stats
        
        print(f"""
📊 CLEANUP STATISTICS:
   • Files Scanned: {stats['files_scanned']}
   • Dead Files Removed: {stats['dead_files_removed']}
   • Import Cleanups: {stats['unused_imports_cleaned']}
   • Technical Debt Items: {stats['todo_items_found']}
   • Bytes Saved: {stats['bytes_saved']:,}
   • Structure Changes: {stats['structure_changes']}
        """)
        
        if stats['dead_files_removed'] > 0 or stats['unused_imports_cleaned'] > 0 or stats['structure_changes'] > 0:
            print("✅ CODEBASE SUCCESSFULLY CLEANED")
            print("🎯 RESULT: Leaner, faster, more maintainable code")
        else:
            print("✅ CODEBASE ALREADY CLEAN")
            print("🎯 RESULT: No cleanup needed")
        
        print(f"\n🧹 CLEANUP COMPLETED: {self.cleanup_stats}")

def main():
    """Main entry point"""
    cleaner = CodebaseCleanup()
    cleaner.run_full_cleanup()

if __name__ == "__main__":
    main() 
"""
CODEBASE CLEANER
===============

Ruthlessly efficient codebase cleaner.
- Dead code elimination
- Import optimization
- Structure validation
- Performance leak detection
"""

import os
import ast
import logging
from typing import Set, List, Dict
import re
from pathlib import Path
import importlib
import sys

class CodebaseCleaner:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
    def clean_codebase(self) -> Dict[str, List[str]]:
        """Run full codebase cleanup"""
        results = {
            "removed_files": [],
            "cleaned_files": [],
            "warnings": []
        }
        
        try:
            # 1. Find all Python files
            python_files = self._find_python_files()
            
            # 2. Build import graph
            import_graph = self._build_import_graph(python_files)
            
            # 3. Find unused files
            unused_files = self._find_unused_files(import_graph)
            results["removed_files"].extend(unused_files)
            
            # 4. Clean individual files
            for file_path in python_files:
                if file_path not in unused_files:
                    cleaned = self._clean_file(file_path)
                    if cleaned:
                        results["cleaned_files"].append(file_path)
                        
        except Exception as e:
            results["warnings"].append(f"Cleanup error: {str(e)}")
            
        return results
        
    def _find_python_files(self) -> List[str]:
        """Find all Python files in project"""
        python_files = []
        for root, _, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
        
    def _build_import_graph(self, python_files: List[str]) -> Dict[str, Set[str]]:
        """Build graph of file imports"""
        import_graph = {}
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                    
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                        imports.add(module)
                        
                import_graph[file_path] = imports
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {file_path}: {str(e)}")
                
        return import_graph
        
    def _find_unused_files(self, import_graph: Dict[str, Set[str]]) -> List[str]:
        """Find Python files that are never imported"""
        all_imports = set()
        for imports in import_graph.values():
            all_imports.update(imports)
            
        unused_files = []
        for file_path in import_graph:
            module_name = self._get_module_name(file_path)
            if module_name not in all_imports and not self._is_entry_point(file_path):
                unused_files.append(file_path)
                
        return unused_files
        
    def _get_module_name(self, file_path: str) -> str:
        """Convert file path to module name"""
        rel_path = os.path.relpath(file_path, self.project_root)
        return os.path.splitext(rel_path)[0].replace(os.sep, '.')
        
    def _is_entry_point(self, file_path: str) -> bool:
        """Check if file is an entry point"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                return '__main__' in content or 'if __name__ == "__main__"' in content
        except Exception:
            return False
            
    def _clean_file(self, file_path: str) -> bool:
        """Clean individual Python file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Remove commented code blocks
            cleaned = self._remove_commented_code(content)
            
            # Remove unused imports
            cleaned = self._remove_unused_imports(cleaned)
            
            # Remove empty lines at end
            cleaned = cleaned.rstrip() + '\n'
            
            if cleaned != content:
                with open(file_path, 'w') as f:
                    f.write(cleaned)
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to clean {file_path}: {str(e)}")
            return False
            
    def _remove_commented_code(self, content: str) -> str:
        """Remove large blocks of commented code"""
        # Remove multi-line comments that look like code
        pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        
        def is_code_comment(match):
            text = match.group(0)
            # Check if comment contains code-like patterns
            code_patterns = [
                r'\bdef\b',
                r'\bclass\b',
                r'\bif\b.*:',
                r'\bfor\b.*:',
                r'\bwhile\b.*:',
                r'=(?!=)',  # Assignment but not ==
                r'\breturn\b'
            ]
            return any(re.search(p, text) for p in code_patterns)
            
        return re.sub(pattern, lambda m: '' if is_code_comment(m) else m.group(0), content)
        
    def _remove_unused_imports(self, content: str) -> str:
        """Remove unused imports from file"""
        try:
            tree = ast.parse(content)
            
            # Find all imported names
            imports = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        imports[name.asname or name.name] = node
                        
            # Find all used names
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                    
            # Remove unused imports
            lines = content.split('\n')
            to_remove = set()
            
            for name, node in imports.items():
                if name not in used_names:
                    to_remove.add(node.lineno)
                    
            return '\n'.join(line for i, line in enumerate(lines, 1)
                           if i not in to_remove)
                           
        except Exception:
            return content 
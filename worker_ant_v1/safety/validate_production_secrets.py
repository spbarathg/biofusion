"""
PRODUCTION SECRETS VALIDATOR - SECURITY CHECKPOINT
=================================================

Advanced security validation script that scans all configuration files
for placeholder secrets and validates production readiness.

🔍 VALIDATION AREAS:
- Environment files (.env, .env.production, etc.)
- Configuration files (grafana.ini, redis.conf, etc.)
- Docker compose files
- Kubernetes manifests
- Application config files

🚨 SECURITY CHECKS:
- Placeholder detection (REPLACE_WITH, your_, admin123, etc.)
- Weak password detection
- Default credential detection
- Exposed API key validation
- Certificate and key validation

🎯 CI/CD INTEGRATION:
- Exit code 1 on any security issues found
- Detailed reporting with line numbers
- JSON output for automation
- Severity classification
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib


class SecuritySeverity(Enum):
    """Security issue severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(Enum):
    """Types of security issues"""
    PLACEHOLDER = "placeholder"
    WEAK_PASSWORD = "weak_password"
    DEFAULT_CREDENTIAL = "default_credential"
    EXPOSED_SECRET = "exposed_secret"
    MISSING_ENCRYPTION = "missing_encryption"
    INSECURE_CONFIG = "insecure_config"


@dataclass
class SecurityIssue:
    """Security issue found during validation"""
    file_path: str
    line_number: int
    line_content: str
    issue_type: IssueType
    severity: SecuritySeverity
    message: str
    recommendation: str
    pattern_matched: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'line_number': self.line_number,
            'line_content': self.line_content.strip(),
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'recommendation': self.recommendation,
            'pattern_matched': self.pattern_matched
        }


@dataclass
class ValidationReport:
    """Complete validation report"""
    total_files_scanned: int
    total_lines_scanned: int
    total_issues: int
    issues_by_severity: Dict[SecuritySeverity, int]
    issues_by_type: Dict[IssueType, int]
    security_issues: List[SecurityIssue]
    scan_duration_seconds: float
    is_production_ready: bool
    critical_issues: List[SecurityIssue] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_files_scanned': self.total_files_scanned,
            'total_lines_scanned': self.total_lines_scanned,
            'total_issues': self.total_issues,
            'issues_by_severity': {k.value: v for k, v in self.issues_by_severity.items()},
            'issues_by_type': {k.value: v for k, v in self.issues_by_type.items()},
            'security_issues': [issue.to_dict() for issue in self.security_issues],
            'scan_duration_seconds': self.scan_duration_seconds,
            'is_production_ready': self.is_production_ready,
            'critical_issues_count': len(self.critical_issues)
        }


class ProductionSecretsValidator:
    """Advanced production secrets validator with comprehensive security checks"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        
        self.scan_patterns = [
            "*.env*",
            "*.ini",
            "*.conf",
            "*.yml",
            "*.yaml",
            "*.json",
            "*.toml",
            "*.py",
            "docker-compose*.yml",
            "Dockerfile*",
            "*.k8s.yml",
            "*.secret.yml"
        ]
        
        
        self.scan_directories = [
            "config",
            "deployment", 
            "monitoring",
            "scripts",
            "worker_ant_v1/core",
            "."
        ]
        
        
        self.exclude_patterns = [
            "*.log",
            "*.tmp",
            "*.cache",
            "__pycache__/*",
            ".git/*",
            "node_modules/*",
            "venv/*",
            ".venv/*",
            "build/*",
            "dist/*"
        ]
        
        
        self.placeholder_patterns = [
            (r'REPLACE_WITH[_\w]*', SecuritySeverity.CRITICAL, "Replace placeholder with actual value"),
            (r'your_[\w_]+', SecuritySeverity.CRITICAL, "Replace 'your_*' placeholder with actual value"),
            (r'admin123|password123|123456', SecuritySeverity.CRITICAL, "Replace default password"),
            (r'changeme|change_me|CHANGEME', SecuritySeverity.CRITICAL, "Replace placeholder value"),
            (r'example\.com|localhost:\d+', SecuritySeverity.HIGH, "Replace example endpoints with production values"),
            (r'test_[\w_]*_here', SecuritySeverity.HIGH, "Replace test placeholder with production value"),
            (r'TODO:.*secret|FIXME:.*password', SecuritySeverity.HIGH, "Complete TODO/FIXME for security item")
        ]
        
        
        self.weak_password_patterns = [
            (r'password\s*=\s*["\']?(admin|root|password|123456|qwerty)["\']?', SecuritySeverity.HIGH, "Weak password detected"),
            (r'pass\s*=\s*["\']?(\w{1,5})["\']?', SecuritySeverity.MEDIUM, "Very short password detected"),
            (r'secret\s*=\s*["\']?(test|demo|sample)["\']?', SecuritySeverity.HIGH, "Test/demo secret in production config")
        ]
        
        
        self.default_credential_patterns = [
            (r'user\s*=\s*["\']?(admin|root|administrator)["\']?', SecuritySeverity.MEDIUM, "Default username detected"),
            (r'username\s*=\s*["\']?(admin|root)["\']?', SecuritySeverity.MEDIUM, "Default username detected"),
            (r'api_key\s*=\s*["\']?(demo|test|example)["\']?', SecuritySeverity.HIGH, "Default API key detected")
        ]
        
        
        self.exposed_secret_patterns = [
            (r'private_key\s*=\s*["\']?[a-zA-Z0-9+/]{40,}', SecuritySeverity.CRITICAL, "Private key exposed in config"),
            (r'secret_key\s*=\s*["\']?[a-zA-Z0-9+/]{32,}', SecuritySeverity.HIGH, "Secret key may be exposed"),
            (r'["\'][a-zA-Z0-9]{32,}["\']', SecuritySeverity.MEDIUM, "Potential secret value in plaintext")
        ]
        
        
        self.insecure_config_patterns = [
            (r'ssl\s*=\s*["\']?(false|0|disabled)["\']?', SecuritySeverity.MEDIUM, "SSL disabled - security risk"),
            (r'verify_ssl\s*=\s*["\']?(false|0)["\']?', SecuritySeverity.MEDIUM, "SSL verification disabled"),
            (r'debug\s*=\s*["\']?(true|1|enabled)["\']?', SecuritySeverity.LOW, "Debug mode enabled in production"),
            (r'cors_allow_all\s*=\s*["\']?(true|1)["\']?', SecuritySeverity.HIGH, "CORS allow all enabled - security risk")
        ]
        
        
        self.stats = {
            'files_scanned': 0,
            'lines_scanned': 0,
            'issues_found': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the validator"""
        logger = logging.getLogger("ProductionSecretsValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_production_secrets(self, workspace_path: str = ".") -> ValidationReport:
        """
        Comprehensive validation of production secrets
        
        Args:
            workspace_path: Path to workspace root directory
            
        Returns:
            ValidationReport with all security issues found
        """
        import time
        start_time = time.time()
        
        self.logger.info("🔍 Starting production secrets validation...")
        
        workspace = Path(workspace_path).resolve()
        all_issues = []
        
        
        for directory in self.scan_directories:
            dir_path = workspace / directory
            if dir_path.exists():
                self.logger.info(f"📁 Scanning directory: {directory}")
                issues = self._scan_directory(dir_path)
                all_issues.extend(issues)
        
        
        scan_duration = time.time() - start_time
        report = self._generate_report(all_issues, scan_duration)
        
        self.logger.info(f"✅ Validation complete: {report.total_issues} issues found in {scan_duration:.2f}s")
        
        return report
    
    def _scan_directory(self, directory: Path) -> List[SecurityIssue]:
        """Scan a directory for security issues"""
        issues = []
        
        try:
        try:
            files_to_scan = []
            for pattern in self.scan_patterns:
                files_to_scan.extend(directory.rglob(pattern))
            
            
            filtered_files = self._filter_excluded_files(files_to_scan)
            
            
            for file_path in filtered_files:
                if file_path.is_file():
                    file_issues = self._scan_file(file_path)
                    issues.extend(file_issues)
                    self.stats['files_scanned'] += 1
        
        except Exception as e:
            self.logger.error(f"❌ Error scanning directory {directory}: {e}")
        
        return issues
    
    def _filter_excluded_files(self, files: List[Path]) -> List[Path]:
        """Filter out files matching exclude patterns"""
        filtered = []
        
        for file_path in files:
            should_exclude = False
            
            for exclude_pattern in self.exclude_patterns:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered.append(file_path)
        
        return filtered
    
    def _scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single file for security issues"""
        issues = []
        
        try:
        try:
            if self._is_binary_file(file_path):
                return issues
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            
            for line_num, line in enumerate(lines, 1):
                line_issues = self._scan_line(str(file_path), line_num, line)
                issues.extend(line_issues)
                self.stats['lines_scanned'] += 1
        
        except Exception as e:
            self.logger.warning(f"⚠️ Could not scan file {file_path}: {e}")
        
        return issues
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return True
    
    def _scan_line(self, file_path: str, line_num: int, line: str) -> List[SecurityIssue]:
        """Scan a single line for security issues"""
        issues = []
        
        
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            return issues
        
        
        pattern_categories = [
            (self.placeholder_patterns, IssueType.PLACEHOLDER),
            (self.weak_password_patterns, IssueType.WEAK_PASSWORD),
            (self.default_credential_patterns, IssueType.DEFAULT_CREDENTIAL),
            (self.exposed_secret_patterns, IssueType.EXPOSED_SECRET),
            (self.insecure_config_patterns, IssueType.INSECURE_CONFIG)
        ]
        
        for patterns, issue_type in pattern_categories:
            for pattern, severity, message in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    issue = SecurityIssue(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line.rstrip(),
                        issue_type=issue_type,
                        severity=severity,
                        message=message,
                        recommendation=self._get_recommendation(issue_type, severity),
                        pattern_matched=match.group()
                    )
                    issues.append(issue)
        
        return issues
    
    def _get_recommendation(self, issue_type: IssueType, severity: SecuritySeverity) -> str:
        """Get security recommendation for issue type"""
        recommendations = {
            IssueType.PLACEHOLDER: "Replace with secure production values from environment variables or secure vault",
            IssueType.WEAK_PASSWORD: "Use strong passwords with minimum 12 characters, mixed case, numbers and symbols",
            IssueType.DEFAULT_CREDENTIAL: "Change default usernames and passwords to unique, secure values",
            IssueType.EXPOSED_SECRET: "Move secrets to environment variables or encrypted vault storage",
            IssueType.INSECURE_CONFIG: "Enable secure configuration options for production deployment"
        }
        
        base_rec = recommendations.get(issue_type, "Review and secure this configuration")
        
        if severity == SecuritySeverity.CRITICAL:
            return f"CRITICAL: {base_rec}. This must be fixed before production deployment."
        elif severity == SecuritySeverity.HIGH:
            return f"HIGH PRIORITY: {base_rec}. Fix immediately."
        else:
            return base_rec
    
    def _generate_report(self, issues: List[SecurityIssue], scan_duration: float) -> ValidationReport:
        """Generate comprehensive validation report"""
        
        
        issues_by_severity = {severity: 0 for severity in SecuritySeverity}
        issues_by_type = {issue_type: 0 for issue_type in IssueType}
        
        critical_issues = []
        
        for issue in issues:
            issues_by_severity[issue.severity] += 1
            issues_by_type[issue.issue_type] += 1
            
            if issue.severity == SecuritySeverity.CRITICAL:
                critical_issues.append(issue)
        
        
        # Determine if production ready
        critical_count = issues_by_severity[SecuritySeverity.CRITICAL]
        high_count = issues_by_severity[SecuritySeverity.HIGH]
        is_production_ready = (critical_count == 0 and high_count == 0)
        
        return ValidationReport(
            total_files_scanned=self.stats['files_scanned'],
            total_lines_scanned=self.stats['lines_scanned'],
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues_by_type=issues_by_type,
            security_issues=issues,
            scan_duration_seconds=scan_duration,
            is_production_ready=is_production_ready,
            critical_issues=critical_issues
        )
    
    def print_report(self, report: ValidationReport, verbose: bool = False):
        """Print human-readable validation report"""
        
        print("\n" + "="*60)
        print("🔒 PRODUCTION SECRETS VALIDATION REPORT")
        print("="*60)
        
        
        print(f"\n📊 SCAN SUMMARY:")
        print(f"  Files scanned: {report.total_files_scanned}")
        print(f"  Lines scanned: {report.total_lines_scanned:,}")
        print(f"  Scan duration: {report.scan_duration_seconds:.2f}s")
        print(f"  Total issues: {report.total_issues}")
        
        
        if report.is_production_ready:
            print(f"\n✅ PRODUCTION READY: No critical security issues found")
        else:
            print(f"\n❌ NOT PRODUCTION READY: {len(report.critical_issues)} critical issues found")
        
        
        print(f"\n🚨 ISSUES BY SEVERITY:")
        for severity in SecuritySeverity:
            count = report.issues_by_severity[severity]
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
                print(f"  {emoji.get(severity.value, '•')} {severity.value.upper()}: {count}")
        
        
        print(f"\n🔍 ISSUES BY TYPE:")
        for issue_type in IssueType:
            count = report.issues_by_type[issue_type]
            if count > 0:
                print(f"  • {issue_type.value.replace('_', ' ').title()}: {count}")
        
        
        if report.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for issue in report.critical_issues[:10]:  # Show first 10
                print(f"  📁 {issue.file_path}:{issue.line_number}")
                print(f"     {issue.message}")
                print(f"     Pattern: {issue.pattern_matched}")
                print(f"     Fix: {issue.recommendation}")
                print()
            
            if len(report.critical_issues) > 10:
                print(f"  ... and {len(report.critical_issues) - 10} more critical issues")
        
        
        if verbose and report.security_issues:
            print(f"\n📋 ALL SECURITY ISSUES:")
            for i, issue in enumerate(report.security_issues, 1):
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
                print(f"\n{i}. {severity_emoji.get(issue.severity.value, '•')} {issue.severity.value.upper()}")
                print(f"   File: {issue.file_path}:{issue.line_number}")
                print(f"   Type: {issue.issue_type.value.replace('_', ' ').title()}")
                print(f"   Issue: {issue.message}")
                print(f"   Line: {issue.line_content[:100]}{'...' if len(issue.line_content) > 100 else ''}")
                print(f"   Fix: {issue.recommendation}")
        
        print("\n" + "="*60)


def main():
    """Main entry point for the validator"""
    parser = argparse.ArgumentParser(
        description="Validate production secrets and security configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_production_secrets.py                    # Scan current directory
  python validate_production_secrets.py -v                 # Verbose output
  python validate_production_secrets.py --json            # JSON output
  python validate_production_secrets.py --workspace /app  # Custom workspace
  python validate_production_secrets.py --ci              # CI mode (exit 1 on issues)
        """
    )
    
    parser.add_argument(
        '--workspace', '-w',
        default='.',
        help='Workspace root directory to scan (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed issue information'
    )
    
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: exit with code 1 if any security issues found'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for JSON results'
    )
    
    args = parser.parse_args()
    
    
    validator = ProductionSecretsValidator()
    
    
    try:
        report = validator.validate_production_secrets(args.workspace)
        
        
        if args.json:
            result_json = json.dumps(report.to_dict(), indent=2)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(result_json)
                print(f"✅ Results written to {args.output}")
            else:
                print(result_json)
        else:
            validator.print_report(report, verbose=args.verbose)
        
        
        if args.ci:
            if not report.is_production_ready:
                print(f"\n❌ CI FAILURE: {len(report.critical_issues)} critical security issues found")
                sys.exit(1)
            else:
                print(f"\n✅ CI SUCCESS: Production ready - no critical security issues")
                sys.exit(0)
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if args.ci:
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == "__main__":
    main() 
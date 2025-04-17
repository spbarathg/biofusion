#!/usr/bin/env python3
"""
AntBot Wallet CLI - Command-line tool for wallet management operations
"""

import os
import sys
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.wallet_manager import WalletManager

def main():
    parser = argparse.ArgumentParser(
        description="AntBot Wallet Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wallet_cli.py create --name "queen_main" --type queen
  wallet_cli.py list
  wallet_cli.py balance --id <wallet_id>
  wallet_cli.py transfer --from <wallet_id> --to <wallet_id> --amount 1.0
  wallet_cli.py backup --path ./backup.json
  wallet_cli.py restore --path ./backup.json
""")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create wallet command
    create_parser = subparsers.add_parser("create", help="Create a new wallet")
    create_parser.add_argument("--name", type=str, required=True, help="Wallet name")
    create_parser.add_argument("--type", type=str, required=True, 
                              choices=["queen", "princess", "worker", "savings"], 
                              help="Wallet type")
    
    # List wallets command
    list_parser = subparsers.add_parser("list", help="List all wallets")
    list_parser.add_argument("--type", type=str, 
                            choices=["queen", "princess", "worker", "savings"], 
                            help="Filter by wallet type")
    
    # Check balance command
    balance_parser = subparsers.add_parser("balance", help="Check wallet balance")
    balance_group = balance_parser.add_mutually_exclusive_group(required=True)
    balance_group.add_argument("--id", type=str, help="Wallet ID")
    balance_group.add_argument("--name", type=str, help="Wallet name")
    
    # Transfer SOL command
    transfer_parser = subparsers.add_parser("transfer", help="Transfer SOL between wallets")
    transfer_parser.add_argument("--from", dest="from_wallet", type=str, required=True, 
                               help="Source wallet ID or name")
    transfer_parser.add_argument("--to", dest="to_wallet", type=str, required=True, 
                               help="Destination wallet ID or name")
    transfer_parser.add_argument("--amount", type=float, required=True, 
                                help="Amount to transfer in SOL")
    
    # External transfer command
    ext_transfer_parser = subparsers.add_parser("external-transfer", 
                                             help="Transfer SOL to external wallet")
    ext_transfer_parser.add_argument("--from", dest="from_wallet", type=str, required=True, 
                                   help="Source wallet ID or name")
    ext_transfer_parser.add_argument("--to-address", type=str, required=True, 
                                   help="Destination public key")
    ext_transfer_parser.add_argument("--amount", type=float, required=True, 
                                    help="Amount to transfer in SOL")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create wallet backup")
    backup_parser.add_argument("--path", type=str, help="Backup file path")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("--path", type=str, required=True, help="Backup file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize wallet manager
    wallet_manager = WalletManager()
    
    # Process commands
    if args.command == "create":
        wallet_id = wallet_manager.create_wallet(args.name, args.type)
        print(f"Created new {args.type} wallet:")
        print(f"  ID: {wallet_id}")
        print(f"  Name: {args.name}")
        print(f"  Public Key: {wallet_manager.wallets[wallet_id]['public_key']}")
    
    elif args.command == "list":
        wallets = wallet_manager.list_wallets(args.type)
        if not wallets:
            print(f"No wallets found{' of type ' + args.type if args.type else ''}")
            return
            
        print(f"Found {len(wallets)} wallets{' of type ' + args.type if args.type else ''}:")
        for wallet in wallets:
            print(f"  {wallet['name']} ({wallet['type']}):")
            print(f"    ID: {wallet['id']}")
            print(f"    Public Key: {wallet['public_key']}")
    
    elif args.command == "balance":
        wallet_id = None
        if args.id:
            wallet_id = args.id
        elif args.name:
            wallet_id = wallet_manager.get_wallet_by_name(args.name)
        
        if not wallet_id:
            print(f"Wallet not found")
            return
            
        try:
            balance = wallet_manager.get_balance(wallet_id)
            wallet_name = wallet_manager.wallets[wallet_id]['name']
            print(f"Balance for {wallet_name}: {balance} SOL")
        except Exception as e:
            print(f"Error getting balance: {e}")
    
    elif args.command == "transfer":
        # Resolve wallet IDs from names if needed
        from_id = args.from_wallet
        if not from_id in wallet_manager.wallets:
            from_id = wallet_manager.get_wallet_by_name(args.from_wallet)
            if not from_id:
                print(f"Source wallet '{args.from_wallet}' not found")
                return
        
        to_id = args.to_wallet
        if not to_id in wallet_manager.wallets:
            to_id = wallet_manager.get_wallet_by_name(args.to_wallet)
            if not to_id:
                print(f"Destination wallet '{args.to_wallet}' not found")
                return
        
        try:
            from_name = wallet_manager.wallets[from_id]['name']
            to_name = wallet_manager.wallets[to_id]['name']
            
            # Check balance before transfer
            balance = wallet_manager.get_balance(from_id)
            if balance < args.amount:
                print(f"Insufficient balance: {balance} SOL (need {args.amount} SOL)")
                return
            
            # Perform transfer
            signature = wallet_manager.transfer_sol(from_id, to_id, args.amount)
            print(f"Transferred {args.amount} SOL from {from_name} to {to_name}")
            print(f"Transaction signature: {signature}")
        except Exception as e:
            print(f"Error during transfer: {e}")
    
    elif args.command == "external-transfer":
        # Resolve wallet ID from name if needed
        from_id = args.from_wallet
        if not from_id in wallet_manager.wallets:
            from_id = wallet_manager.get_wallet_by_name(args.from_wallet)
            if not from_id:
                print(f"Source wallet '{args.from_wallet}' not found")
                return
        
        try:
            from_name = wallet_manager.wallets[from_id]['name']
            
            # Check balance before transfer
            balance = wallet_manager.get_balance(from_id)
            if balance < args.amount:
                print(f"Insufficient balance: {balance} SOL (need {args.amount} SOL)")
                return
            
            # Perform transfer
            signature = wallet_manager.transfer_sol_to_external(
                from_id, args.to_address, args.amount
            )
            print(f"Transferred {args.amount} SOL from {from_name} to {args.to_address}")
            print(f"Transaction signature: {signature}")
        except Exception as e:
            print(f"Error during external transfer: {e}")
    
    elif args.command == "backup":
        try:
            backup_path = wallet_manager.create_backup(args.path)
            print(f"Created encrypted wallet backup: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    elif args.command == "restore":
        try:
            count = wallet_manager.restore_from_backup(args.path)
            print(f"Restored {count} wallets from backup")
        except Exception as e:
            print(f"Error restoring from backup: {e}")

if __name__ == "__main__":
    main() 
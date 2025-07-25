"""
NIGHTLY EVOLUTION SYSTEM - GENETIC ALGORITHM FOR WALLET OPTIMIZATION
==================================================================

Implements evolutionary algorithms to continuously improve wallet performance
through genetic selection, mutation, and adaptation.
"""

import asyncio
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.database import get_database_manager


class EvolutionAction(Enum):
    """Evolution action types"""
    RETIRE = "retire"
    CLONE = "clone"
    MUTATE = "mutate"
    PROMOTE = "promote"
    MAINTAIN = "maintain"


@dataclass
class WalletFitnessMetrics:
    """Comprehensive wallet fitness evaluation"""
    wallet_id: str
    win_rate: float
    total_profit_sol: float
    average_profit_per_trade: float
    risk_adjusted_return: float
    consistency_score: float
    adaptability_score: float
    total_trades: int
    active_days: int
    overall_fitness: float
    evolution_action: EvolutionAction


@dataclass
class EvolutionGenetics:
    """Genetic traits for wallet evolution"""
    aggression: float = 0.5
    patience: float = 0.5
    risk_tolerance: float = 0.5
    momentum_bias: float = 0.5
    contrarian_bias: float = 0.5
    technical_weight: float = 0.3
    sentiment_weight: float = 0.3
    narrative_weight: float = 0.4
    
    def mutate(self, mutation_rate: float = 0.1) -> 'EvolutionGenetics':
        """Create a mutated copy of these genetics"""
        new_genetics = EvolutionGenetics()
        
        # Mutate each trait
        for attr_name in ['aggression', 'patience', 'risk_tolerance', 'momentum_bias', 
                         'contrarian_bias', 'technical_weight', 'sentiment_weight', 'narrative_weight']:
            current_value = getattr(self, attr_name)
            
            if random.random() < mutation_rate:
                # Apply gaussian mutation
                mutation = np.random.normal(0, 0.1)
                new_value = current_value + mutation
                
                # Clamp to valid range
                if attr_name.endswith('_weight'):
                    new_value = max(0.0, min(1.0, new_value))
                else:
                    new_value = max(0.0, min(1.0, new_value))
                
                setattr(new_genetics, attr_name, new_value)
            else:
                setattr(new_genetics, attr_name, current_value)
        
        # Normalize weights
        total_weight = new_genetics.technical_weight + new_genetics.sentiment_weight + new_genetics.narrative_weight
        if total_weight > 0:
            new_genetics.technical_weight /= total_weight
            new_genetics.sentiment_weight /= total_weight
            new_genetics.narrative_weight /= total_weight
        
        return new_genetics


class NightlyEvolutionSystem:
    """Genetic algorithm system for wallet evolution and optimization"""
    
    def __init__(self, wallet_manager=None, database_manager=None):
        self.logger = setup_logger("NightlyEvolutionSystem")
        
        # System dependencies
        self.wallet_manager = wallet_manager
        self.database_manager = database_manager or get_database_manager()
        
        # Evolution configuration
        self.evolution_config = {
            'evaluation_period_days': 7,
            'min_trades_for_evaluation': 10,
            'fitness_threshold_retire': 0.3,
            'fitness_threshold_clone': 0.7,
            'max_population_size': 50,
            'min_population_size': 10,
            'mutation_rate': 0.15,
            'elite_preservation_rate': 0.2,
            'diversity_factor': 0.1
        }
        
        # Performance tracking
        self.evolution_history: List[Dict[str, Any]] = []
        self.generation_count = 0
        self.total_evolved_wallets = 0
        
        # Running state
        self.is_running = False
        self.last_evolution_time: Optional[datetime] = None
        
        self.logger.info("🧬 Nightly Evolution System initialized")
    
    async def run_nightly_evolution(self) -> Dict[str, Any]:
        """Execute the complete nightly evolution cycle"""
        if self.is_running:
            self.logger.warning("Evolution already running, skipping this cycle")
            return {'success': False, 'reason': 'Already running'}
        
        self.is_running = True
        evolution_start = datetime.now()
        
        try:
            self.logger.info("🌙 Starting nightly evolution cycle...")
            
            # Step 1: Evaluate all wallet fitness
            wallet_fitness = await self._evaluate_wallet_fitness()
            if not wallet_fitness:
                self.logger.warning("No wallets to evaluate, skipping evolution")
                return {'success': False, 'reason': 'No wallets to evaluate'}
            
            # Step 2: Determine evolution actions
            evolution_plan = await self._create_evolution_plan(wallet_fitness)
            
            # Step 3: Execute evolution actions
            evolution_results = await self._execute_evolution_plan(evolution_plan)
            
            # Step 4: Update generation count and tracking
            self.generation_count += 1
            self.last_evolution_time = datetime.now()
            
            # Step 5: Record evolution history
            evolution_summary = {
                'generation': self.generation_count,
                'timestamp': evolution_start.isoformat(),
                'duration_seconds': (datetime.now() - evolution_start).total_seconds(),
                'wallets_evaluated': len(wallet_fitness),
                'actions_executed': evolution_results,
                'fitness_stats': self._calculate_fitness_statistics(wallet_fitness)
            }
            
            self.evolution_history.append(evolution_summary)
            
            self.logger.info(f"✅ Evolution cycle {self.generation_count} completed successfully")
            self.logger.info(f"📊 Actions: {evolution_results}")
            
            return {
                'success': True,
                'generation': self.generation_count,
                'summary': evolution_summary
            }
            
        except Exception as e:
            self.logger.error(f"❌ Evolution cycle failed: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            self.is_running = False
    
    async def _evaluate_wallet_fitness(self) -> List[WalletFitnessMetrics]:
        """Evaluate fitness of all wallets based on performance data"""
        try:
            # Query wallet performance from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.evolution_config['evaluation_period_days'])
            
            wallet_performance = await self.database_manager.get_wallet_performance_summary(
                start_date=start_date,
                end_date=end_date
            )
            
            fitness_results = []
            
            for wallet_data in wallet_performance:
                wallet_id = wallet_data['wallet_id']
                
                # Skip wallets with insufficient trading history
                if wallet_data['total_trades'] < self.evolution_config['min_trades_for_evaluation']:
                    continue
                
                # Calculate fitness metrics
                fitness = await self._calculate_wallet_fitness(wallet_data)
                fitness_results.append(fitness)
            
            # Sort by overall fitness (highest first)
            fitness_results.sort(key=lambda x: x.overall_fitness, reverse=True)
            
            self.logger.info(f"📊 Evaluated {len(fitness_results)} wallets for evolution")
            
            return fitness_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating wallet fitness: {e}")
            return []
    
    async def _calculate_wallet_fitness(self, wallet_data: Dict[str, Any]) -> WalletFitnessMetrics:
        """Calculate comprehensive fitness score for a wallet"""
        try:
            # Extract performance metrics
            total_trades = wallet_data.get('total_trades', 0)
            successful_trades = wallet_data.get('successful_trades', 0)
            total_profit_sol = wallet_data.get('total_profit_sol', 0.0)
            total_volume = wallet_data.get('total_volume_sol', 1.0)
            active_days = wallet_data.get('active_days', 1)
            
            # Calculate basic metrics
            win_rate = successful_trades / max(total_trades, 1)
            avg_profit_per_trade = total_profit_sol / max(total_trades, 1)
            
            # Risk-adjusted return (Sharpe-like ratio)
            profit_volatility = wallet_data.get('profit_volatility', 0.1)
            risk_adjusted_return = total_profit_sol / max(profit_volatility, 0.01)
            
            # Consistency score (how consistently profitable)
            profitable_days = wallet_data.get('profitable_days', 0)
            consistency_score = profitable_days / max(active_days, 1)
            
            # Adaptability score (performance in different market conditions)
            adaptability_score = self._calculate_adaptability_score(wallet_data)
            
            # Overall fitness calculation (weighted combination)
            fitness_components = {
                'win_rate': win_rate * 0.25,
                'profit_amount': min(total_profit_sol / 10.0, 1.0) * 0.20,  # Normalized profit
                'risk_adjusted': min(risk_adjusted_return / 5.0, 1.0) * 0.20,
                'consistency': consistency_score * 0.20,
                'adaptability': adaptability_score * 0.15
            }
            
            overall_fitness = sum(fitness_components.values())
            
            # Determine evolution action
            if overall_fitness >= self.evolution_config['fitness_threshold_clone']:
                evolution_action = EvolutionAction.CLONE
            elif overall_fitness <= self.evolution_config['fitness_threshold_retire']:
                evolution_action = EvolutionAction.RETIRE
            else:
                evolution_action = EvolutionAction.MAINTAIN
            
            return WalletFitnessMetrics(
                wallet_id=wallet_data['wallet_id'],
                win_rate=win_rate,
                total_profit_sol=total_profit_sol,
                average_profit_per_trade=avg_profit_per_trade,
                risk_adjusted_return=risk_adjusted_return,
                consistency_score=consistency_score,
                adaptability_score=adaptability_score,
                total_trades=total_trades,
                active_days=active_days,
                overall_fitness=overall_fitness,
                evolution_action=evolution_action
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating fitness for wallet: {e}")
            return WalletFitnessMetrics(
                wallet_id=wallet_data.get('wallet_id', 'unknown'),
                win_rate=0.0,
                total_profit_sol=0.0,
                average_profit_per_trade=0.0,
                risk_adjusted_return=0.0,
                consistency_score=0.0,
                adaptability_score=0.0,
                total_trades=0,
                active_days=0,
                overall_fitness=0.0,
                evolution_action=EvolutionAction.RETIRE
            )
    
    def _calculate_adaptability_score(self, wallet_data: Dict[str, Any]) -> float:
        """Calculate how well wallet adapts to different market conditions"""
        try:
            # Placeholder for adaptability calculation
            # In production, this would analyze performance across different market regimes
            market_conditions = wallet_data.get('market_conditions_performance', {})
            
            if not market_conditions:
                return 0.5  # Neutral score
            
            # Calculate performance variance across conditions
            performances = list(market_conditions.values())
            if len(performances) < 2:
                return 0.5
            
            # Lower variance = better adaptability
            variance = np.var(performances)
            adaptability = max(0.0, 1.0 - variance)
            
            return adaptability
            
        except Exception:
            return 0.5
    
    async def _create_evolution_plan(self, wallet_fitness: List[WalletFitnessMetrics]) -> Dict[str, List[str]]:
        """Create evolution plan based on fitness evaluations"""
        plan = {
            'retire': [],
            'clone': [],
            'maintain': []
        }
        
        current_population = len(wallet_fitness)
        max_pop = self.evolution_config['max_population_size']
        min_pop = self.evolution_config['min_population_size']
        
        for fitness in wallet_fitness:
            if fitness.evolution_action == EvolutionAction.RETIRE and current_population > min_pop:
                plan['retire'].append(fitness.wallet_id)
                current_population -= 1
            elif fitness.evolution_action == EvolutionAction.CLONE and current_population < max_pop:
                plan['clone'].append(fitness.wallet_id)
                current_population += 1
            else:
                plan['maintain'].append(fitness.wallet_id)
        
        self.logger.info(f"🎯 Evolution plan: Retire {len(plan['retire'])}, Clone {len(plan['clone'])}, Maintain {len(plan['maintain'])}")
        
        return plan
    
    async def _execute_evolution_plan(self, evolution_plan: Dict[str, List[str]]) -> Dict[str, int]:
        """Execute the evolution plan"""
        results = {
            'retired': 0,
            'cloned': 0,
            'maintained': 0,
            'errors': 0
        }
        
        try:
            # Retire poor performers
            for wallet_id in evolution_plan['retire']:
                try:
                    if self.wallet_manager:
                        await self.wallet_manager.retire_wallet(wallet_id)
                        results['retired'] += 1
                        self.logger.info(f"🪦 Retired wallet {wallet_id}")
                except Exception as e:
                    self.logger.error(f"Error retiring wallet {wallet_id}: {e}")
                    results['errors'] += 1
            
            # Clone high performers
            for wallet_id in evolution_plan['clone']:
                try:
                    if self.wallet_manager:
                        # Get the source wallet genetics
                        source_wallet = await self.wallet_manager.get_wallet(wallet_id)
                        if source_wallet:
                            # Create mutated genetics
                            new_genetics = source_wallet.genetics.mutate(self.evolution_config['mutation_rate'])
                            
                            # Create new wallet with mutated genetics
                            new_wallet_id = await self.wallet_manager.create_evolved_wallet(
                                parent_wallet_id=wallet_id,
                                genetics=new_genetics
                            )
                            
                            results['cloned'] += 1
                            self.total_evolved_wallets += 1
                            self.logger.info(f"🧬 Cloned wallet {wallet_id} -> {new_wallet_id}")
                except Exception as e:
                    self.logger.error(f"Error cloning wallet {wallet_id}: {e}")
                    results['errors'] += 1
            
            # Maintain others (no action needed)
            results['maintained'] = len(evolution_plan['maintain'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing evolution plan: {e}")
            results['errors'] += 1
            return results
    
    def _calculate_fitness_statistics(self, wallet_fitness: List[WalletFitnessMetrics]) -> Dict[str, float]:
        """Calculate statistical summary of fitness scores"""
        if not wallet_fitness:
            return {}
        
        fitness_scores = [w.overall_fitness for w in wallet_fitness]
        
        return {
            'mean_fitness': np.mean(fitness_scores),
            'median_fitness': np.median(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'min_fitness': np.min(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'total_wallets': len(wallet_fitness)
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution system status"""
        return {
            'generation_count': self.generation_count,
            'total_evolved_wallets': self.total_evolved_wallets,
            'is_running': self.is_running,
            'last_evolution_time': self.last_evolution_time.isoformat() if self.last_evolution_time else None,
            'evolution_config': self.evolution_config,
            'recent_evolution_history': self.evolution_history[-5:] if self.evolution_history else []
        }
    
    async def start_evolution_scheduler(self, interval_hours: int = 24):
        """Start the nightly evolution scheduler"""
        self.logger.info(f"🕒 Starting evolution scheduler (every {interval_hours} hours)")
        
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
                await self.run_nightly_evolution()
            except Exception as e:
                self.logger.error(f"Evolution scheduler error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def shutdown(self):
        """Graceful shutdown of evolution system"""
        self.is_running = False
        self.logger.info("🛑 Nightly Evolution System shutting down")


# Global instance
_evolution_system = None

def get_evolution_system(wallet_manager=None) -> NightlyEvolutionSystem:
    """Get global evolution system instance"""
    global _evolution_system
    if _evolution_system is None:
        _evolution_system = NightlyEvolutionSystem(wallet_manager=wallet_manager)
    return _evolution_system 
"""
NIGHTLY EVOLUTION SYSTEM - GENETIC WALLET OPTIMIZATION
=====================================================

System that runs nightly to evolve wallet genetics based on performance,
optimize trading parameters, and implement adaptive learning mechanisms.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
import copy

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager as WalletManager, EvolutionGenetics

@dataclass
class WalletPerformanceMetrics:
    """Performance metrics for a wallet"""
    wallet_id: str
    total_trades: int
    successful_trades: int
    total_profit_loss: float
    win_rate: float
    avg_profit_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    risk_score: float
    adaptation_score: float
    survival_score: float

@dataclass
class EvolutionReport:
    """Report of nightly evolution results"""
    timestamp: datetime
    wallets_evolved: int
    wallets_killed: int
    wallets_cloned: int
    new_genetics_created: int
    avg_fitness_improvement: float
    best_performer: str
    worst_performer: str
    evolution_insights: List[str]

class NightlyEvolutionSystem:
    """Genetic evolution system for wallet optimization"""
    
    def __init__(self, wallet_manager: WalletManager):
        self.logger = setup_logger("NightlyEvolution")
        self.wallet_manager = wallet_manager
        
        
        self.evolution_config = {
            'mutation_rate': 0.15,           # Probability of mutation
            'crossover_rate': 0.7,           # Probability of crossover
            'elite_preservation': 0.3,       # Top % to preserve
            'kill_threshold': 0.2,           # Bottom % to kill
            'clone_threshold': 0.1,          # Top % to clone
            'adaptation_weight': 0.4,        # Weight for adaptation in fitness
            'profit_weight': 0.4,            # Weight for profit in fitness
            'risk_weight': 0.2,              # Weight for risk management
        }
        
        
        self.performance_history: Dict[str, List[WalletPerformanceMetrics]] = {}
        
        
        self.evolution_history: List[EvolutionReport] = []
        
        
        self.successful_patterns: List[EvolutionGenetics] = []
        
        # Genetic Reservoir: Store genetics from killed wallets for diversity preservation
        self.genetic_reservoir: List[Dict[str, Any]] = []
        self.genetic_reservoir_file = "data/genetic_reservoir.json"
        
        
        self.learning_database = {
            'successful_trades': [],
            'failed_trades': [],
            'market_conditions': [],
            'optimal_parameters': {}
        }
        
        # Load genetic reservoir from file
        self._load_genetic_reservoir()
        
        self.logger.info("✅ Nightly evolution system initialized with genetic diversity preservation")
    
    async def run_nightly_evolution(self) -> EvolutionReport:
        """Run the complete nightly evolution process"""
        
        self.logger.info("🧬 Starting nightly evolution process...")
        
        try:
            start_time = datetime.now()
            
            
            performance_metrics = await self._analyze_wallet_performance()
            
            
            fitness_scores = await self._calculate_fitness_scores(performance_metrics)
            
            
            evolution_decisions = await self._make_evolution_decisions(fitness_scores)
            
            
            evolution_results = await self._execute_genetic_operations(evolution_decisions)
            
            
            await self._update_learning_database(performance_metrics)
            
            
            await self._optimize_global_parameters()
            
            
            report = await self._generate_evolution_report(evolution_results, start_time)
            
            
            self.evolution_history.append(report)
            
            
            if len(self.evolution_history) > 30:
                self.evolution_history = self.evolution_history[-30:]
            
            self.logger.info(f"✅ Nightly evolution completed: {report.wallets_evolved} evolved, "
                           f"{report.wallets_killed} killed, {report.wallets_cloned} cloned")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Nightly evolution failed: {e}")
            return EvolutionReport(
                timestamp=datetime.now(),
                wallets_evolved=0,
                wallets_killed=0,
                wallets_cloned=0,
                new_genetics_created=0,
                avg_fitness_improvement=0.0,
                best_performer="",
                worst_performer="",
                evolution_insights=[f"Evolution failed: {str(e)}"]
            )
    
    async def _analyze_wallet_performance(self) -> Dict[str, WalletPerformanceMetrics]:
        """Analyze performance of all wallets over the last 24 hours"""
        
        performance_metrics = {}
        
        try:
            active_wallets = await self.wallet_manager.get_all_wallets()
            
            for wallet_id, wallet_info in active_wallets.items():
                metrics = await self._calculate_wallet_metrics(wallet_id, wallet_info)
                performance_metrics[wallet_id] = metrics
                
                
                if wallet_id not in self.performance_history:
                    self.performance_history[wallet_id] = []
                
                self.performance_history[wallet_id].append(metrics)
                
                
                if len(self.performance_history[wallet_id]) > 30:
                    self.performance_history[wallet_id] = self.performance_history[wallet_id][-30:]
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {}
    
    async def _calculate_wallet_metrics(self, wallet_id: str, wallet_info: Dict) -> WalletPerformanceMetrics:
        """Calculate detailed performance metrics for a single wallet"""
        
        try:
            trade_history = wallet_info.get('trade_history', [])
            recent_trades = [
                trade for trade in trade_history 
                if datetime.now() - datetime.fromisoformat(trade.get('timestamp', '2023-01-01')) < timedelta(hours=24)
            ]
            
            total_trades = len(recent_trades)
            successful_trades = len([t for t in recent_trades if t.get('profit_loss', 0) > 0])
            
            
            total_profit_loss = sum(t.get('profit_loss', 0) for t in recent_trades)
            win_rate = successful_trades / total_trades if total_trades > 0 else 0.0
            avg_profit_per_trade = total_profit_loss / total_trades if total_trades > 0 else 0.0
            
            
            max_drawdown = self._calculate_max_drawdown(recent_trades)
            sharpe_ratio = self._calculate_sharpe_ratio(recent_trades)
            risk_score = self._calculate_risk_score(wallet_info)
            
            
            adaptation_score = self._calculate_adaptation_score(wallet_id, wallet_info)
            
            
            survival_score = self._calculate_survival_score(
                win_rate, avg_profit_per_trade, risk_score, adaptation_score
            )
            
            return WalletPerformanceMetrics(
                wallet_id=wallet_id,
                total_trades=total_trades,
                successful_trades=successful_trades,
                total_profit_loss=total_profit_loss,
                win_rate=win_rate,
                avg_profit_per_trade=avg_profit_per_trade,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                risk_score=risk_score,
                adaptation_score=adaptation_score,
                survival_score=survival_score
            )
            
        except Exception as e:
            self.logger.warning(f"Metric calculation failed for wallet {wallet_id}: {e}")
            return WalletPerformanceMetrics(
                wallet_id=wallet_id,
                total_trades=0,
                successful_trades=0,
                total_profit_loss=0.0,
                win_rate=0.0,
                avg_profit_per_trade=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                risk_score=0.5,
                adaptation_score=0.5,
                survival_score=0.0
            )
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade history"""
        
        if not trades:
            return 0.0
        
        cumulative_pnl = []
        running_total = 0.0
        
        for trade in trades:
            running_total += trade.get('profit_loss', 0)
            cumulative_pnl.append(running_total)
        
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / abs(peak) if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trade returns"""
        
        if len(trades) < 2:
            return 0.0
        
        returns = [trade.get('profit_loss', 0) for trade in trades]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _calculate_risk_score(self, wallet_info: Dict) -> float:
        """Calculate risk management score for wallet"""
        
        genetics = wallet_info.get('genetics', {})
        
        
        risk_tolerance = genetics.get('risk_tolerance', 0.5)
        position_sizing = genetics.get('position_sizing_multiplier', 1.0)
        stop_loss_discipline = genetics.get('stop_loss_discipline', 0.5)
        
        
        risk_score = (
            (1.0 - risk_tolerance) * 0.4 +           # Lower risk tolerance = better
            min(1.0, 2.0 / position_sizing) * 0.3 +  # Smaller positions = better
            stop_loss_discipline * 0.3               # Better discipline = better
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _calculate_adaptation_score(self, wallet_id: str, wallet_info: Dict) -> float:
        """Calculate how well the wallet adapts to changing conditions"""
        
        if wallet_id not in self.performance_history:
            return 0.5
        
        history = self.performance_history[wallet_id]
        
        if len(history) < 3:
            return 0.5
        
        
        recent_scores = [h.survival_score for h in history[-7:]]  # Last week
        older_scores = [h.survival_score for h in history[-14:-7]] if len(history) >= 14 else recent_scores
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        
        adaptation = (recent_avg - older_avg + 1.0) / 2.0  # Normalize to 0-1
        
        return np.clip(adaptation, 0.0, 1.0)
    
    def _calculate_survival_score(self, win_rate: float, avg_profit: float, 
                                risk_score: float, adaptation_score: float) -> float:
        """Calculate overall survival score for wallet"""
        
        
        profit_normalized = np.tanh(avg_profit / 0.001)
        
        
        survival_score = (
            win_rate * self.evolution_config['profit_weight'] +
            profit_normalized * self.evolution_config['profit_weight'] +
            risk_score * self.evolution_config['risk_weight'] +
            adaptation_score * self.evolution_config['adaptation_weight']
        )
        
        return np.clip(survival_score, 0.0, 1.0)
    
    async def _calculate_fitness_scores(self, performance_metrics: Dict[str, WalletPerformanceMetrics]) -> Dict[str, float]:
        """Calculate fitness scores for all wallets"""
        
        if not performance_metrics:
            return {}
        
        
        survival_scores = {wallet_id: metrics.survival_score 
                          for wallet_id, metrics in performance_metrics.items()}
        
        
        if survival_scores:
            min_score = min(survival_scores.values())
            max_score = max(survival_scores.values())
            
            if max_score - min_score > 0:
                fitness_scores = {
                    wallet_id: (score - min_score) / (max_score - min_score)
                    for wallet_id, score in survival_scores.items()
                }
            else:
                fitness_scores = {wallet_id: 0.5 for wallet_id in survival_scores.keys()}
        else:
            fitness_scores = {}
        
        return fitness_scores
    
    async def _make_evolution_decisions(self, fitness_scores: Dict[str, float]) -> Dict[str, str]:
        """Decide which wallets to evolve, kill, or clone"""
        
        decisions = {}
        
        if not fitness_scores:
            return decisions
        
        
        sorted_wallets = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_wallets = len(sorted_wallets)
        
        
        elite_count = max(1, int(total_wallets * self.evolution_config['elite_preservation']))
        kill_count = max(0, int(total_wallets * self.evolution_config['kill_threshold']))
        clone_count = max(0, int(total_wallets * self.evolution_config['clone_threshold']))
        
        
        for i in range(elite_count):
            wallet_id, _ = sorted_wallets[i]
            decisions[wallet_id] = 'preserve'
        
        
        # Implement 10% wild card mandate for genetic diversity
        wildcard_count = max(1, int(clone_count * 0.1)) if len(self.genetic_reservoir) > 0 else 0
        normal_clone_count = clone_count - wildcard_count
        
        # Create wild card clones from genetic reservoir
        for i in range(wildcard_count):
            decisions[f"wildcard_clone_{i}"] = 'wildcard_clone'
        
        # Create normal clones from top performers
        for i in range(min(normal_clone_count, elite_count)):
            wallet_id, _ = sorted_wallets[i]
            decisions[f"{wallet_id}_clone"] = 'clone'
        
        
        for i in range(max(0, total_wallets - kill_count), total_wallets):
            wallet_id, _ = sorted_wallets[i]
            decisions[wallet_id] = 'kill'
        
        
        for i in range(elite_count, max(elite_count, total_wallets - kill_count)):
            wallet_id, _ = sorted_wallets[i]
            decisions[wallet_id] = 'evolve'
        
        return decisions
    
    async def _execute_genetic_operations(self, evolution_decisions: Dict[str, str]) -> Dict[str, int]:
        """Execute genetic operations based on evolution decisions"""
        
        results = {
            'evolved': 0,
            'killed': 0,
            'cloned': 0,
            'preserved': 0
        }
        
        try:
            all_wallets = await self.wallet_manager.get_all_wallets()
            
            
            for wallet_key, action in evolution_decisions.items():
                
                if action == 'kill':
                    if wallet_key in all_wallets:
                        await self._kill_wallet(wallet_key)
                        results['killed'] += 1
                
                elif action == 'evolve':
                    if wallet_key in all_wallets:
                        await self._evolve_wallet(wallet_key, all_wallets[wallet_key])
                        results['evolved'] += 1
                
                elif action == 'clone':
                    source_wallet = wallet_key.replace('_clone', '')
                    if source_wallet in all_wallets:
                        await self._clone_wallet(source_wallet, all_wallets[source_wallet])
                        results['cloned'] += 1
                
                elif action == 'wildcard_clone':
                    await self._create_wildcard_wallet(wallet_key)
                    results['cloned'] += 1
                
                elif action == 'preserve':
                    results['preserved'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Genetic operations failed: {e}")
            return results
    
    async def _kill_wallet(self, wallet_id: str):
        """Kill underperforming wallet and preserve its genetics in the reservoir"""
        
        try:
            self.logger.info(f"🪦 Killing underperforming wallet: {wallet_id}")
            
            # Get wallet info before killing
            wallet_info = await self.wallet_manager.get_wallet_info(wallet_id)
            if wallet_info:
                # Store genetics in genetic reservoir for diversity preservation
                await self._store_genetics_in_reservoir(wallet_id, wallet_info)
                
                # Archive wallet data
                await self._archive_wallet_data(wallet_id, wallet_info, "killed")
            
            # Remove wallet
            await self.wallet_manager.remove_wallet(wallet_id)
            
        except Exception as e:
            self.logger.error(f"Failed to kill wallet {wallet_id}: {e}")
    
    async def _store_genetics_in_reservoir(self, wallet_id: str, wallet_info: Dict):
        """Store wallet genetics in the genetic reservoir for future diversity"""
        try:
            genetics_data = {
                'wallet_id': wallet_id,
                'genetics': wallet_info.get('genetics', {}),
                'performance_metrics': {
                    'total_trades': wallet_info.get('total_trades', 0),
                    'win_rate': wallet_info.get('win_rate', 0.0),
                    'total_profit': wallet_info.get('total_profit', 0.0),
                    'survival_score': wallet_info.get('survival_score', 0.0),
                },
                'timestamp_killed': datetime.now().isoformat(),
                'reason': 'underperformance'
            }
            
            # Add to reservoir
            self.genetic_reservoir.append(genetics_data)
            
            # Limit reservoir size (keep last 200 genetic patterns)
            if len(self.genetic_reservoir) > 200:
                self.genetic_reservoir = self.genetic_reservoir[-200:]
            
            # Save to file
            self._save_genetic_reservoir()
            
            self.logger.info(f"🧬 Stored genetics from {wallet_id} in genetic reservoir ({len(self.genetic_reservoir)} total)")
            
        except Exception as e:
            self.logger.error(f"Failed to store genetics in reservoir: {e}")
    
    def _load_genetic_reservoir(self):
        """Load genetic reservoir from file"""
        try:
            import os
            if os.path.exists(self.genetic_reservoir_file):
                with open(self.genetic_reservoir_file, 'r') as f:
                    self.genetic_reservoir = json.load(f)
                self.logger.info(f"📚 Loaded {len(self.genetic_reservoir)} genetic patterns from reservoir")
            else:
                self.genetic_reservoir = []
                self.logger.info("📚 Initialized empty genetic reservoir")
        except Exception as e:
            self.logger.warning(f"Failed to load genetic reservoir: {e}")
            self.genetic_reservoir = []
    
    def _save_genetic_reservoir(self):
        """Save genetic reservoir to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.genetic_reservoir_file), exist_ok=True)
            with open(self.genetic_reservoir_file, 'w') as f:
                json.dump(self.genetic_reservoir, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save genetic reservoir: {e}")
    
    async def _evolve_wallet(self, wallet_id: str, wallet_info: Dict):
        """Evolve wallet genetics based on performance"""
        
        try:
            current_genetics = EvolutionGenetics(**wallet_info.get('genetics', {}))
            
            
            evolved_genetics = await self._mutate_genetics(current_genetics)
            
            
            await self.wallet_manager.update_wallet_genetics(wallet_id, evolved_genetics)
            
            self.logger.info(f"🧬 Evolved wallet {wallet_id} genetics")
            
        except Exception as e:
            self.logger.error(f"Failed to evolve wallet {wallet_id}: {e}")
    
    async def _clone_wallet(self, source_wallet_id: str, source_wallet_info: Dict):
        """Clone high-performing wallet"""
        
        try:
            source_genetics = EvolutionGenetics(**source_wallet_info.get('genetics', {}))
            
            
            cloned_genetics = await self._mutate_genetics(source_genetics, mutation_rate=0.05)
            
            
            new_wallet_id = f"clone_{source_wallet_id}_{int(datetime.now().timestamp())}"
            await self.wallet_manager.create_wallet(new_wallet_id, cloned_genetics)
            
            self.logger.info(f"👯 Cloned wallet {source_wallet_id} as {new_wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to clone wallet {source_wallet_id}: {e}")
    
    async def _create_wildcard_wallet(self, wallet_key: str):
        """Create a wild card wallet using genetics from the genetic reservoir"""
        try:
            if not self.genetic_reservoir:
                self.logger.warning("🃏 No genetics in reservoir for wild card creation")
                return
            
            # Randomly select genetics from the reservoir
            import random
            selected_genetics_data = random.choice(self.genetic_reservoir)
            reservoir_genetics = EvolutionGenetics(**selected_genetics_data['genetics'])
            
            # Apply light mutation to the reservoir genetics (smaller capital allocation initially)
            wildcard_genetics = await self._mutate_genetics(reservoir_genetics, mutation_rate=0.08)
            
            # Create new wallet with wild card genetics
            new_wallet_id = f"wildcard_{int(datetime.now().timestamp())}_{wallet_key.split('_')[-1]}"
            await self.wallet_manager.create_wallet(new_wallet_id, wildcard_genetics)
            
            # Assign smaller initial capital allocation for wild cards
            if hasattr(self.wallet_manager, 'set_wallet_capital_allocation'):
                await self.wallet_manager.set_wallet_capital_allocation(new_wallet_id, 0.5)  # 50% of normal allocation
            
            self.logger.info(f"🃏 Created wild card wallet {new_wallet_id} from genetic reservoir "
                           f"(source: {selected_genetics_data['wallet_id']})")
            
        except Exception as e:
            self.logger.error(f"Failed to create wild card wallet {wallet_key}: {e}")
    
    async def _mutate_genetics(self, genetics: EvolutionGenetics, 
                             mutation_rate: float = None) -> EvolutionGenetics:
        """Apply genetic mutations to evolve genetics"""
        
        if mutation_rate is None:
            mutation_rate = self.evolution_config['mutation_rate']
        
        
        genetics_dict = asdict(genetics)
        
        
        for key, value in genetics_dict.items():
            if isinstance(value, (int, float)) and np.random.random() < mutation_rate:
                
                if isinstance(value, float):
                    mutation = np.random.normal(0, 0.1)
                    new_value = value + mutation
                    
                    
                    if key in ['aggression', 'risk_tolerance', 'patience', 'signal_trust', 
                              'adaptation_rate', 'memory_strength']:
                        new_value = np.clip(new_value, 0.0, 1.0)
                    elif key == 'position_sizing_multiplier':
                        new_value = np.clip(new_value, 0.1, 2.0)
                    elif key == 'stop_loss_discipline':
                        new_value = np.clip(new_value, 0.0, 1.0)
                    
                    genetics_dict[key] = new_value
                
                elif isinstance(value, int):
                    mutation = np.random.randint(-2, 3)
                    new_value = max(1, value + mutation)
                    genetics_dict[key] = new_value
        
        return EvolutionGenetics(**genetics_dict)
    
    async def _update_learning_database(self, performance_metrics: Dict[str, WalletPerformanceMetrics]):
        """Update learning database with new insights"""
        
        try:
            successful_wallets = [
                (wallet_id, metrics) for wallet_id, metrics in performance_metrics.items()
                if metrics.survival_score > 0.7
            ]
            
            
            for wallet_id, metrics in successful_wallets:
                wallet_info = await self.wallet_manager.get_wallet_info(wallet_id)
                if wallet_info:
                    genetics = EvolutionGenetics(**wallet_info.get('genetics', {}))
                    self.successful_patterns.append(genetics)
            
            
            if len(self.successful_patterns) > 50:
                self.successful_patterns = self.successful_patterns[-50:]
            
            
            await self._update_optimal_parameters(performance_metrics)
            
        except Exception as e:
            self.logger.warning(f"Learning database update failed: {e}")
    
    async def _update_optimal_parameters(self, performance_metrics: Dict[str, WalletPerformanceMetrics]):
        """Update optimal parameter ranges based on successful wallets"""
        
        try:
            successful_wallets = [
                wallet_id for wallet_id, metrics in performance_metrics.items()
                if metrics.survival_score > 0.6
            ]
            
            if not successful_wallets:
                return
            
            
            successful_genetics = []
            for wallet_id in successful_wallets:
                wallet_info = await self.wallet_manager.get_wallet_info(wallet_id)
                if wallet_info:
                    successful_genetics.append(wallet_info.get('genetics', {}))
            
            if successful_genetics:
                optimal_ranges = {}
                for key in ['aggression', 'risk_tolerance', 'patience', 'signal_trust']:
                    values = [g.get(key, 0.5) for g in successful_genetics if key in g]
                    if values:
                        optimal_ranges[key] = {
                            'min': np.percentile(values, 25),
                            'max': np.percentile(values, 75),
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
                
                self.learning_database['optimal_parameters'] = optimal_ranges
            
        except Exception as e:
            self.logger.warning(f"Optimal parameter update failed: {e}")
    
    async def _optimize_global_parameters(self):
        """Optimize global system parameters based on performance"""
        
        try:
            if not self.evolution_history:
                return
            
            recent_evolution = self.evolution_history[-1]
            
            
            if recent_evolution.avg_fitness_improvement > 0.1:
                self.evolution_config['mutation_rate'] = max(0.05, 
                    self.evolution_config['mutation_rate'] * 0.9)
            elif recent_evolution.avg_fitness_improvement < 0.01:
                self.evolution_config['mutation_rate'] = min(0.3, 
                    self.evolution_config['mutation_rate'] * 1.1)
            
            self.logger.debug(f"Adjusted mutation rate to {self.evolution_config['mutation_rate']:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Global parameter optimization failed: {e}")
    
    async def _generate_evolution_report(self, evolution_results: Dict[str, int], 
                                       start_time: datetime) -> EvolutionReport:
        """Generate comprehensive evolution report"""
        
        try:
            insights = []
            
            if evolution_results['evolved'] > 0:
                insights.append(f"Evolved {evolution_results['evolved']} wallets with genetic mutations")
            
            if evolution_results['killed'] > 0:
                insights.append(f"Eliminated {evolution_results['killed']} underperforming wallets")
            
            if evolution_results['cloned'] > 0:
                insights.append(f"Cloned {evolution_results['cloned']} high-performing wallets")
            
            
            performance_metrics = await self._analyze_wallet_performance()
            
            if performance_metrics:
                best_performer = max(performance_metrics.items(), 
                                   key=lambda x: x[1].survival_score)[0]
                worst_performer = min(performance_metrics.items(), 
                                    key=lambda x: x[1].survival_score)[0]
                
                avg_fitness = np.mean([m.survival_score for m in performance_metrics.values()])
            else:
                best_performer = "Unknown"
                worst_performer = "Unknown"
                avg_fitness = 0.0
            
            
            if len(self.evolution_history) > 0:
                prev_avg_fitness = np.mean([
                    m.survival_score for m in self.performance_history.get(wallet_id, [])[-2:-1]
                    for wallet_id in performance_metrics.keys()
                    if wallet_id in self.performance_history and len(self.performance_history[wallet_id]) > 1
                ]) if any(len(self.performance_history.get(wid, [])) > 1 
                         for wid in performance_metrics.keys()) else avg_fitness
                
                fitness_improvement = avg_fitness - prev_avg_fitness
            else:
                fitness_improvement = 0.0
            
            return EvolutionReport(
                timestamp=datetime.now(),
                wallets_evolved=evolution_results['evolved'],
                wallets_killed=evolution_results['killed'],
                wallets_cloned=evolution_results['cloned'],
                new_genetics_created=evolution_results['evolved'] + evolution_results['cloned'],
                avg_fitness_improvement=fitness_improvement,
                best_performer=best_performer,
                worst_performer=worst_performer,
                evolution_insights=insights
            )
            
        except Exception as e:
            self.logger.error(f"Evolution report generation failed: {e}")
            return EvolutionReport(
                timestamp=datetime.now(),
                wallets_evolved=0,
                wallets_killed=0,
                wallets_cloned=0,
                new_genetics_created=0,
                avg_fitness_improvement=0.0,
                best_performer="",
                worst_performer="",
                evolution_insights=[f"Report generation failed: {str(e)}"]
            )
    
    async def _archive_wallet_data(self, wallet_id: str, wallet_info: Dict, reason: str):
        """Archive wallet data before removal"""
        
        try:
            archive_data = {
                'wallet_id': wallet_id,
                'genetics': wallet_info.get('genetics', {}),
                'performance_history': self.performance_history.get(wallet_id, []),
                'removal_reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            
            self.logger.debug(f"Archived wallet {wallet_id} data (reason: {reason})")
            
        except Exception as e:
            self.logger.warning(f"Failed to archive wallet {wallet_id}: {e}")
    
    async def update_signal_probabilities(self, database_connection) -> Dict[str, Any]:
        """
        Update signal probabilities for Naive Bayes win-rate calculation
        
        Analyzes historical trades to calculate:
        - P(Signal | Win): Probability of signal occurring in winning trades
        - P(Signal | Loss): Probability of signal occurring in losing trades
        
        Args:
            database_connection: TimescaleDB connection for querying trades
            
        Returns:
            Dict containing calculated probabilities and metadata
        """
        try:
            self.logger.info("📊 Starting signal probability analysis for Naive Bayes...")
            
            # Query trades from the past 30 days with signal snapshots
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            query = """
                SELECT success, signal_snapshot 
                FROM trades 
                WHERE timestamp >= $1 AND timestamp <= $2 
                AND signal_snapshot IS NOT NULL
                AND trade_type = 'BUY'
            """
            
            rows = await database_connection.fetch(query, start_date, end_date)
            
            if len(rows) < 50:  # Need minimum data for meaningful statistics
                self.logger.warning(f"Insufficient trade data ({len(rows)} trades) for signal probability analysis")
                return {"error": "insufficient_data", "trade_count": len(rows)}
            
            # Separate winning and losing trades
            winning_trades = []
            losing_trades = []
            
            for row in rows:
                if row['signal_snapshot']:
                    signals = json.loads(row['signal_snapshot']) if isinstance(row['signal_snapshot'], str) else row['signal_snapshot']
                    if row['success']:
                        winning_trades.append(signals)
                    else:
                        losing_trades.append(signals)
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            total_count = win_count + loss_count
            
            self.logger.info(f"📈 Analyzing {total_count} trades: {win_count} wins, {loss_count} losses")
            
            # Calculate base probabilities
            p_win = win_count / total_count if total_count > 0 else 0.5
            p_loss = loss_count / total_count if total_count > 0 else 0.5
            
            # Extract all unique signal types
            all_signals = set()
            for trade in winning_trades + losing_trades:
                all_signals.update(trade.keys())
            
            signal_probabilities = {
                'base_probabilities': {'p_win': p_win, 'p_loss': p_loss},
                'signal_conditionals': {},
                'metadata': {
                    'calculation_date': datetime.now().isoformat(),
                    'total_trades': total_count,
                    'winning_trades': win_count,
                    'losing_trades': loss_count,
                    'analysis_period_days': 30
                }
            }
            
            # Calculate conditional probabilities for each signal
            for signal_name in all_signals:
                try:
                    # Count signal occurrences in wins and losses
                    signal_in_wins = 0
                    signal_in_losses = 0
                    
                    for trade in winning_trades:
                        if signal_name in trade and self._signal_is_positive(trade[signal_name]):
                            signal_in_wins += 1
                    
                    for trade in losing_trades:
                        if signal_name in trade and self._signal_is_positive(trade[signal_name]):
                            signal_in_losses += 1
                    
                    # Calculate P(Signal | Win) and P(Signal | Loss)
                    p_signal_given_win = signal_in_wins / win_count if win_count > 0 else 0
                    p_signal_given_loss = signal_in_losses / loss_count if loss_count > 0 else 0
                    
                    signal_probabilities['signal_conditionals'][signal_name] = {
                        'p_signal_given_win': p_signal_given_win,
                        'p_signal_given_loss': p_signal_given_loss,
                        'signal_win_count': signal_in_wins,
                        'signal_loss_count': signal_in_losses,
                        'confidence': min(signal_in_wins + signal_in_losses, 100) / 100  # Confidence based on sample size
                    }
                    
                    self.logger.debug(f"📊 {signal_name}: P(S|W)={p_signal_given_win:.3f}, P(S|L)={p_signal_given_loss:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating probabilities for signal {signal_name}: {e}")
                    continue
            
            # Save to JSON file for fast access
            import os
            os.makedirs("data", exist_ok=True)
            
            with open("data/signal_probabilities.json", "w") as f:
                json.dump(signal_probabilities, f, indent=2)
            
            # Also cache in Redis if available
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
                await redis_client.set("signal_probabilities", json.dumps(signal_probabilities), ex=86400)  # 24 hour expiry
                await redis_client.close()
            except Exception as e:
                self.logger.debug(f"Redis caching unavailable: {e}")
            
            self.logger.info(f"✅ Signal probabilities updated: {len(signal_probabilities['signal_conditionals'])} signals analyzed")
            return signal_probabilities
            
        except Exception as e:
            self.logger.error(f"Error in signal probability analysis: {e}")
            return {"error": str(e)}
    
    def _signal_is_positive(self, signal_value: Any) -> bool:
        """
        Determine if a signal value should be considered 'positive' for analysis
        
        Args:
            signal_value: The signal value to evaluate
            
        Returns:
            bool: True if signal is considered positive
        """
        if isinstance(signal_value, (int, float)):
            return signal_value > 0.5  # Threshold for numerical signals
        elif isinstance(signal_value, bool):
            return signal_value
        elif isinstance(signal_value, str):
            return signal_value.lower() in ['true', 'positive', 'bullish', 'buy']
        else:
            return False
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution system status"""
        
        return {
            'total_evolutions': len(self.evolution_history),
            'successful_patterns_count': len(self.successful_patterns),
            'current_mutation_rate': self.evolution_config['mutation_rate'],
            'last_evolution': self.evolution_history[-1].timestamp.isoformat() if self.evolution_history else None,
            'optimal_parameters': self.learning_database.get('optimal_parameters', {}),
            'recent_insights': self.evolution_history[-1].evolution_insights if self.evolution_history else []
        } 
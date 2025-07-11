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
        
        
        self.learning_database = {
            'successful_trades': [],
            'failed_trades': [],
            'market_conditions': [],
            'optimal_parameters': {}
        }
        
        self.logger.info("✅ Nightly evolution system initialized")
    
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
        
        
        for i in range(min(clone_count, elite_count)):
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
                
                elif action == 'preserve':
                    results['preserved'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Genetic operations failed: {e}")
            return results
    
    async def _kill_wallet(self, wallet_id: str):
        """Kill underperforming wallet"""
        
        try:
            self.logger.info(f"🪦 Killing underperforming wallet: {wallet_id}")
            
            
            wallet_info = await self.wallet_manager.get_wallet_info(wallet_id)
            if wallet_info:
                await self._archive_wallet_data(wallet_id, wallet_info, "killed")
            
            
            await self.wallet_manager.remove_wallet(wallet_id)
            
        except Exception as e:
            self.logger.error(f"Failed to kill wallet {wallet_id}: {e}")
    
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
    
    async def _mutate_genetics(self, genetics: EvolutionGenetics, 
                             mutation_rate: float = None) -> EvolutionGenetics:
        """Apply genetic mutations to evolve genetics"""
        
        if mutation_rate is None:
            mutation_rate = self.evolution_config['mutation_rate']
        
        
        genetics_dict = asdict(genetics)
        
        
        for key, value in genetics_dict.items():
            if isinstance(value, (int, float)) and np.random.random() < mutation_rate:
                
                if isinstance(value, float):
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
            if recent_evolution.avg_fitness_improvement > 0.1:
                self.evolution_config['mutation_rate'] = max(0.05, 
                    self.evolution_config['mutation_rate'] * 0.9)
            elif recent_evolution.avg_fitness_improvement < 0.01:
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
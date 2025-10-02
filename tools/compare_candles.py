#!/usr/bin/env python3
"""
Nexus AI - Ferramenta de Comparação de Candles
Compara candles de diferentes fontes para validação de precisão.

Funcionalidades:
- Comparação entre extensão vs screen-share vs histórico
- Análise de diferenças OHLC
- Relatórios de precisão
- Visualização de divergências

Autor: Manus AI
Data: 2025-09-29
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics

import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CandleData:
    """Estrutura de dados para um candle."""
    symbol: str
    timeframe: str
    start: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    timestamp: float

@dataclass
class ComparisonResult:
    """Resultado da comparação entre candles."""
    symbol: str
    timeframe: str
    start: float
    source_a: str
    source_b: str
    
    # Diferenças absolutas
    open_diff: float
    high_diff: float
    low_diff: float
    close_diff: float
    volume_diff: int
    
    # Diferenças percentuais
    open_diff_pct: float
    high_diff_pct: float
    low_diff_pct: float
    close_diff_pct: float
    
    # Métricas agregadas
    max_diff: float
    avg_diff: float
    is_significant: bool

class CandleComparator:
    """Comparador de candles entre diferentes fontes."""
    
    def __init__(self, tolerance_pct: float = 0.01):
        """
        Inicializa o comparador.
        
        Args:
            tolerance_pct: Tolerância percentual para considerar diferenças significativas
        """
        self.tolerance_pct = tolerance_pct
        self.candles_by_source = {}  # source -> symbol -> timeframe -> [candles]
        self.comparison_results = []
    
    def load_candles_from_file(self, file_path: str, source: str):
        """
        Carrega candles de um arquivo JSON.
        
        Args:
            file_path: Caminho para arquivo JSON com candles
            source: Nome da fonte (ex: 'extension', 'screen_share', 'historical')
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            candles = []
            for item in data:
                if 'type' in item and item['type'] in ['candle:update', 'candle:closed']:
                    candle = CandleData(
                        symbol=item['symbol'],
                        timeframe=item['timeframe'],
                        start=item['start'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        close=item['close'],
                        volume=item.get('volume', 0),
                        source=source,
                        timestamp=item.get('timestamp', time.time())
                    )
                    candles.append(candle)
            
            self._organize_candles(candles, source)
            logger.info(f"Carregados {len(candles)} candles de {source} ({file_path})")
            
        except Exception as e:
            logger.error(f"Erro ao carregar candles de {file_path}: {e}")
    
    def load_candles_from_websocket_log(self, file_path: str, source: str):
        """
        Carrega candles de um log de WebSocket.
        
        Args:
            file_path: Caminho para arquivo de log
            source: Nome da fonte
        """
        try:
            candles = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        if data.get('type') in ['candle:update', 'candle:closed']:
                            candle = CandleData(
                                symbol=data['symbol'],
                                timeframe=data['timeframe'],
                                start=data['start'],
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                volume=data.get('volume', 0),
                                source=source,
                                timestamp=data.get('timestamp', time.time())
                            )
                            candles.append(candle)
                    except:
                        continue
            
            self._organize_candles(candles, source)
            logger.info(f"Carregados {len(candles)} candles de {source} (log WebSocket)")
            
        except Exception as e:
            logger.error(f"Erro ao carregar log WebSocket {file_path}: {e}")
    
    def _organize_candles(self, candles: List[CandleData], source: str):
        """Organiza candles por fonte, símbolo e timeframe."""
        if source not in self.candles_by_source:
            self.candles_by_source[source] = {}
        
        for candle in candles:
            if candle.symbol not in self.candles_by_source[source]:
                self.candles_by_source[source][candle.symbol] = {}
            
            if candle.timeframe not in self.candles_by_source[source][candle.symbol]:
                self.candles_by_source[source][candle.symbol][candle.timeframe] = []
            
            self.candles_by_source[source][candle.symbol][candle.timeframe].append(candle)
    
    def compare_sources(self, source_a: str, source_b: str) -> List[ComparisonResult]:
        """
        Compara candles entre duas fontes.
        
        Args:
            source_a: Nome da primeira fonte
            source_b: Nome da segunda fonte
        
        Returns:
            results: Lista de resultados de comparação
        """
        results = []
        
        if source_a not in self.candles_by_source or source_b not in self.candles_by_source:
            logger.error(f"Fontes {source_a} ou {source_b} não encontradas")
            return results
        
        # Iterar por símbolos e timeframes comuns
        symbols_a = set(self.candles_by_source[source_a].keys())
        symbols_b = set(self.candles_by_source[source_b].keys())
        common_symbols = symbols_a.intersection(symbols_b)
        
        for symbol in common_symbols:
            timeframes_a = set(self.candles_by_source[source_a][symbol].keys())
            timeframes_b = set(self.candles_by_source[source_b][symbol].keys())
            common_timeframes = timeframes_a.intersection(timeframes_b)
            
            for timeframe in common_timeframes:
                candles_a = self.candles_by_source[source_a][symbol][timeframe]
                candles_b = self.candles_by_source[source_b][symbol][timeframe]
                
                # Criar índices por timestamp de início
                index_a = {c.start: c for c in candles_a}
                index_b = {c.start: c for c in candles_b}
                
                # Comparar candles com mesmo timestamp
                common_starts = set(index_a.keys()).intersection(set(index_b.keys()))
                
                for start in common_starts:
                    candle_a = index_a[start]
                    candle_b = index_b[start]
                    
                    result = self._compare_candles(candle_a, candle_b)
                    results.append(result)
        
        self.comparison_results.extend(results)
        logger.info(f"Comparação concluída: {len(results)} candles comparados")
        
        return results
    
    def _compare_candles(self, candle_a: CandleData, candle_b: CandleData) -> ComparisonResult:
        """Compara dois candles individuais."""
        
        # Diferenças absolutas
        open_diff = abs(candle_a.open - candle_b.open)
        high_diff = abs(candle_a.high - candle_b.high)
        low_diff = abs(candle_a.low - candle_b.low)
        close_diff = abs(candle_a.close - candle_b.close)
        volume_diff = abs(candle_a.volume - candle_b.volume)
        
        # Diferenças percentuais
        open_diff_pct = (open_diff / candle_a.open) * 100 if candle_a.open > 0 else 0
        high_diff_pct = (high_diff / candle_a.high) * 100 if candle_a.high > 0 else 0
        low_diff_pct = (low_diff / candle_a.low) * 100 if candle_a.low > 0 else 0
        close_diff_pct = (close_diff / candle_a.close) * 100 if candle_a.close > 0 else 0
        
        # Métricas agregadas
        diffs = [open_diff_pct, high_diff_pct, low_diff_pct, close_diff_pct]
        max_diff = max(diffs)
        avg_diff = statistics.mean(diffs)
        is_significant = max_diff > self.tolerance_pct
        
        return ComparisonResult(
            symbol=candle_a.symbol,
            timeframe=candle_a.timeframe,
            start=candle_a.start,
            source_a=candle_a.source,
            source_b=candle_b.source,
            open_diff=open_diff,
            high_diff=high_diff,
            low_diff=low_diff,
            close_diff=close_diff,
            volume_diff=volume_diff,
            open_diff_pct=open_diff_pct,
            high_diff_pct=high_diff_pct,
            low_diff_pct=low_diff_pct,
            close_diff_pct=close_diff_pct,
            max_diff=max_diff,
            avg_diff=avg_diff,
            is_significant=is_significant
        )
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Gera relatório de comparação.
        
        Args:
            output_file: Arquivo para salvar relatório (opcional)
        
        Returns:
            report: Dict com estatísticas do relatório
        """
        if not self.comparison_results:
            logger.warning("Nenhum resultado de comparação disponível")
            return {}
        
        # Estatísticas gerais
        total_comparisons = len(self.comparison_results)
        significant_differences = sum(1 for r in self.comparison_results if r.is_significant)
        accuracy_rate = ((total_comparisons - significant_differences) / total_comparisons) * 100
        
        # Estatísticas por métrica
        open_diffs = [r.open_diff_pct for r in self.comparison_results]
        high_diffs = [r.high_diff_pct for r in self.comparison_results]
        low_diffs = [r.low_diff_pct for r in self.comparison_results]
        close_diffs = [r.close_diff_pct for r in self.comparison_results]
        max_diffs = [r.max_diff for r in self.comparison_results]
        
        report = {
            'summary': {
                'total_comparisons': total_comparisons,
                'significant_differences': significant_differences,
                'accuracy_rate_pct': accuracy_rate,
                'tolerance_pct': self.tolerance_pct
            },
            'statistics': {
                'open_diff': {
                    'mean': statistics.mean(open_diffs),
                    'median': statistics.median(open_diffs),
                    'max': max(open_diffs),
                    'std': statistics.stdev(open_diffs) if len(open_diffs) > 1 else 0
                },
                'high_diff': {
                    'mean': statistics.mean(high_diffs),
                    'median': statistics.median(high_diffs),
                    'max': max(high_diffs),
                    'std': statistics.stdev(high_diffs) if len(high_diffs) > 1 else 0
                },
                'low_diff': {
                    'mean': statistics.mean(low_diffs),
                    'median': statistics.median(low_diffs),
                    'max': max(low_diffs),
                    'std': statistics.stdev(low_diffs) if len(low_diffs) > 1 else 0
                },
                'close_diff': {
                    'mean': statistics.mean(close_diffs),
                    'median': statistics.median(close_diffs),
                    'max': max(close_diffs),
                    'std': statistics.stdev(close_diffs) if len(close_diffs) > 1 else 0
                },
                'max_diff': {
                    'mean': statistics.mean(max_diffs),
                    'median': statistics.median(max_diffs),
                    'max': max(max_diffs),
                    'std': statistics.stdev(max_diffs) if len(max_diffs) > 1 else 0
                }
            },
            'worst_cases': self._get_worst_cases(10),
            'by_symbol': self._get_stats_by_symbol(),
            'by_timeframe': self._get_stats_by_timeframe()
        }
        
        # Salvar relatório se solicitado
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Relatório salvo em: {output_file}")
            except Exception as e:
                logger.error(f"Erro ao salvar relatório: {e}")
        
        return report
    
    def _get_worst_cases(self, limit: int = 10) -> List[Dict]:
        """Retorna os piores casos de divergência."""
        sorted_results = sorted(self.comparison_results, key=lambda r: r.max_diff, reverse=True)
        
        worst_cases = []
        for result in sorted_results[:limit]:
            worst_cases.append({
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'start': result.start,
                'start_datetime': datetime.fromtimestamp(result.start, tz=timezone.utc).isoformat(),
                'source_a': result.source_a,
                'source_b': result.source_b,
                'max_diff_pct': result.max_diff,
                'avg_diff_pct': result.avg_diff,
                'open_diff_pct': result.open_diff_pct,
                'high_diff_pct': result.high_diff_pct,
                'low_diff_pct': result.low_diff_pct,
                'close_diff_pct': result.close_diff_pct
            })
        
        return worst_cases
    
    def _get_stats_by_symbol(self) -> Dict:
        """Retorna estatísticas agrupadas por símbolo."""
        by_symbol = {}
        
        for result in self.comparison_results:
            if result.symbol not in by_symbol:
                by_symbol[result.symbol] = []
            by_symbol[result.symbol].append(result)
        
        stats = {}
        for symbol, results in by_symbol.items():
            max_diffs = [r.max_diff for r in results]
            significant = sum(1 for r in results if r.is_significant)
            
            stats[symbol] = {
                'total_comparisons': len(results),
                'significant_differences': significant,
                'accuracy_rate_pct': ((len(results) - significant) / len(results)) * 100,
                'avg_max_diff_pct': statistics.mean(max_diffs),
                'max_diff_pct': max(max_diffs)
            }
        
        return stats
    
    def _get_stats_by_timeframe(self) -> Dict:
        """Retorna estatísticas agrupadas por timeframe."""
        by_timeframe = {}
        
        for result in self.comparison_results:
            if result.timeframe not in by_timeframe:
                by_timeframe[result.timeframe] = []
            by_timeframe[result.timeframe].append(result)
        
        stats = {}
        for timeframe, results in by_timeframe.items():
            max_diffs = [r.max_diff for r in results]
            significant = sum(1 for r in results if r.is_significant)
            
            stats[timeframe] = {
                'total_comparisons': len(results),
                'significant_differences': significant,
                'accuracy_rate_pct': ((len(results) - significant) / len(results)) * 100,
                'avg_max_diff_pct': statistics.mean(max_diffs),
                'max_diff_pct': max(max_diffs)
            }
        
        return stats
    
    def print_summary(self):
        """Imprime resumo da comparação."""
        if not self.comparison_results:
            print("❌ Nenhum resultado de comparação disponível")
            return
        
        report = self.generate_report()
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE COMPARAÇÃO DE CANDLES - NEXUS AI")
        print("="*60)
        print(f"Total de comparações: {summary['total_comparisons']}")
        print(f"Diferenças significativas: {summary['significant_differences']}")
        print(f"Taxa de precisão: {summary['accuracy_rate_pct']:.2f}%")
        print(f"Tolerância configurada: {summary['tolerance_pct']:.2f}%")
        
        if summary['accuracy_rate_pct'] >= 95:
            print("✅ EXCELENTE: Precisão muito alta entre as fontes")
        elif summary['accuracy_rate_pct'] >= 90:
            print("✅ BOM: Precisão adequada entre as fontes")
        elif summary['accuracy_rate_pct'] >= 80:
            print("⚠️ ATENÇÃO: Precisão moderada, revisar configurações")
        else:
            print("❌ CRÍTICO: Baixa precisão, verificar fontes de dados")
        
        print("\n📈 Estatísticas por Métrica:")
        stats = report['statistics']
        for metric, data in stats.items():
            print(f"  {metric.upper()}: média={data['mean']:.4f}%, máx={data['max']:.4f}%")
        
        print("\n🔍 Piores Casos:")
        for i, case in enumerate(report['worst_cases'][:5], 1):
            print(f"  {i}. {case['symbol']} {case['timeframe']} - {case['max_diff_pct']:.4f}%")

def main():
    """Função principal da ferramenta."""
    parser = argparse.ArgumentParser(description='Nexus AI - Comparador de Candles')
    parser.add_argument('--extension-file', help='Arquivo JSON com candles da extensão')
    parser.add_argument('--screen-file', help='Arquivo JSON com candles do screen-share')
    parser.add_argument('--historical-file', help='Arquivo JSON com candles históricos')
    parser.add_argument('--websocket-log', help='Arquivo de log do WebSocket')
    parser.add_argument('--tolerance', type=float, default=0.01, 
                       help='Tolerância percentual para diferenças (padrão: 0.01)')
    parser.add_argument('--output', help='Arquivo para salvar relatório JSON')
    parser.add_argument('--compare', nargs=2, metavar=('SOURCE_A', 'SOURCE_B'),
                       help='Comparar duas fontes específicas')
    
    args = parser.parse_args()
    
    # Criar comparador
    comparator = CandleComparator(tolerance_pct=args.tolerance)
    
    # Carregar dados
    if args.extension_file:
        comparator.load_candles_from_file(args.extension_file, 'extension')
    
    if args.screen_file:
        comparator.load_candles_from_file(args.screen_file, 'screen_share')
    
    if args.historical_file:
        comparator.load_candles_from_file(args.historical_file, 'historical')
    
    if args.websocket_log:
        comparator.load_candles_from_websocket_log(args.websocket_log, 'websocket')
    
    # Verificar se há dados suficientes
    if len(comparator.candles_by_source) < 2:
        logger.error("É necessário pelo menos duas fontes de dados para comparação")
        return
    
    # Realizar comparações
    sources = list(comparator.candles_by_source.keys())
    
    if args.compare:
        source_a, source_b = args.compare
        if source_a in sources and source_b in sources:
            comparator.compare_sources(source_a, source_b)
        else:
            logger.error(f"Fontes {source_a} ou {source_b} não encontradas")
            return
    else:
        # Comparar todas as combinações
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                comparator.compare_sources(sources[i], sources[j])
    
    # Gerar relatório
    report = comparator.generate_report(args.output)
    
    # Imprimir resumo
    comparator.print_summary()

if __name__ == '__main__':
    main()


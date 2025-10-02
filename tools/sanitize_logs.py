#!/usr/bin/env python3
"""
Nexus AI - Ferramenta de Sanitização de Logs
Remove informações sensíveis de logs e arquivos de dados.

Funcionalidades:
- Remoção de tokens, senhas e credenciais
- Mascaramento de dados pessoais
- Preservação da estrutura dos dados
- Relatório de sanitização

Autor: Manus AI
Data: 2025-09-29
"""

import argparse
import json
import re
import logging
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogSanitizer:
    """Sanitizador de logs e dados sensíveis."""
    
    def __init__(self):
        # Padrões de dados sensíveis
        self.sensitive_patterns = {
            'tokens': [
                r'token["\s]*[:=]["\s]*([^",\s}]+)',
                r'auth["\s]*[:=]["\s]*([^",\s}]+)',
                r'bearer\s+([a-zA-Z0-9\-_\.]+)',
                r'authorization["\s]*[:=]["\s]*([^",\s}]+)'
            ],
            'passwords': [
                r'password["\s]*[:=]["\s]*([^",\s}]+)',
                r'passwd["\s]*[:=]["\s]*([^",\s}]+)',
                r'pwd["\s]*[:=]["\s]*([^",\s}]+)'
            ],
            'keys': [
                r'key["\s]*[:=]["\s]*([^",\s}]+)',
                r'secret["\s]*[:=]["\s]*([^",\s}]+)',
                r'api_key["\s]*[:=]["\s]*([^",\s}]+)',
                r'private_key["\s]*[:=]["\s]*([^",\s}]+)'
            ],
            'emails': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'phones': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
            ],
            'ips': [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ],
            'urls': [
                r'https?://[^\s<>"{}|\\^`\[\]]+',
                r'ftp://[^\s<>"{}|\\^`\[\]]+',
                r'ws://[^\s<>"{}|\\^`\[\]]+',
                r'wss://[^\s<>"{}|\\^`\[\]]+'
            ]
        }
        
        # Campos sensíveis em estruturas JSON
        self.sensitive_fields = {
            'token', 'auth', 'authorization', 'bearer',
            'password', 'passwd', 'pwd', 'pass',
            'key', 'secret', 'api_key', 'private_key',
            'email', 'mail', 'phone', 'tel', 'mobile',
            'ssn', 'social_security', 'cpf', 'cnpj',
            'credit_card', 'card_number', 'cvv', 'cvc'
        }
        
        # Estatísticas de sanitização
        self.stats = {
            'files_processed': 0,
            'items_sanitized': 0,
            'patterns_found': {},
            'fields_sanitized': {}
        }
    
    def sanitize_file(self, input_file: str, output_file: str = None, 
                     preserve_structure: bool = True) -> bool:
        """
        Sanitiza um arquivo de log ou dados.
        
        Args:
            input_file: Arquivo de entrada
            output_file: Arquivo de saída (se None, sobrescreve o original)
            preserve_structure: Se deve preservar a estrutura JSON
        
        Returns:
            success: True se a sanitização foi bem-sucedida
        """
        try:
            if not os.path.exists(input_file):
                logger.error(f"Arquivo não encontrado: {input_file}")
                return False
            
            # Determinar arquivo de saída
            if output_file is None:
                output_file = input_file + '.sanitized'
            
            # Detectar tipo de arquivo
            file_ext = Path(input_file).suffix.lower()
            
            if file_ext == '.json':
                success = self._sanitize_json_file(input_file, output_file, preserve_structure)
            else:
                success = self._sanitize_text_file(input_file, output_file)
            
            if success:
                self.stats['files_processed'] += 1
                logger.info(f"Arquivo sanitizado: {input_file} -> {output_file}")
            
            return success
            
        except Exception as e:
            logger.error(f"Erro ao sanitizar arquivo {input_file}: {e}")
            return False
    
    def _sanitize_json_file(self, input_file: str, output_file: str, 
                           preserve_structure: bool) -> bool:
        """Sanitiza arquivo JSON."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tentar parsear como JSON
            try:
                data = json.loads(content)
                sanitized_data = self._sanitize_json_object(data)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
                
                return True
                
            except json.JSONDecodeError:
                # Se não for JSON válido, tratar como texto
                logger.warning(f"Arquivo {input_file} não é JSON válido, tratando como texto")
                return self._sanitize_text_file(input_file, output_file)
                
        except Exception as e:
            logger.error(f"Erro ao sanitizar JSON {input_file}: {e}")
            return False
    
    def _sanitize_text_file(self, input_file: str, output_file: str) -> bool:
        """Sanitiza arquivo de texto."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sanitized_content = self._sanitize_text(content)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(sanitized_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao sanitizar texto {input_file}: {e}")
            return False
    
    def _sanitize_json_object(self, obj: Any) -> Any:
        """Sanitiza objeto JSON recursivamente."""
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                # Verificar se a chave é sensível
                if self._is_sensitive_field(key):
                    sanitized[key] = self._mask_value(str(value), key)
                    self._update_field_stats(key)
                else:
                    sanitized[key] = self._sanitize_json_object(value)
            return sanitized
            
        elif isinstance(obj, list):
            return [self._sanitize_json_object(item) for item in obj]
            
        elif isinstance(obj, str):
            return self._sanitize_text(obj)
            
        else:
            return obj
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitiza texto usando padrões regex."""
        sanitized = text
        
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sanitized, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        # Substituir apenas o grupo capturado
                        sensitive_part = match.group(1)
                        masked_part = self._mask_value(sensitive_part, category)
                        sanitized = sanitized.replace(sensitive_part, masked_part)
                    else:
                        # Substituir toda a correspondência
                        masked_part = self._mask_value(match.group(0), category)
                        sanitized = sanitized.replace(match.group(0), masked_part)
                    
                    self._update_pattern_stats(category)
        
        return sanitized
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Verifica se um campo é sensível."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)
    
    def _mask_value(self, value: str, category: str) -> str:
        """
        Mascara um valor sensível.
        
        Args:
            value: Valor a ser mascarado
            category: Categoria do dado sensível
        
        Returns:
            masked_value: Valor mascarado
        """
        if not value:
            return value
        
        # Estratégias de mascaramento por categoria
        if category in ['tokens', 'keys', 'passwords']:
            # Manter apenas primeiros e últimos caracteres
            if len(value) <= 4:
                return '*' * len(value)
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
        
        elif category == 'emails':
            # Manter domínio, mascarar usuário
            if '@' in value:
                user, domain = value.split('@', 1)
                masked_user = user[0] + '*' * (len(user) - 1) if len(user) > 1 else '*'
                return f"{masked_user}@{domain}"
            else:
                return self._hash_value(value)
        
        elif category == 'phones':
            # Manter apenas últimos 4 dígitos
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return '*' * (len(digits) - 4) + digits[-4:]
            else:
                return '*' * len(digits)
        
        elif category == 'ips':
            # Mascarar últimos octetos
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.* "
            else:
                return self._hash_value(value)
        
        elif category == 'urls':
            # Manter apenas protocolo e domínio principal
            try:
                from urllib.parse import urlparse
                parsed = urlparse(value)
                domain_parts = parsed.netloc.split('.')
                if len(domain_parts) >= 2:
                    masked_domain = '*.'.join(domain_parts[-2:])
                    return f"{parsed.scheme}://{masked_domain}/***"
                else:
                    return f"{parsed.scheme}://***"
            except:
                return '[REDACTED_URL]'
        
        else:
            # Mascaramento genérico
            return self._hash_value(value)
    
    def _hash_value(self, value: str) -> str:
        """Cria hash determinístico para um valor."""
        hash_obj = hashlib.md5(value.encode('utf-8'))
        return f"[HASH_{hash_obj.hexdigest()[:8].upper()}]"
    
    def _update_pattern_stats(self, category: str):
        """Atualiza estatísticas de padrões encontrados."""
        if category not in self.stats['patterns_found']:
            self.stats['patterns_found'][category] = 0
        self.stats['patterns_found'][category] += 1
        self.stats['items_sanitized'] += 1
    
    def _update_field_stats(self, field: str):
        """Atualiza estatísticas de campos sanitizados."""
        if field not in self.stats['fields_sanitized']:
            self.stats['fields_sanitized'][field] = 0
        self.stats['fields_sanitized'][field] += 1
        self.stats['items_sanitized'] += 1
    
    def sanitize_directory(self, directory: str, output_dir: str = None, 
                          file_patterns: List[str] = None) -> bool:
        """
        Sanitiza todos os arquivos em um diretório.
        
        Args:
            directory: Diretório de entrada
            output_dir: Diretório de saída (se None, cria subdiretório 'sanitized')
            file_patterns: Padrões de arquivos para processar
        
        Returns:
            success: True se a sanitização foi bem-sucedida
        """
        try:
            if not os.path.exists(directory):
                logger.error(f"Diretório não encontrado: {directory}")
                return False
            
            # Determinar diretório de saída
            if output_dir is None:
                output_dir = os.path.join(directory, 'sanitized')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Padrões padrão se não especificados
            if file_patterns is None:
                file_patterns = ['*.log', '*.json', '*.txt', '*.csv']
            
            # Processar arquivos
            processed_count = 0
            
            for pattern in file_patterns:
                for file_path in Path(directory).glob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(directory)
                        output_file = os.path.join(output_dir, relative_path)
                        
                        # Criar diretórios necessários
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        if self.sanitize_file(str(file_path), output_file):
                            processed_count += 1
            
            logger.info(f"Diretório sanitizado: {processed_count} arquivos processados")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao sanitizar diretório {directory}: {e}")
            return False
    
    def generate_report(self) -> Dict:
        """Gera relatório de sanitização."""
        return {
            'summary': {
                'files_processed': self.stats['files_processed'],
                'items_sanitized': self.stats['items_sanitized']
            },
            'patterns_found': self.stats['patterns_found'],
            'fields_sanitized': self.stats['fields_sanitized'],
            'categories': list(self.sensitive_patterns.keys()),
            'sensitive_fields': list(self.sensitive_fields)
        }
    
    def print_report(self):
        """Imprime relatório de sanitização."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("🔒 RELATÓRIO DE SANITIZAÇÃO - NEXUS AI")
        print("="*60)
        print(f"Arquivos processados: {report['summary']['files_processed']}")
        print(f"Itens sanitizados: {report['summary']['items_sanitized']}")
        
        if report['patterns_found']:
            print("\n📊 Padrões Encontrados:")
            for category, count in report['patterns_found'].items():
                print(f"  {category}: {count}")
        
        if report['fields_sanitized']:
            print("\n🔑 Campos Sanitizados:")
            for field, count in report['fields_sanitized'].items():
                print(f"  {field}: {count}")
        
        if report['summary']['items_sanitized'] > 0:
            print("\n✅ Sanitização concluída com sucesso!")
        else:
            print("\n⚠️ Nenhum dado sensível encontrado.")

def main():
    """Função principal da ferramenta."""
    parser = argparse.ArgumentParser(description='Nexus AI - Sanitizador de Logs')
    parser.add_argument('input', help='Arquivo ou diretório de entrada')
    parser.add_argument('-o', '--output', help='Arquivo ou diretório de saída')
    parser.add_argument('-d', '--directory', action='store_true',
                       help='Processar diretório inteiro')
    parser.add_argument('-p', '--patterns', nargs='+', 
                       default=['*.log', '*.json', '*.txt'],
                       help='Padrões de arquivos para processar (apenas para diretórios)')
    parser.add_argument('-r', '--report', help='Arquivo para salvar relatório JSON')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preservar estrutura JSON original')
    
    args = parser.parse_args()
    
    # Criar sanitizador
    sanitizer = LogSanitizer()
    
    # Processar entrada
    if args.directory:
        success = sanitizer.sanitize_directory(args.input, args.output, args.patterns)
    else:
        success = sanitizer.sanitize_file(args.input, args.output, args.preserve_structure)
    
    # Gerar relatório
    if success:
        sanitizer.print_report()
        
        if args.report:
            try:
                report = sanitizer.generate_report()
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Relatório salvo em: {args.report}")
            except Exception as e:
                logger.error(f"Erro ao salvar relatório: {e}")
    else:
        logger.error("Falha na sanitização")

if __name__ == '__main__':
    main()


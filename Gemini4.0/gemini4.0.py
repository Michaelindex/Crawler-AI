import pandas as pd
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
import concurrent.futures
import time
import logging
import sys
from io import StringIO
import re # Importar regex para extração de JSON

# Configuração do logging
def setup_logging():
    """Configura o sistema de logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'gemini4.0_{timestamp}.log'
    
    # Configuração do logger
    logger = logging.getLogger("gemini4.0")
    logger.setLevel(logging.DEBUG)
    
    # Remover handlers existentes para evitar duplicação
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Handler para arquivo
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formato do log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adiciona os handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_api_keys(logger):
    """Carrega as 8 chaves da API dos arquivos."""
    keys = []
    for i in range(1, 9):
        key_file = f"../apis/gemini{i}.key" if i > 1 else "../apis/gemini.key"
        try:
            with open(key_file, 'r') as f:
                key = f.read().strip()
                keys.append(key)
                logger.debug(f"Chave API {i} carregada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar chave API {i}: {str(e)}")
            raise
    logger.info(f"Total de {len(keys)} chaves API carregadas")
    return keys

def load_email_examples(logger):
    """Carrega os exemplos de e-mails para treinamento."""
    try:
        with open('exemplos.txt', 'r', encoding='utf-8') as f:
            examples = f.read()
            logger.debug(f"Exemplos de e-mail carregados: {len(examples)} caracteres")
            return examples
    except Exception as e:
        logger.error(f"Erro ao carregar exemplos de e-mail: {str(e)}")
        raise

def build_prompt(row_data, iteration, email_examples, logger):
    """Constrói o prompt para cada iteração."""
    dados_atuais_json = json.dumps(row_data, indent=2, ensure_ascii=False)
    
    if iteration < 6:
        prompt = f"""
Você é um assistente especialista em encontrar e organizar informações de profissionais de saúde no Brasil.

**Tarefa Principal:** Sua missão é completar e padronizar os dados do médico abaixo em uma única etapa.

**Dados Atuais do Médico:**
```json
{dados_atuais_json}
```

**Instruções:**
1. Utilize a ferramenta de busca para encontrar exclusivamente as informações que estão ausentes ou claramente desatualizadas nos dados atuais.
2. Compile TODOS os dados (tanto os que você já tinha quanto os que encontrou) em um único JSON, seguindo as regras de padronização abaixo.

**Regras de Padronização Obrigatórias:**
- **Especialidade:** Deve conter apenas o nome da especialidade. Exemplo: "Cardiologia", "Dermatologia".
- **Endereço:**
  - `endereco_completo_a1`: Junte todas as partes do endereço em uma única string.
  - `logradouro_a1`: Nome da rua/avenida.
  - `numero_a1`: Apenas o número.
  - `complemento_a1`: Sala, andar, bloco, etc.
  - `bairro_a1`: Nome do bairro.
  - `cep_a1`: Formato 00000-000.
  - `cidade_a1`: Nome da cidade.
  - `estado_a1`: Sigla da UF (ex: "SP", "RJ").
- **Telefones e Celulares:**
  - Padronize para o formato +55 (DDD) 9XXXX-XXXX para celulares e +55 (DDD) XXXX-XXXX para fixos.
  - Inclua o código do país +55.
  - Não inclua números incompletos ou genéricos (com "X" ou "*").
- **E-mails:**
  - Converta para letras minúsculas.
  - Remova espaços em branco no início ou fim.

Retorne APENAS um objeto JSON válido, sem nenhum texto ou explicação adicional. Preencha todos os campos. Se uma informação não for encontrada, retorne um valor vazio "" para a chave correspondente.
"""
    elif iteration == 6:
        prompt = f"""
Você é um especialista em encontrar informações de contato de profissionais de saúde.

**Tarefa CRÍTICA:** Encontrar números de telefone ou celular do médico abaixo. Esta é uma tarefa de ALTA PRIORIDADE.

**Dados do Médico:**
```json
{dados_atuais_json}
```

**Instruções:**
1. Faça uma busca EXAUSTIVA por números de telefone ou celular deste médico.
2. Verifique sites de clínicas, consultórios, planos de saúde, conselhos regionais.
3. Procure em listagens de profissionais, diretórios médicos, redes sociais.
4. NÃO ACEITE números genéricos ou incompletos.
5. Padronize TODOS os números encontrados no formato:
   - Celular: +55 (DDD) 9XXXX-XXXX
   - Fixo: +55 (DDD) XXXX-XXXX

**Formato de Retorno:**
Retorne APENAS um objeto JSON válido, sem nenhum texto ou explicação adicional. Exemplo:
```json
{{
    "phone_a1": "+55 (XX) XXXX-XXXX",
    "phone_a2": "+55 (XX) XXXX-XXXX",
    "cell_phone_a1": "+55 (XX) 9XXXX-XXXX",
    "cell_phone_a2": "+55 (XX) 9XXXX-XXXX"
}}
```

**IMPORTANTE:**
- Você DEVE encontrar pelo menos um número de contato.
- Não retorne números genéricos ou incompletos.
- Verifique a autenticidade dos números encontrados.
- Se não encontrar números válidos, retorne strings vazias.
"""
    elif iteration == 7:
        prompt = f"""
Você é um especialista em encontrar e-mails profissionais de médicos.

**Tarefa CRÍTICA:** Encontrar e-mails de contato do médico abaixo. Esta é uma tarefa de ALTA PRIORIDADE.

**Dados do Médico:**
```json
{dados_atuais_json}
```

**Instruções:**
1. Faça uma busca EXAUSTIVA por e-mails deste médico.
2. Verifique sites de clínicas, consultórios, planos de saúde.
3. Procure em listagens de profissionais, diretórios médicos.
4. Verifique redes sociais profissionais (LinkedIn, etc).
5. Padronize TODOS os e-mails encontrados:
   - Letras minúsculas.
   - Sem espaços.
   - Remova caracteres especiais desnecessários.

**Formato de Retorno:**
Retorne APENAS um objeto JSON válido, sem nenhum texto ou explicação adicional. Exemplo:
```json
{{
    "email_a1": "email1@exemplo.com",
    "email_a2": "email2@exemplo.com"
}}
```

**IMPORTANTE:**
- Você DEVE encontrar pelo menos um e-mail válido.
- Não retorne e-mails genéricos ou temporários.
- Verifique a autenticidade dos e-mails encontrados.
- Se não encontrar e-mails válidos, retorne strings vazias.
"""
    else:
        prompt = f"""
Você é um especialista em análise de e-mails profissionais.

**Tarefa:** Analise a probabilidade dos e-mails encontrados pertencerem ao médico, baseado nos dados disponíveis.

**Dados do Médico:**
```json
{dados_atuais_json}
```

**Instruções:**
1. Analise cada e-mail separadamente (E-mail A1 e E-mail A2).
2. Compare com os dados do médico (nome, especialidade, localização).
3. Avalie a probabilidade de cada e-mail pertencer ao médico.
4. Retorne um JSON com os e-mails e suas respectivas probabilidades.

**Formato de Retorno:**
Retorne APENAS um objeto JSON válido, sem nenhum texto ou explicação adicional. Exemplo:
```json
{{
    "email1": "email1@exemplo.com",
    "chance_email_a1": "MUITO PROVAVEL|PROVAVEL|NADA PROVAVEL",
    "email2": "email2@exemplo.com",
    "chance_email_a2": "MUITO PROVAVEL|PROVAVEL|NADA PROVAVEL"
}}
```

**Critérios de Avaliação:**
- MUITO PROVAVEL: E-mail segue padrões claros do nome/especialidade.
- PROVAVEL: Há alguma relação, mas não totalmente clara.
- NADA PROVAVEL: E-mail parece genérico ou não relacionado.

**Observações:**
- Analise cada e-mail independentemente.
- Considere o contexto do médico para cada avaliação.
- Se um e-mail estiver vazio, retorne "NADA PROVAVEL" para sua chance.
"""
    
    return prompt

def process_row(row, api_key, email_examples, logger):
    """Processa uma linha usando a API do Gemini."""
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash-preview-04-17"
    
    # Dados iniciais - mantém apenas as colunas originais
    current_data = {
        'Hash': row['Hash'],
        'CRM': row['CRM'],
        'UF': row['UF'],
        'Firstname': row['Firstname'],
        'LastName': row['LastName'],
        'Medical specialty': row['Medical specialty'],
        'Endereco Completo A1': row['Endereco Completo A1'],
        'Address A1': row['Address A1'],
        'Numero A1': row['Numero A1'],
        'Complement A1': row['Complement A1'],
        'Bairro A1': row['Bairro A1'],
        'postal code A1': row['postal code A1'],
        'City A1': row['City A1'],
        'State A1': row['State A1'],
        'Phone A1': row['Phone A1'],
        'Phone A2': row['Phone A2'],
        'Cell phone A1': row['Cell phone A1'],
        'Cell phone A2': row['Cell phone A2'],
        'E-mail A1': row['E-mail A1'],
        'E-mail A2': row['E-mail A2'],
        'OPT-IN': row['OPT-IN'],
        'STATUS': row['STATUS'],
        'LOTE': row['LOTE']
    }
    
    logger.info(f"Iniciando processamento do CRM {row['CRM']}")
    
    # Processa as 9 iterações
    for iteration in range(9):
        logger.info(f"Processando CRM {row['CRM']} - Iteração {iteration + 1}")
        
        # Delay incremental
        if iteration > 0:
            delay = 45 if iteration >= 6 else 7 * iteration
            time.sleep(delay)
        
        # Constrói o prompt para a iteração atual
        prompt_text = build_prompt(current_data, iteration, email_examples, logger)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            tools=tools,
            response_mime_type="text/plain",
        )
        
        max_retries = 5 # Aumentado o número de retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if response is None or response.text is None:
                    logger.warning(f"Resposta da API ou texto da resposta é None para CRM {row['CRM']} (tentativa {retry_count + 1}).")
                    retry_count += 1
                    time.sleep(30) # Aumenta o delay para retries em caso de resposta vazia
                    continue
                
                response_text = response.text
                
                # Tenta encontrar o JSON na resposta usando regex
                json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'({.*?})', response_text, re.DOTALL)

                if json_match:
                    json_str = json_match.group(1)
                    
                    try:
                        new_data = json.loads(json_str)
                        logger.info(f"CRM {row['CRM']} - Iteração {iteration + 1} - JSON recebido:\n{json.dumps(new_data, indent=2, ensure_ascii=False)}")
                        
                        # Atualiza os dados atuais de forma mais robusta
                        if iteration < 6:
                            # Mapeamento de chaves para garantir consistência
                            key_mapping = {
                                'first_name': 'Firstname',
                                'primeiro_nome': 'Firstname',
                                'last_name': 'LastName',
                                'sobrenome': 'LastName',
                                'medical_specialty': 'Medical specialty',
                                'especialidade': 'Medical specialty',
                                'especialidade_medica': 'Medical specialty',
                                'endereco_completo_a1': 'Endereco Completo A1',
                                'logradouro_a1': 'Address A1',
                                'numero_a1': 'Numero A1',
                                'complemento_a1': 'Complement A1',
                                'bairro_a1': 'Bairro A1',
                                'cep_a1': 'postal code A1',
                                'cidade_a1': 'City A1',
                                'estado_a1': 'State A1',
                                'phone_a1': 'Phone A1',
                                'telefone_a1': 'Phone A1',
                                'phone_a2': 'Phone A2',
                                'telefone_a2': 'Phone A2',
                                'cell_phone_a1': 'Cell phone A1',
                                'celular_a1': 'Cell phone A1',
                                'cell_phone_a2': 'Cell phone A2',
                                'celular_a2': 'Cell phone A2',
                                'email_a1': 'E-mail A1',
                                'email_a2': 'E-mail A2'
                            }

                            for json_key, df_key in key_mapping.items():
                                value = new_data.get(json_key)
                                if value is not None and str(value).strip() != '':
                                    current_data[df_key] = value
                        elif iteration == 6:
                            if new_data.get('phone_a1') and str(new_data['phone_a1']).strip() != '':
                                current_data['Phone A1'] = new_data['phone_a1']
                            if new_data.get('phone_a2') and str(new_data['phone_a2']).strip() != '':
                                current_data['Phone A2'] = new_data['phone_a2']
                            if new_data.get('cell_phone_a1') and str(new_data['cell_phone_a1']).strip() != '':
                                current_data['Cell phone A1'] = new_data['cell_phone_a1']
                            if new_data.get('cell_phone_a2') and str(new_data['cell_phone_a2']).strip() != '':
                                current_data['Cell phone A2'] = new_data['cell_phone_a2']
                        elif iteration == 7:
                            if new_data.get('email_a1') and str(new_data['email_a1']).strip() != '':
                                current_data['E-mail A1'] = new_data['email_a1']
                            if new_data.get('email_a2') and str(new_data['email_a2']).strip() != '':
                                current_data['E-mail A2'] = new_data['email_a2']
                        else:
                            current_data['chance_email_a1'] = new_data.get('chance_email_a1', 'NADA PROVAVEL')
                            current_data['chance_email_a2'] = new_data.get('chance_email_a2', 'NADA PROVAVEL')
                        
                        logger.info(f"CRM {row['CRM']} - Iteração {iteration + 1} - Dados atualizados:\n{json.dumps(current_data, indent=2, ensure_ascii=False)}")
                        break
                    except json.JSONDecodeError as je:
                        logger.error(f"Erro ao decodificar JSON para CRM {row['CRM']}: {str(je)}. Resposta original: {response_text}")
                        # Tenta uma limpeza mais agressiva se a primeira falhar
                        try:
                            # Remove qualquer texto antes do primeiro '{' e depois do último '}'
                            clean_json_str = re.search(r'({.*})', response_text, re.DOTALL).group(1)
                            new_data = json.loads(clean_json_str)
                            logger.info(f"CRM {row['CRM']} - JSON limpo com sucesso após erro de decodificação")
                            
                            # Atualiza os dados como antes
                            if iteration < 6:
                                key_mapping = {
                                    'first_name': 'Firstname',
                                    'primeiro_nome': 'Firstname',
                                    'last_name': 'LastName',
                                    'sobrenome': 'LastName',
                                    'medical_specialty': 'Medical specialty',
                                    'especialidade': 'Medical specialty',
                                    'especialidade_medica': 'Medical specialty',
                                    'endereco_completo_a1': 'Endereco Completo A1',
                                    'logradouro_a1': 'Address A1',
                                    'numero_a1': 'Numero A1',
                                    'complemento_a1': 'Complement A1',
                                    'bairro_a1': 'Bairro A1',
                                    'cep_a1': 'postal code A1',
                                    'cidade_a1': 'City A1',
                                    'estado_a1': 'State A1',
                                    'phone_a1': 'Phone A1',
                                    'telefone_a1': 'Phone A1',
                                    'phone_a2': 'Phone A2',
                                    'telefone_a2': 'Phone A2',
                                    'cell_phone_a1': 'Cell phone A1',
                                    'celular_a1': 'Cell phone A1',
                                    'cell_phone_a2': 'Cell phone A2',
                                    'celular_a2': 'Cell phone A2',
                                    'email_a1': 'E-mail A1',
                                    'email_a2': 'E-mail A2'
                                }

                                for json_key, df_key in key_mapping.items():
                                    value = new_data.get(json_key)
                                    if value is not None and str(value).strip() != '':
                                        current_data[df_key] = value
                            elif iteration == 6:
                                if new_data.get('phone_a1') and str(new_data['phone_a1']).strip() != '':
                                    current_data['Phone A1'] = new_data['phone_a1']
                                if new_data.get('phone_a2') and str(new_data['phone_a2']).strip() != '':
                                    current_data['Phone A2'] = new_data['phone_a2']
                                if new_data.get('cell_phone_a1') and str(new_data['cell_phone_a1']).strip() != '':
                                    current_data['Cell phone A1'] = new_data['cell_phone_a1']
                                if new_data.get('cell_phone_a2') and str(new_data['cell_phone_a2']).strip() != '':
                                    current_data['Cell phone A2'] = new_data['cell_phone_a2']
                            elif iteration == 7:
                                if new_data.get('email_a1') and str(new_data['email_a1']).strip() != '':
                                    current_data['E-mail A1'] = new_data['email_a1']
                                if new_data.get('email_a2') and str(new_data['email_a2']).strip() != '':
                                    current_data['E-mail A2'] = new_data['email_a2']
                            else:
                                current_data['chance_email_a1'] = new_data.get('chance_email_a1', 'NADA PROVAVEL')
                                current_data['chance_email_a2'] = new_data.get('chance_email_a2', 'NADA PROVAVEL')
                            
                            logger.info(f"CRM {row['CRM']} - Iteração {iteration + 1} - Dados atualizados após limpeza:\n{json.dumps(current_data, indent=2, ensure_ascii=False)}")
                            break
                        except Exception as e:
                            logger.error(f"Erro crítico ao tentar limpar JSON para CRM {row['CRM']}: {str(e)}. Resposta original: {response_text}")
                            # Se a limpeza agressiva falhar, não há mais o que fazer com esta resposta
                            retry_count += 1
                else:
                    logger.error(f"Não foi possível encontrar JSON na resposta para CRM {row['CRM']}. Resposta original: {response_text}")
                    retry_count += 1
                    
            except Exception as e:
                logger.error(f"Erro ao processar CRM {row['CRM']} (tentativa {retry_count + 1}): {str(e)}", exc_info=True) # Adicionado exc_info=True para stack trace
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(30)
                else:
                    logger.critical(f"Número máximo de tentativas atingido para CRM {row['CRM']}. Dados atuais: {json.dumps(current_data, indent=2, ensure_ascii=False)}")
                    # Se todas as tentativas falharem, retorna os dados atuais (mesmo que incompletos)
                    return current_data
    
    logger.info(f"Processamento concluído para CRM {row['CRM']}")
    return current_data

def process_chunk(chunk, api_key, email_examples, logger):
    """Processa um chunk de linhas usando uma chave da API."""
    results = []
    logger.info(f"Iniciando processamento de chunk com {len(chunk)} registros")
    
    for index, row in chunk.iterrows(): # Adicionado index para melhor log
        try:
            logger.info(f"Iniciando processamento do registro {index} (CRM {row['CRM']}) no chunk")
            result = process_row(row, api_key, email_examples, logger)
            results.append(result)
            logger.debug(f"Registro {index} (CRM {row['CRM']}) processado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao processar registro {index} (CRM {row['CRM']}): {str(e)}")
            logger.debug(f"Stack trace completo do erro para registro {index} (CRM {row['CRM']}):", exc_info=True)
            # Adiciona os dados originais em caso de erro
            results.append(row.to_dict())
            logger.warning(f"Dados originais preservados para registro {index} (CRM {row['CRM']}) devido a erro.\nDados: {json.dumps(row.to_dict(), indent=2, ensure_ascii=False)}")
    
    logger.info(f"Chunk processado: {len(results)} resultados")
    return results

def main():
    # Configura o logging
    logger = setup_logging()
    
    # Gera timestamp para o nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Carregar chaves da API e exemplos de e-mail
        api_keys = load_api_keys(logger)
        email_examples = load_email_examples(logger)
        
        # Ler o CSV
        logger.info("Lendo arquivo input.csv")
        df = pd.read_csv('input.csv')
        logger.info(f"Total de registros carregados: {len(df)}")
        logger.debug(f"Colunas do DataFrame: {df.columns.tolist()}")
        
        # Dividir o DataFrame em chunks para processamento paralelo
        chunk_size = max(1, len(df) // len(api_keys))  # Garante chunk_size mínimo de 1
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        logger.info(f"DataFrame dividido em {len(chunks)} chunks de aproximadamente {chunk_size} registros cada")
        
        # Processar chunks em paralelo
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
            # Garante que temos chaves API suficientes para todos os chunks
            futures = []
            for i, chunk in enumerate(chunks):
                api_key = api_keys[i % len(api_keys)]  # Usa módulo para garantir que temos uma chave válida
                future = executor.submit(process_chunk, chunk, api_key, email_examples, logger)
                futures.append((future, i)) # Armazena o índice do chunk para ordenação
            
            # Coletar resultados na ordem original dos chunks
            # Cria um dicionário para armazenar os resultados dos futures, indexados pelo índice do chunk
            ordered_results = {}
            for future, chunk_index in futures:
                try:
                    ordered_results[chunk_index] = future.result()
                except Exception as e:
                    logger.error(f"Erro ao coletar resultados do chunk {chunk_index}: {str(e)}")
                    # Em caso de erro no chunk, tenta adicionar os dados originais do chunk
                    # Isso pode ser problemático se o erro ocorreu antes do processamento de linhas individuais
                    # Uma abordagem melhor seria passar o chunk original para o future.result() em caso de falha
                    # Por simplicidade, vamos apenas logar e continuar
                    if chunk_index < len(chunks):
                        logger.warning(f"Adicionando dados originais do chunk {chunk_index} devido a erro na coleta de resultados.")
                        all_results.extend(chunks[chunk_index].to_dict('records'))
                    else:
                        logger.error(f"Índice de chunk {chunk_index} fora dos limites para chunks existentes.")

            
            # Concatena os resultados na ordem correta
            for i in range(len(chunks)):
                if i in ordered_results:
                    all_results.extend(ordered_results[i])
                else:
                    # Este caso já é tratado acima, mas mantido para clareza
                    pass

        # Criar DataFrame final e salvar
        final_df = pd.DataFrame(all_results)
        output_filename = f'output_gemini_{timestamp}.csv'
        final_df.to_csv(output_filename, index=False, encoding='utf-8')
        logger.info(f"Processamento concluído. Resultados salvos em {output_filename}")
        
    except Exception as e:
        logger.critical(f"Erro crítico no processo principal: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()


